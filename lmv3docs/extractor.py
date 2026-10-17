"""Stage 2b: one field extractor per doc type = LoRA 'ie_<type>' on the frozen backbone + linear tag head.

Labels are plain word tags ("O" or a field name, from the OCR TSV Tag column) on each word's
first sub-token. Long pages are split into overlapping windows; each word keeps the prediction
from the window where it sits farthest from an edge. Runs of equal tags become entities (for
multi-valued fields a new text line starts a new value), and entities are merged per document.
Doc types are fully decoupled: adding or retraining one never touches the others.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import collate, to_device
from .lora import adapter_params
from .training import autocast, make_optimizer, step


class FieldExtractor(nn.Module):
    def __init__(self, hidden, doc_type, field_specs, dropout=0.1):
        super().__init__()
        self.doc_type, self.specs = doc_type, field_specs
        self.labels = ["O"] + [f["name"] for f in field_specs]
        self.adapter = f"ie_{doc_type}"
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(self.labels)))

    def logits(self, backbone, batch):
        backbone.set_adapter(self.adapter)
        text_h, _, _ = backbone(**batch)
        return self.head(text_h.to(self.head[1].weight.dtype)).float()

    def word_tags(self, page):
        canon = {l.lower(): l for l in self.labels}
        return [canon.get(w["tag"].lower(), "O") for w in page.words]

    def items(self, page, overlap):
        l2i = {l: i for i, l in enumerate(self.labels)}
        tags = self.word_tags(page)
        first = [w is not None and (t == 0 or page.wids[t - 1] != w) for t, w in enumerate(page.wids)]
        out = []
        for s, e in page.windows(overlap):
            it = page.window(s, e)
            it["pixel_values"] = page.image()
            it["labels"] = [-100] + [l2i[tags[w]] if f else -100 for w, f in zip(page.wids[s:e], first[s:e])] + [-100]
            out.append(it)
        return out


def train_extractor(backbone, ext, pages, cfg, device, log=print):
    items = [it for p in pages for it in ext.items(p, cfg["window_overlap"])]
    if not items:
        raise ValueError(f"No training pages for {ext.doc_type}")
    cw = torch.ones(len(ext.labels), device=device)
    cw[0] = cfg["o_weight"]
    params_lora, params_head = adapter_params(backbone.model.encoder, ext.adapter), list(ext.parameters())
    n_steps = cfg["epochs"] * math.ceil(len(items) / cfg["batch_size"])
    opt, sched = make_optimizer([{"params": params_lora, "lr": cfg["lr"]}, {"params": params_head, "lr": cfg["head_lr"]}], n_steps)
    rng, history, t0 = np.random.default_rng(13), {"loss": []}, time.time()
    pad = pages[0].fz.pad
    for _ in range(cfg["epochs"]):
        backbone.train(); ext.train()
        order, losses = rng.permutation(len(items)), []
        for i in range(0, len(order), cfg["batch_size"]):
            b = to_device(collate([items[j] for j in order[i:i + cfg["batch_size"]]], pad), device)
            with autocast(device):
                logits = ext.logits(backbone, b)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b["labels"].reshape(-1), weight=cw, ignore_index=-100)
            step(loss, params_lora + params_head, opt, sched)
            losses.append(loss.item())
        history["loss"].append(float(np.mean(losses)))
    history["seconds"], history["windows"] = time.time() - t0, len(items)
    if log:
        log(f"[{ext.doc_type}] {len(items)} windows x {cfg['epochs']} epochs  final loss {history['loss'][-1]:.3f}  ({history['seconds']:.0f}s)")
    return history


def _new_line(b0, b1):
    c0, c1 = (b0[1] + b0[3]) / 2, (b1[1] + b1[3]) / 2
    return abs(c1 - c0) > 0.6 * max(b0[3] - b0[1], b1[3] - b1[1], 1)


def group_spans(tags, boxes, split_fields=()):
    """Plain tags -> [(field, [word positions])]: runs of equal tags, split at new lines for split_fields."""
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t == "O":
            cur = None
            continue
        if cur is not None and cur[0] == t and not (t in split_fields and _new_line(boxes[i - 1], boxes[i])):
            cur[1].append(i)
        else:
            cur = (t, [i])
            spans.append(cur)
    return spans


@torch.no_grad()
def extract_page(backbone, ext, page, device, overlap=64):
    backbone.eval(); ext.eval()
    n = len(page.words)
    if n == 0:
        return []
    probs, best = torch.zeros(n, len(ext.labels)), torch.full((n,), -1)
    wins = page.windows(overlap)
    items = [dict(page.window(s, e), pixel_values=page.image()) for s, e in wins]
    with autocast(device):
        pr = ext.logits(backbone, to_device(collate(items, page.fz.pad), device)).softmax(-1).cpu()
    for k, (s, e) in enumerate(wins):
        prev = None
        for t, w in enumerate(page.wids[s:e]):
            if w is not None and w != prev:
                dist = min(t, e - s - 1 - t)
                if dist > best[w]:
                    best[w], probs[w] = dist, pr[k, t + 1]
            prev = w
    conf, pred = probs.max(-1)
    tags = [ext.labels[i] for i in pred.tolist()]
    boxes = [w["box"] for w in page.words]
    split = {f["name"] for f in ext.specs if f["split_lines"]}
    return [{"field": f, "value": " ".join(page.words[i]["text"] for i in idx), "word_idx": idx,
             "confidence": float(conf[idx].mean()),
             "bbox": [min(boxes[i][0] for i in idx), min(boxes[i][1] for i in idx),
                      max(boxes[i][2] for i in idx), max(boxes[i][3] for i in idx)]}
            for f, idx in group_spans(tags, boxes, split)]


def aggregate(page_entities, field_specs):
    """[(page_num, entities)] -> {field: best value} (single) or {field: [values]} (multi)."""
    keep = ("value", "confidence", "page", "bbox")
    out = {}
    for f in field_specs:
        c = [dict(e, page=pn) for pn, ents in page_entities for e in ents if e["field"] == f["name"]]
        if f["multi"]:
            out[f["name"]] = [{k: e[k] for k in keep} for e in c]
        else:
            out[f["name"]] = {k: max(c, key=lambda e: e["confidence"])[k] for k in keep} if c else None
    return out
