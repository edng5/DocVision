"""Stage 2a: document classifier = LoRA 'cls' on the frozen backbone + Transformer over pages.

page vector = [text CLS ; mean of text tokens ; image CLS] -> Linear(d) + page-position embedding
            -> Transformer (or BiLSTM) across ALL pages of the bundle
            -> per-page doc type  +  per-page boundary ("this page starts a new sub-document")
Every page gets its own label, but each decision sees the whole bundle, which handles uploads
that contain several documents. First pages get extra loss weight: their headers carry most of
the type signal, and errors there break document splitting.
"""
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import collate, to_device
from .lora import adapter_params
from .training import autocast, make_optimizer, step

ADAPTER = "cls"


class DocClassifier(nn.Module):
    def __init__(self, hidden, labels, cfg):
        super().__init__()
        d = cfg["dim"]
        self.labels, self.cfg = list(labels), dict(cfg)
        self.proj = nn.Sequential(nn.LayerNorm(3 * hidden), nn.Linear(3 * hidden, d), nn.GELU(), nn.Dropout(cfg["dropout"]))
        self.pos = nn.Embedding(cfg["max_pages"], d)
        if cfg["doc_encoder"] == "transformer":
            layer = nn.TransformerEncoderLayer(d, cfg["heads"], 4 * d, cfg["dropout"], activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        elif cfg["doc_encoder"] == "bilstm":
            self.enc = nn.LSTM(d, d // 2, cfg["layers"], batch_first=True, bidirectional=True)
        else:
            raise ValueError(cfg["doc_encoder"])
        self.norm = nn.LayerNorm(d)
        self.page_head, self.bnd_head = nn.Linear(d, len(labels)), nn.Linear(d, 1)

    def page_features(self, backbone, pages, chunk=8):
        backbone.set_adapter(ADAPTER)
        dev = self.pos.weight.device
        feats = []
        for i in range(0, len(pages), chunk):
            b = to_device(collate([p.first_window() for p in pages[i:i + chunk]], pages[0].fz.pad), dev)
            text_h, img_cls, _ = backbone(**b)
            m = b["attention_mask"].unsqueeze(-1).to(text_h.dtype)
            f = torch.cat([text_h[:, 0], (text_h * m).sum(1) / m.sum(1), img_cls], -1)
            feats.append(self.proj(f.to(self.proj[1].weight.dtype)))
        return torch.cat(feats)

    def forward(self, backbone, pages, offset=0):
        dev = self.pos.weight.device
        pos = torch.arange(offset, offset + len(pages), device=dev).clamp_max(self.cfg["max_pages"] - 1)
        x = (self.page_features(backbone, pages) + self.pos(pos)).unsqueeze(0)
        x = self.enc(x) if isinstance(self.enc, nn.TransformerEncoder) else self.enc(x)[0]
        x = self.norm(x)[0]
        return self.page_head(x), self.bnd_head(x).squeeze(-1)


def bundle_targets(pages, labels, cfg):
    """Per page: label id, starts-a-segment flag, and loss weight max(min, first * decay^p)."""
    l2i = {l: i for i, l in enumerate(labels)}
    y, starts, weights, p = [], [], [], 0
    for i, pg in enumerate(pages):
        st = bool(pg.meta.get("segment_start")) or i == 0 or pg.doc_type != pages[i - 1].doc_type
        p = 0 if st else p + 1
        y.append(l2i[pg.doc_type])
        starts.append(float(st))
        weights.append(max(cfg["min_weight"], cfg["first_page_weight"] * cfg["decay"] ** p))
    return y, starts, weights


def train_classifier(backbone, clf, train_bundles, val_bundles, cfg, device, log=print):
    labels = clf.labels
    cnt = Counter(p.doc_type for b in train_bundles for p in b)
    class_w = torch.tensor([1 / math.sqrt(max(1, cnt[l])) for l in labels], device=device)
    class_w /= class_w.mean()
    params_lora, params_head = adapter_params(backbone.model.encoder, ADAPTER), list(clf.parameters())
    opt, sched = make_optimizer([{"params": params_lora, "lr": cfg["lr"]}, {"params": params_head, "lr": cfg["head_lr"]}],
                                cfg["epochs"] * len(train_bundles))
    rng, history, t0 = np.random.default_rng(13), {"loss": [], "val_acc": []}, time.time()
    for ep in range(cfg["epochs"]):
        clf.train(); backbone.train()
        losses = []
        for bi in rng.permutation(len(train_bundles)):
            pages, lo = train_bundles[bi], 0
            if len(pages) > cfg["train_max_pages"]:                 # random page window (memory)
                lo = 0 if rng.random() < 0.5 else int(rng.integers(0, len(pages) - cfg["train_max_pages"] + 1))
                pages = pages[lo:lo + cfg["train_max_pages"]]
            y, starts, w = bundle_targets(pages, labels, cfg)
            with autocast(device):
                logits, bnd = clf(backbone, pages, lo)
            y, w = torch.tensor(y, device=device), torch.tensor(w, device=device) * class_w[torch.tensor(y, device=device)]
            ce = F.cross_entropy(logits.float(), y, reduction="none", label_smoothing=cfg["label_smoothing"])
            loss = (ce * w).sum() / w.sum()
            bm = torch.ones(len(pages), dtype=torch.bool, device=device)
            if lo > 0:
                bm[0] = False                                         # no left context for this page
            if bm.any():
                loss = loss + cfg["boundary_weight"] * F.binary_cross_entropy_with_logits(
                    bnd.float()[bm], torch.tensor(starts, device=device)[bm])
            step(loss, params_lora + params_head, opt, sched)
            losses.append(loss.item())
        history["loss"] += losses
        acc = np.mean([(classify_bundle(backbone, clf, b, device)[0].argmax(-1).numpy()
                        == np.array(bundle_targets(b, labels, cfg)[0])).mean() for b in val_bundles]) if val_bundles else float("nan")
        history["val_acc"].append(acc)
        if log:
            log(f"classifier epoch {ep + 1}/{cfg['epochs']}  loss {np.mean(losses):.3f}  val page-acc {acc:.3f}  ({time.time() - t0:.0f}s)")
    history["seconds"] = time.time() - t0
    return history


@torch.no_grad()
def classify_bundle(backbone, clf, pages, device):
    """-> (probs (P, C), boundary probs (P,)). Documents longer than max_pages use clamped positions."""
    clf.eval(); backbone.eval()
    with autocast(device):
        logits, bnd = clf(backbone, pages)
    probs, bprob = logits.float().softmax(-1).cpu(), bnd.float().sigmoid().cpu()
    bprob[0] = 1.0
    return probs, bprob
