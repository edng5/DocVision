"""Ablations and audits used by the notebook.

run_variant  trains one complete pipeline variant on the same pages and returns test metrics:
               - "full":          MLM + MIM domain pretraining, then LoRA tasks
               - "no_pretrain":   public LayoutLMv3 weights, no domain pretraining (backbone never fine-tuned)
               - "mlm_only":      domain pretraining with MLM, without MIM
               - "mim_only":      domain pretraining with MIM, without MLM
             Downstream training (classifier + every extractor) is identical across variants.
audit        split integrity (bundle overlap) and memorisation check (accuracy on test values that
             never occur in training).
"""
import copy
import time
from collections import defaultdict

import numpy as np
import torch

from .classifier import train_classifier
from .extractor import group_spans, train_extractor
from .metrics import evaluate_classification, evaluate_extraction
from .pipeline import DocPipeline
from .pretraining import Pretrainer

VARIANTS = {
    "full": {"pretrain": True, "mlm_weight": 1.0, "mim_weight": 1.0},
    "no_pretrain": {"pretrain": False},
    "mlm_only": {"pretrain": True, "mlm_weight": 1.0, "mim_weight": 0.0},
    "mim_only": {"pretrain": True, "mlm_weight": 0.0, "mim_weight": 1.0},
}


def summarize(cls_sum, ie_sum, pretrain_val=None, seconds=None):
    """One row of the ablation table."""
    row = {"page_accuracy": cls_sum["page_accuracy"], "macro_f1": cls_sum["macro_f1"],
           "boundary_f1": cls_sum["boundary_f1"], "document_exact_match": cls_sum["document_exact_match"],
           "entity_f1_strict": float(ie_sum["entity_f1_strict"].mean()),
           "entity_f1_text": float(ie_sum["entity_f1_text"].mean()),
           "doc_value_accuracy": float(ie_sum["doc_value_accuracy"].mean())}
    for k, v in (pretrain_val or {}).items():
        row[f"pretrain_val_{k}"] = v
    if seconds is not None:
        row["train_minutes"] = seconds / 60
    return row


def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_variant(name, cfg, pages, splits, specs, device, seed=13, log=print):
    """Train + evaluate one variant from scratch. pages/splits: already-built Page objects (reused)."""
    v = VARIANTS[name]
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    pipe = DocPipeline.create(cfg, device)
    pretrain_val = None
    if v["pretrain"]:
        pcfg = dict(copy.deepcopy(cfg["pretrain"]), mlm_weight=v["mlm_weight"], mim_weight=v["mim_weight"])
        pt = Pretrainer(pipe.backbone, pipe.fz, pcfg, device, seed=seed)
        pt.train(pages["train"], log=log)
        pretrain_val = pt.evaluate(pages["val"])
        pt.heads.cpu()
        del pt
    pipe.backbone.freeze()
    free_gpu()
    clf = pipe.add_classifier(cfg["classifier"]["doc_types"], cfg["classifier"])
    train_classifier(pipe.backbone, clf, splits["train"], splits["val"], cfg["classifier"], device, log=None)
    free_gpu()
    for dt in specs:
        tp = [p for p in pages["train"] if p.doc_type == dt]
        if tp:
            train_extractor(pipe.backbone, pipe.add_extractor(dt, specs[dt]), tp, cfg["extraction"], device, log=None)
            free_gpu()
    seconds = time.time() - t0
    cls_sum, _, _, _ = evaluate_classification(pipe, splits["test"], cfg["classifier"])
    ie_sum, _, _ = evaluate_extraction(pipe, splits["test"])
    row = summarize(cls_sum, ie_sum, pretrain_val, seconds)
    if log:
        log(f"[{name}] page acc {row['page_accuracy']:.4f} | entity F1 {row['entity_f1_strict']:.4f} | "
            f"doc value acc {row['doc_value_accuracy']:.4f} | {seconds / 60:.1f} min")
    pipe.backbone.cpu()
    del pipe
    free_gpu()
    return row, ie_sum


# ----------------------------------------------------------------------------- audit
def split_integrity(pages_all):
    by_bundle = defaultdict(set)
    for p in pages_all:
        by_bundle[(p.meta.get("dataset"), p.meta["doc_id"])].add(p.meta["split"])
    return {"bundles": len(by_bundle), "bundles_in_more_than_one_split": sum(len(s) > 1 for s in by_bundle.values())}


def _entities(page, ext):
    split_f = {f["name"] for f in ext.specs if f["split_lines"]}
    tags = ext.word_tags(page)
    return [(f, idx, " ".join(page.words[i]["text"].lower() for i in idx))
            for f, idx in group_spans(tags, [w["box"] for w in page.words], split_f)]


def _norm(s):
    import re
    return re.sub(r"[^0-9a-z.]", "", s.lower())


def end_to_end_values(pipe, test_pages):
    """Business-level accuracy on noisy pages: is the extracted value equal to the TRUE value from
    the clean source page (same file stem in the dataset dir without "_noisy")?

    For every true (field, value) on a test page:
      model    = some predicted entity of that field has the same normalised value
      ceiling  = the noisy page's OCR words at that field (labels transferred by geometry) spell the
                 true value, i.e. the best any extractor could do on top of this OCR.
    """
    import os
    from .ocr import read_tsv
    res = defaultdict(lambda: [0, 0, 0])            # doc_type -> [model correct, ocr ceiling, total]
    for p in test_pages:
        ext = pipe.extractors.get(p.doc_type)
        clean_dir = p.meta["_dir"].replace("_noisy", "")
        clean_tsv = os.path.join(clean_dir, "pages", os.path.basename(p.meta["ocr"]))
        if not ext or not os.path.exists(clean_tsv):
            continue
        split_f = {f["name"] for f in ext.specs if f["split_lines"]}
        canon = {l.lower(): l for l in ext.labels}

        def spans(words):
            tags = [canon.get(w["tag"].lower(), "O") for w in words]
            return [(f, _norm(" ".join(words[i]["text"] for i in idx)))
                    for f, idx in group_spans(tags, [w["box"] for w in words], split_f)]

        truth = spans(read_tsv(clean_tsv))
        ocr_vals = defaultdict(set)
        for f, v in spans(p.words):
            ocr_vals[f].add(v)
        pred_vals = defaultdict(set)
        for e in pipe.extract(p, p.doc_type):
            pred_vals[e["field"]].add(_norm(e["value"]))
        for f, v in truth:
            r = res[p.doc_type]
            r[0] += v in pred_vals[f]
            r[1] += v in ocr_vals[f]
            r[2] += 1
    rows = {dt: {"value_accuracy": a / n, "ocr_ceiling": c / n, "of_ceiling": a / max(c, 1), "values": n}
            for dt, (a, c, n) in res.items()}
    tot = [sum(x[i] for x in res.values()) for i in range(3)]
    rows["ALL"] = {"value_accuracy": tot[0] / tot[2], "ocr_ceiling": tot[1] / tot[2], "of_ceiling": tot[0] / max(tot[1], 1),
                   "values": tot[2]}
    return rows


def seen_vs_unseen(pipe, pages):
    """Exact-span recall on test entities, split by whether the same value occurs in training
    (same doc type and field). High recall on never-seen values = the model reads, not memorises."""
    train_vals = defaultdict(set)
    for p in pages["train"]:
        ext = pipe.extractors.get(p.doc_type)
        if ext:
            for f, _, v in _entities(p, ext):
                train_vals[(p.doc_type, f)].add(v)
    res = defaultdict(lambda: [0, 0])
    for p in pages["test"]:
        ext = pipe.extractors.get(p.doc_type)
        if not ext:
            continue
        pred = {(e["field"], tuple(e["word_idx"])) for e in pipe.extract(p, p.doc_type)}
        for f, idx, v in _entities(p, ext):
            key = (p.meta.get("dataset"), "seen in train" if v in train_vals[(p.doc_type, f)] else "never in train")
            res[key][0] += (f, tuple(idx)) in pred
            res[key][1] += 1
    return {k: {"correct": a, "total": b, "recall": a / b if b else float("nan")} for k, (a, b) in sorted(res.items())}
