"""Evaluation: page/document classification and field extraction, plus latency."""
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .classifier import bundle_targets
from .extractor import aggregate, group_spans


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_classification(pipe, bundles, ccfg):
    """Page accuracy, macro/weighted F1, per-class P/R/F1, first-page-of-segment accuracy,
    boundary F1, document exact match, confusion matrix and per-page latency."""
    labels = pipe.classifier.labels
    y_t, y_p, first_t, first_p, bt, bp, exact, ms = [], [], [], [], [], [], [], []
    for pages in bundles:
        y, starts, _ = bundle_targets(pages, labels, ccfg)
        t0 = time.perf_counter()
        probs, bprob = pipe.classify(pages)
        ms.append((time.perf_counter() - t0) * 1000 / len(pages))
        pred = probs.argmax(-1).tolist()
        pstart = [True] + [bool(bprob[i] >= 0.5 or pred[i] != pred[i - 1]) for i in range(1, len(pred))]
        y_t += y; y_p += pred; exact.append(y == pred)
        first_t += [a for a, s in zip(y, starts) if s]; first_p += [b for b, s in zip(pred, starts) if s]
        bt += [bool(s) for s in starts[1:]]; bp += pstart[1:]
    cm = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_t, y_p):
        cm[t, p] += 1
    rows = []
    for i, l in enumerate(labels):
        p, r, f = prf(cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i])
        rows.append({"doc_type": l, "precision": p, "recall": r, "f1": f, "support": int(cm[i].sum())})
    report = pd.DataFrame(rows).set_index("doc_type")
    sup = report.support > 0
    summary = {
        "page_accuracy": float(np.mean(np.array(y_t) == np.array(y_p))),
        "macro_f1": float(report.loc[sup, "f1"].mean()),
        "weighted_f1": float(np.average(report.f1, weights=report.support)),
        "first_page_of_segment_accuracy": float(np.mean(np.array(first_t) == np.array(first_p))),
        "boundary_f1": prf(sum(a and b for a, b in zip(bt, bp)), sum(b and not a for a, b in zip(bt, bp)),
                           sum(a and not b for a, b in zip(bt, bp)))[2],
        "document_exact_match": float(np.mean(exact)),
        "bundles": len(bundles), "pages": len(y_t),
    }
    return summary, report, cm, ms


def _norm(s):
    return re.sub(r"[^\w./-]", "", re.sub(r"\s+", " ", s.lower().strip()))


def evaluate_extraction(pipe, bundles):
    """Per field: strict span F1 and normalised-text F1; per document: chosen-value accuracy.
    Runs each gold sub-document through its own doc type's extractor (isolates extraction quality)."""
    strict, loose, doc_acc, ms = defaultdict(Counter), defaultdict(Counter), defaultdict(list), defaultdict(list)
    for bundle in bundles:
        segs = []
        for i, p in enumerate(bundle):
            if not segs or p.meta.get("segment_start") or p.doc_type != bundle[i - 1].doc_type:
                segs.append([])
            segs[-1].append(p)
        for seg in segs:
            dt = seg[0].doc_type
            ext = pipe.extractors.get(dt)
            if ext is None:
                continue
            split = {f["name"] for f in ext.specs if f["split_lines"]}
            page_ents, gold_vals = [], defaultdict(list)
            for p in seg:
                t0 = time.perf_counter()
                ents = pipe.extract(p, dt)
                ms[dt].append((time.perf_counter() - t0) * 1000)
                page_ents.append((p.meta.get("page_num"), ents))
                gold = [(f, tuple(i)) for f, i in group_spans(ext.word_tags(p), [w["box"] for w in p.words], split)]
                text = lambda idx: _norm(" ".join(p.words[i]["text"] for i in idx))
                for f, idx in gold:
                    gold_vals[f].append(text(idx))
                for g, q, store in ((Counter(gold), Counter((e["field"], tuple(e["word_idx"])) for e in ents), strict),
                                    (Counter((f, text(i)) for f, i in gold), Counter((e["field"], _norm(e["value"])) for e in ents), loose)):
                    for key in set(g) | set(q):
                        tp = min(g[key], q[key])
                        store[(dt, key[0])].update(tp=tp, fp=q[key] - tp, fn=g[key] - tp)
            agg = aggregate(page_ents, ext.specs)
            for f in ext.specs:
                if not f["multi"]:
                    v, gv = agg[f["name"]], gold_vals.get(f["name"], [])
                    doc_acc[(dt, f["name"])].append((v is not None and _norm(v["value"]) in gv) if gv else v is None)
    rows = []
    for (dt, f), c in sorted(strict.items()):
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        lc = loose[(dt, f)]
        rows.append({"doc_type": dt, "field": f, "precision": p, "recall": r, "f1_strict": f1,
                     "f1_text": prf(lc["tp"], lc["fp"], lc["fn"])[2], "support": c["tp"] + c["fn"],
                     "doc_value_acc": float(np.mean(doc_acc[(dt, f)])) if (dt, f) in doc_acc else np.nan})
    per_field = pd.DataFrame(rows).set_index(["doc_type", "field"])
    summary = {}
    for dt in pipe.extractors:
        tot = sum((c for k, c in strict.items() if k[0] == dt), Counter())
        ltot = sum((c for k, c in loose.items() if k[0] == dt), Counter())
        accs = [np.mean(v) for k, v in doc_acc.items() if k[0] == dt]
        summary[dt] = {"entity_f1_strict": prf(tot["tp"], tot["fp"], tot["fn"])[2],
                       "entity_f1_text": prf(ltot["tp"], ltot["fp"], ltot["fn"])[2],
                       "doc_value_accuracy": float(np.mean(accs)) if accs else np.nan, "pages": len(ms[dt])}
    return pd.DataFrame(summary).T, per_field, ms


def latency_table(**series):
    rows = {}
    for name, ms in series.items():
        a = np.asarray(ms, dtype=float)
        rows[name] = {"mean_ms": a.mean(), "p50_ms": np.percentile(a, 50), "p95_ms": np.percentile(a, 95), "pages_per_s": 1000 / a.mean()}
    return pd.DataFrame(rows).T
