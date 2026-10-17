"""Train the full pipeline, or add/retrain single task models on an existing one.

    python train.py                                   # pretrain -> classifier -> all extractors -> evaluate -> save
    python train.py --set pretrain.steps=500 --set extraction.epochs=10
    python train.py --from artifacts/pipeline --extractor invoice    # retrain ONE extractor; nothing else changes
    python train.py --from artifacts/pipeline --classifier           # retrain only the classifier
    python train.py --tiny --synthetic ...                           # tiny random backbone on synthetic data (smoke tests)

Data: run `python prepare_data.py` first (Company Documents -> data/company_docs/manifest.jsonl).
"""
import argparse
import json
import os

from lmv3docs.config import all_field_specs, load_config
from lmv3docs.data import Featurizer, build_bundles, split_bundles
from lmv3docs.datasets import ensure_datasets, load_rows
from lmv3docs.extractor import train_extractor
from lmv3docs.classifier import train_classifier
from lmv3docs.lora import remove_adapter
from lmv3docs.metrics import evaluate_classification, evaluate_extraction, latency_table
from lmv3docs.pipeline import DocPipeline
from lmv3docs.pretraining import Pretrainer
from lmv3docs.training import get_device


def load_pages(cfg, fz, synthetic=True):
    ensure_datasets(cfg, synthetic_ok=synthetic)
    rows = load_rows(cfg, keep_types=set(cfg["classifier"]["doc_types"]))
    return split_bundles(build_bundles([fz.page_from_row(r, r["_dir"]) for r in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="CONFIG.yml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--from", dest="from_dir", help="existing pipeline dir: keep its backbone, train only what is asked")
    ap.add_argument("--extractor", action="append", default=[], help="doc type(s) to (re)train with --from")
    ap.add_argument("--classifier", action="store_true", help="(re)train the classifier with --from")
    ap.add_argument("--tiny", action="store_true", help="tiny random backbone instead of layoutlmv3-base")
    ap.add_argument("--synthetic", action="store_true", help="(kept for compatibility; synthetic data is generated when missing)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = get_device(args.device)
    specs = all_field_specs(cfg)

    if args.from_dir:
        pipe = DocPipeline.load(args.from_dir, device)
        pipe.fz.cache_images = cfg["data"].get("cache_images", False)
        todo_clf, todo_ie = args.classifier, args.extractor
    else:
        pipe = DocPipeline.create(cfg, device, tiny=args.tiny)
        todo_clf, todo_ie = True, list(specs)
    splits = load_pages(cfg, pipe.fz)
    print({k: len(v) for k, v in splits.items()}, "bundles per split")

    if not args.from_dir:                                       # Stage 1: pretrain the backbone
        pages = lambda s: [p for b in splits[s] for p in b]
        pt = Pretrainer(pipe.backbone, pipe.fz, cfg["pretrain"], device)
        print("before pretraining (val):", pt.evaluate(pages("val")))
        pt.train(pages("train"))
        print("after pretraining (val): ", pt.evaluate(pages("val")))
        pipe.backbone.freeze()                                  # frozen from here on

    if todo_clf:                                                # Stage 2a: classifier
        if pipe.classifier is not None:
            remove_adapter(pipe.backbone.model.encoder, "cls")
        clf = pipe.add_classifier(cfg["classifier"]["doc_types"], cfg["classifier"])
        train_classifier(pipe.backbone, clf, splits["train"], splits["val"], cfg["classifier"], device)

    for dt in todo_ie:                                          # Stage 2b: extractors (independent)
        train_pages = [p for b in splits["train"] for p in b if p.doc_type == dt]
        if not train_pages:
            print(f"[{dt}] no training pages - skipped")
            continue
        if dt in pipe.extractors:
            remove_adapter(pipe.backbone.model.encoder, pipe.extractors.pop(dt).adapter)
        ext = pipe.add_extractor(dt, specs[dt])
        train_extractor(pipe.backbone, ext, train_pages, cfg["extraction"], device)

    out = cfg["output_dir"] if not args.from_dir else args.from_dir
    pipe.save(out, fp16=True)                                  # backbone + adapters + heads in fp16
    print("saved pipeline ->", out)

    cls_sum, cls_rep, _, cls_ms = evaluate_classification(pipe, splits["test"], cfg["classifier"])
    ie_sum, ie_rep, ie_ms = evaluate_extraction(pipe, splits["test"])
    print("\nclassification (test):", json.dumps({k: round(v, 4) for k, v in cls_sum.items()}))
    print(cls_rep.round(3).to_string())
    print("\nextraction (test):\n" + ie_sum.round(3).to_string())
    print("\nlatency:\n" + latency_table(classify=cls_ms, extract=[x for v in ie_ms.values() for x in v]).round(1).to_string())
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"classification": cls_sum, "extraction": ie_sum.to_dict(orient="index")}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
