"""Prepare the Company Documents dataset: PDFs -> page JPGs + docTR OCR TSVs (with Tags) + manifest.

    # 1. download (Apache-2.0, ~9 MB) into data/raw/ and unzip to data/raw/company_docs/
    #    https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset
    python prepare_data.py                       # all 853 bundles (~3.3k pages; ~40 min OCR on CPU)
    python prepare_data.py --limit-bundles 40    # quick subset
    python prepare_data.py --fast-ocr            # docTR mobile models (~2x faster)

Output (data/company_docs/): pages/*.jpg|*.tsv, labels/*.json (gold field values per document),
ocr_cache/ (resumable), manifest.jsonl, label_coverage.json. Re-running reuses cached OCR.
"""
import argparse
import json
from collections import Counter

from lmv3docs.company_docs import prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/company_docs/CompanyDocuments")
    ap.add_argument("--out", default="data/company_docs")
    ap.add_argument("--limit-bundles", type=int, default=None)
    ap.add_argument("--fast-ocr", action="store_true", help="db_mobilenet_v3_large + crnn_mobilenet_v3_small")
    args = ap.parse_args()
    det, reco = ("db_mobilenet_v3_large", "crnn_mobilenet_v3_small") if args.fast_ocr else ("fast_base", "crnn_vgg16_bn")
    rows, cov = prepare(args.raw, args.out, det=det, reco=reco, limit_bundles=args.limit_bundles)

    bundles = {r["doc_id"] for r in rows}
    print(f"\n{len(bundles)} bundles, {len(rows)} pages -> {args.out}/manifest.jsonl")
    print("pages per doc type:", dict(Counter(r["doc_type"] for r in rows)))
    print("bundles per split:", dict(Counter({r["doc_id"]: r["split"] for r in rows}.values())))
    low = {k: round(v["rate"], 3) for k, v in cov.items() if v["rate"] < 0.95}
    print(f"label coverage (gold value recovered in OCR tags): mean {sum(v['rate'] for v in cov.values()) / len(cov):.3f}")
    print("fields below 95%:", json.dumps(low, indent=1) if low else "none")


if __name__ == "__main__":
    main()
