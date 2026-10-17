"""Make "bad scan" versions of the prepared datasets and re-OCR them with docTR.

    python prepare_noisy.py                    # company_docs + synthetic -> *_noisy (about 25 min on a GPU)
    python prepare_noisy.py --only synthetic

Each page gets a random degradation (severity 0.2-1.0): perspective skew/rotation, uneven
lighting, faded ink, stains, fold lines, resolution loss, blur, sensor noise, JPEG compression.
Labels follow the known warp, and the noisy page is OCR'd again, so OCR errors are part of the data.
Splits, bundles and doc types are unchanged. Writes data/<name>_noisy/{pages,manifest.jsonl,noise_coverage.json}.
"""
import argparse

from lmv3docs.noise import build_noisy_dataset

SOURCES = {"company_docs": "data/company_docs/manifest.jsonl", "synthetic": "data/synthetic/manifest.jsonl"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(SOURCES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fast-ocr", action="store_true", help="docTR mobile models")
    args = ap.parse_args()
    det, reco = ("db_mobilenet_v3_large", "crnn_mobilenet_v3_small") if args.fast_ocr else ("fast_base", "crnn_vgg16_bn")
    for name, src in SOURCES.items():
        if args.only and name != args.only:
            continue
        rows, cov = build_noisy_dataset(src, f"data/{name}_noisy", seed=args.seed, det=det, reco=reco)
        tot = sum(v["total"] for v in cov.values())
        read = sum(v["found"] for v in cov.values()) / tot
        tagged = sum(v["tagged_rate"] * v["total"] for v in cov.values()) / tot
        print(f"{name}: {len(rows)} noisy pages | gold values read verbatim by OCR {read:.3f} | field labels kept {tagged:.3f}")


if __name__ == "__main__":
    main()
