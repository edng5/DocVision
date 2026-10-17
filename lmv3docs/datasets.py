"""Load one or more datasets (manifests) into a single training set.

CONFIG.yml `data.manifests` lists them, e.g. the prepared Company Documents and the synthetic bank
bundles. Doc types and doc_ids from different datasets never collide, so they simply concatenate:
one classifier learns every page type, and each doc type keeps its own extractor.
"""
import os

from .data import load_manifest


def manifest_paths(cfg):
    m = cfg["data"].get("manifests") or cfg["data"].get("manifest")
    return [m] if isinstance(m, str) else list(m)


def ensure_datasets(cfg, synthetic_ok=True, log=print):
    """Create missing datasets where possible: synthetic data is generated on the fly; the Company
    Documents set must be prepared with `python prepare_data.py` (docTR OCR)."""
    for path in manifest_paths(cfg):
        if os.path.exists(path):
            continue
        if "_noisy" in path:
            raise FileNotFoundError(f"{path} not found. Run `python prepare_noisy.py` first.")
        if "synthetic" in path and synthetic_ok:
            from .synthetic import generate_dataset
            s = cfg["data"]["synthetic"]
            if log:
                log(f"{path} not found -> generating {s['n_bundles']} synthetic bundles")
            generate_dataset(os.path.dirname(path), s["n_bundles"], s["seed"], tuple(cfg["data"]["split_ratios"]), s["max_rows"])
        else:
            raise FileNotFoundError(f"{path} not found. Run `python prepare_data.py` first.")


def load_rows(cfg, keep_types=None):
    """All manifest rows, tagged with their dataset name; optionally only some doc types."""
    rows = []
    for path in manifest_paths(cfg):
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        for r in load_manifest(path, tuple(cfg["data"]["split_ratios"])):
            r["dataset"] = name.replace("_noisy", "")          # company_docs | synthetic
            r["variant"] = "noisy" if name.endswith("_noisy") else "clean"
            rows.append(r)
    if keep_types is not None:
        rows = [r for r in rows if r["doc_type"] in keep_types]
    return rows
