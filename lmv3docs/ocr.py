"""OCR TSV in the AWS Textract layout -> words with 0-1000 boxes and tags.

    BlockType  Page  Text      Confidence  Left    Top     Width   Height  Tag
    LINE       1     Account 12345678 ...                                   (ignored)
    WORD       1     12345678  99.6        0.1401  0.1023  0.0790  0.0118  account_number

* Only WORD rows are used (when a BlockType column exists).
* Left/Top/Width/Height are Textract's 0-1 page ratios; pixel coordinates also work if the image
  size is given. Boxes are scaled to LayoutLMv3's 0-1000 integer grid.
* Tag = plain field name per word ("O" / empty = no field). Only needed for extraction training.
* Column names are matched case- and punctuation-insensitively (e.g. Geometry.BoundingBox.Left).
"""
import csv
import io
import os
import re

ALIASES = {
    "text": ["Text", "DetectedText", "Word"],
    "left": ["Left", "Geometry.BoundingBox.Left", "BoundingBox.Left"],
    "top": ["Top", "Geometry.BoundingBox.Top", "BoundingBox.Top"],
    "width": ["Width", "Geometry.BoundingBox.Width", "BoundingBox.Width"],
    "height": ["Height", "Geometry.BoundingBox.Height", "BoundingBox.Height"],
    "block_type": ["BlockType", "Type"],
    "page": ["Page", "PageNumber"],
    "tag": ["Tag", "Label", "Field"],
}
NULL_TAGS = {"", "O", "0", "NONE", "NULL"}


def _key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_tsv(src, image_size=None, page=None, columns=None):
    """src: path, or the TSV text itself. Returns [{"text", "box": [x0,y0,x1,y1] (0-1000), "tag"}]."""
    if "\t" in src or "\n" in src:
        text = src
    else:
        with open(src, encoding="utf-8-sig") as f:
            text = f.read()
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")), delimiter="\t", quoting=csv.QUOTE_NONE)
    by_key = {_key(h): h for h in (reader.fieldnames or [])}
    col = {}
    for logical, names in ALIASES.items():
        for n in ([columns[logical]] if columns and columns.get(logical) else names):
            if _key(n) in by_key:
                col[logical] = by_key[_key(n)]
                break
    missing = [c for c in ("text", "left", "top", "width", "height") if c not in col]
    if missing:
        raise ValueError(f"OCR TSV missing columns {missing}; header: {reader.fieldnames}")

    raw = []
    for r in reader:
        if "block_type" in col and (r.get(col["block_type"]) or "").strip().upper() != "WORD":
            continue
        if page is not None and "page" in col and str(r.get(col["page"]) or "1").strip() != str(page):
            continue
        t = (r.get(col["text"]) or "").strip()
        if not t:
            continue
        l, tp, w, h = (float(r[col[k]]) for k in ("left", "top", "width", "height"))
        tag = (r.get(col["tag"]) or "").strip() if "tag" in col else ""
        raw.append((t, (l, tp, l + w, tp + h), "O" if tag.upper() in NULL_TAGS else tag))

    normalized = all(b[2] <= 1.01 and b[3] <= 1.01 for _, b, _ in raw)
    if not normalized and not image_size:
        raise ValueError("Pixel coordinates in OCR TSV: pass image_size=(width, height)")
    W, H = (1.0, 1.0) if normalized else image_size

    def norm(v, s):
        return int(max(0, min(1000, round(1000 * v / s))))

    out = []
    for t, (x0, y0, x1, y1), tag in raw:
        b = [norm(x0, W), norm(y0, H), norm(x1, W), norm(y1, H)]
        b[2], b[3] = max(b[2], b[0]), max(b[3], b[1])
        out.append({"text": t, "box": b, "tag": tag})
    return out
