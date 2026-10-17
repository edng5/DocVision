"""Pages -> LayoutLMv3 inputs: BPE tokens with word boxes, sliding windows, 224x224 images.

Manifest = JSON Lines, one row per page (paths relative to the manifest):
  {"doc_id": "B001", "page_num": 1, "image": "pages/B001_p1.jpg", "ocr": "pages/B001_p1.tsv",
   "doc_type": "bank_statement", "segment_start": true, "split": "train"}
"""
import hashlib
import io
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from .ocr import read_tsv


class Featurizer:
    """Shared tokenizer + image preprocessing settings (one per backbone)."""

    def __init__(self, tokenizer, max_seq_len=512, image_size=224, cache_images=False):
        self.tok = tokenizer
        self.tok.model_max_length = int(1e9)      # pages are windowed by Page.windows(); silence length warnings
        self.max_seq_len, self.image_size = int(max_seq_len), int(image_size)
        self.body = self.max_seq_len - 2
        self.cls, self.sep, self.pad, self.mask = (tokenizer.cls_token_id, tokenizer.sep_token_id,
                                                   tokenizer.pad_token_id, tokenizer.mask_token_id)
        self.cache_images = cache_images

    def page(self, words, image, meta=None):
        return Page(self, words, image, meta or {})

    def page_from_row(self, row, data_dir=""):
        img = os.path.join(data_dir, row["image"])
        return Page(self, read_tsv(os.path.join(data_dir, row["ocr"])), img, row)

    def image_tensor(self, image):
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
        img = image.convert("RGB").resize((self.image_size,) * 2, Image.BICUBIC)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5      # LayoutLMv3 normalisation
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class Page:
    """One page: OCR words, BPE tokens (each inheriting its word's box), and the page image."""

    def __init__(self, fz, words, image, meta):
        self.fz, self.words, self.meta, self._image, self._tensor = fz, words, meta, image, None
        texts = [w["text"] for w in words] or ["."]
        boxes = [w["box"] for w in words] or [[0, 0, 0, 0]]
        enc = fz.tok(texts, boxes=boxes, add_special_tokens=False)
        self.ids, self.bbox = enc["input_ids"], enc["bbox"]
        self.wids = enc.word_ids() if words else [None] * len(self.ids)

    @property
    def doc_type(self):
        return self.meta.get("doc_type")

    def image(self):
        if self._tensor is not None:                       # cached as uint8 (4x smaller than float)
            return (self._tensor.float() / 255.0 - 0.5) / 0.5
        t = self.fz.image_tensor(self._image)
        if self.fz.cache_images:
            self._tensor = ((t * 0.5 + 0.5) * 255.0).round().clamp(0, 255).to(torch.uint8)
        return t

    def windows(self, overlap=0):
        n, body = len(self.ids), self.fz.body
        if n <= body:
            return [(0, n)]
        step = body - overlap
        return [(s, s + body) for s in list(range(0, n - body, step)) + [n - body]]

    def window(self, s, e):
        fz = self.fz
        return {"input_ids": [fz.cls] + self.ids[s:e] + [fz.sep], "bbox": [[0] * 4] + self.bbox[s:e] + [[0] * 4],
                "word_ids": [None] + self.wids[s:e] + [None]}

    def first_window(self):
        it = self.window(0, min(len(self.ids), self.fz.body))
        it["pixel_values"] = self.image()
        return it


def collate(items, pad_id):
    B, L = len(items), max(len(it["input_ids"]) for it in items)
    out = {"input_ids": torch.full((B, L), pad_id), "bbox": torch.zeros(B, L, 4, dtype=torch.long),
           "attention_mask": torch.zeros(B, L, dtype=torch.long),
           "pixel_values": torch.stack([it["pixel_values"] for it in items])}
    for key in ("labels", "mlm_labels"):
        if key in items[0]:
            out[key] = torch.full((B, L), -100)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        out["input_ids"][i, :n] = torch.tensor(it["input_ids"])
        out["bbox"][i, :n] = torch.tensor(it["bbox"])
        out["attention_mask"][i, :n] = 1
        for key in ("labels", "mlm_labels"):
            if key in it:
                out[key][i, :n] = torch.tensor(it[key])
    if "patch_mask" in items[0]:
        out["patch_mask"] = torch.stack([it["patch_mask"] for it in items])
    return out


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_manifest(path, split_ratios=(0.7, 0.15, 0.15)):
    """Read manifest rows; rows without 'split' are assigned by a hash of doc_id (no page leakage)."""
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["_dir"] = base
                if not r.get("split"):
                    u = int(hashlib.sha1(str(r["doc_id"]).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    r["split"] = "train" if u < split_ratios[0] else "val" if u < sum(split_ratios[:2]) else "test"
                rows.append(r)
    return rows


def build_bundles(pages):
    """Group pages by doc_id (a bundle = one customer upload, possibly several documents)."""
    bundles = OrderedDict()
    for p in pages:
        bundles.setdefault(p.meta["doc_id"], []).append(p)
    return OrderedDict((k, sorted(v, key=lambda p: p.meta.get("page_num", 0))) for k, v in bundles.items())


def split_bundles(bundles):
    return {s: [b for b in bundles.values() if b[0].meta.get("split") == s] for s in ("train", "val", "test")}
