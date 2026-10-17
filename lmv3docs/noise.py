"""Scan-style degradation of page images + re-OCR, with labels carried through the known geometry.

For every page of a dataset:
  1. degrade the clean image with a random "bad scan" pipeline (severity 0.2-1.0 per page):
     perspective skew + rotation, uneven lighting, faded ink, stains, fold lines, resolution loss,
     blur, sensor noise, salt-and-pepper specks, JPEG compression;
  2. keep the exact homography H of the geometric step and push every tagged word box of the
     clean page through it, so we know where each field now lies on the noisy image;
  3. run docTR OCR on the noisy image (OCR errors are now part of the data, as in production);
  4. transfer tags to the new OCR words by box overlap and write a Textract-layout TSV.
Labels therefore stay correct even where OCR misreads a value; `noise_coverage.json` reports how
often the clean gold value is still recovered verbatim(ish) by OCR on the noisy page.
"""
import hashlib
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from .company_docs import transfer_tags, value_found, write_tsv
from .ocr import read_tsv


def degrade(img, rng, severity):
    """img: HxWx3 uint8 RGB. Returns (noisy uint8 image, 3x3 homography clean->noisy pixels, jpeg quality)."""
    h, w = img.shape[:2]
    s = float(severity)
    # 1. geometry: rotation + perspective jitter of the corners (a phone / skewed flatbed scan)
    ang = np.deg2rad(rng.uniform(-3.0, 3.0) * s)
    c, sn = np.cos(ang), np.sin(ang)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    ctr = np.float32([w / 2, h / 2])
    dst = (src - ctr) @ np.float32([[c, sn], [-sn, c]]) + ctr
    dst += rng.uniform(-1, 1, size=(4, 2)).astype(np.float32) * 0.03 * s * min(w, h)
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    bg = int(rng.integers(150, 236))                              # scanner-lid border colour
    out = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(bg, bg, bg)).astype(np.float32)
    # 2. uneven lighting (gradient shadow across the page)
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = ((xx / w - 0.5) * np.cos(theta) + (yy / h - 0.5) * np.sin(theta)) + 0.5
    out *= (1.0 - rng.uniform(0.05, 0.35) * s * np.clip(ramp, 0, 1))[..., None]
    # 3. faded ink / low contrast
    out = 255.0 - (255.0 - out) * rng.uniform(1.0 - 0.45 * s, 1.0)
    # 4. stains (coffee rings / blotches) and 5. fold lines
    if rng.random() < 0.35 * s:
        overlay = out.copy()
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        ax, ay = int(rng.uniform(0.05, 0.15) * w), int(rng.uniform(0.04, 0.12) * h)
        color = (rng.uniform(150, 200), rng.uniform(120, 160), rng.uniform(80, 120))
        cv2.ellipse(overlay, (cx, cy), (ax, ay), rng.uniform(0, 180), 0, 360, color, -1)
        a = rng.uniform(0.10, 0.25)
        out = out * (1 - a) + overlay * a
    for _ in range(int(rng.random() < 0.4 * s) + int(rng.random() < 0.2 * s)):
        y = int(rng.uniform(0.25, 0.75) * h)
        out[max(0, y - 2):y + 2] *= rng.uniform(0.70, 0.90)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 6. resolution loss (low-dpi scan / fax)
    f = 1.0 - rng.uniform(0.15, 0.55) * s
    out = cv2.resize(cv2.resize(out, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA),
                     (w, h), interpolation=cv2.INTER_LINEAR)
    # 7. blur
    sigma = rng.uniform(0.0, 1.4) * s
    if sigma > 0.25:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    # 8. sensor noise + salt-and-pepper specks
    out = out.astype(np.float32) + rng.normal(0, rng.uniform(2.0, 16.0) * s, size=out.shape)
    sp = rng.random(out.shape[:2])
    out[sp < 0.0015 * s] = 0
    out[sp > 1 - 0.0015 * s] = 255
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 9. JPEG compression (applied when saving)
    quality = int(np.clip(92 - 60 * s * rng.uniform(0.5, 1.0), 25, 92))
    return out, H, quality


def warp_box(box_1000, H, w, h):
    """0-1000 box on the clean page -> 0-1 box on the noisy page (bbox of the warped corners)."""
    x0, y0, x1, y1 = (v / 1000.0 for v in box_1000)
    pts = np.float32([[[x0 * w, y0 * h]], [[x1 * w, y0 * h]], [[x1 * w, y1 * h]], [[x0 * w, y1 * h]]])
    q = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [float(np.clip(q[:, 0].min() / w, 0, 1)), float(np.clip(q[:, 1].min() / h, 0, 1)),
            float(np.clip(q[:, 0].max() / w, 0, 1)), float(np.clip(q[:, 1].max() / h, 0, 1))]


def _spans(words):
    """Values per field from tagged words in reading order (runs of equal tags)."""
    out, run = defaultdict(list), []
    for w in list(words) + [{"tag": None, "text": ""}]:
        if run and w["tag"] != run[-1]["tag"]:
            if run[0]["tag"] != "O":
                out[run[0]["tag"]].append(" ".join(x["text"] for x in run))
            run = []
        if w["tag"] is not None:
            run.append(w)
    return out


def build_noisy_dataset(src_manifest, out_dir, seed=7, det="fast_base", reco="crnn_vgg16_bn", batch=16,
                        device=None, severity=(0.2, 1.0), limit=None, log=print):
    """Degrade + re-OCR every page of `src_manifest` into `out_dir` (same rows, splits and labels)."""
    import torch
    from doctr.models import ocr_predictor

    src_dir = os.path.dirname(os.path.abspath(src_manifest))
    rows = [json.loads(l) for l in open(src_manifest, encoding="utf-8") if l.strip()][:limit]
    for sub in ("pages", "ocr_cache"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    out_rows, coverage = [], defaultdict(lambda: [0, 0, 0])
    for i in range(0, len(rows), batch):
        chunk = []
        for r in rows[i:i + batch]:
            name = os.path.splitext(os.path.basename(r["image"]))[0]
            img_path = os.path.join(out_dir, "pages", name + ".jpg")
            cache = os.path.join(out_dir, "ocr_cache", name + ".json")
            clean = np.asarray(Image.open(os.path.join(src_dir, r["image"])).convert("RGB"))
            hgt, wid = clean.shape[:2]
            rng = np.random.default_rng(int(hashlib.sha1(f"{seed}:{r['image']}".encode()).hexdigest()[:8], 16))
            sev = float(rng.uniform(*severity))
            noisy, Hm, q = degrade(clean, rng, sev)
            if not os.path.exists(img_path):
                Image.fromarray(noisy).save(img_path, quality=q)
            clean_words = read_tsv(os.path.join(src_dir, r["ocr"]))
            tagged = [{"box": warp_box(w["box"], Hm, wid, hgt), "tag": w["tag"]} for w in clean_words]
            chunk.append((r, name, img_path, cache, clean_words, tagged, sev, q))
        todo = [c for c in chunk if not os.path.exists(c[3])]
        if todo:
            if model is None:
                model = ocr_predictor(det, reco, pretrained=True, assume_straight_pages=True).to(device)
            imgs = [np.asarray(Image.open(c[2]).convert("RGB")) for c in todo]
            with torch.inference_mode():
                pages = model(imgs).pages
            for c, pg in zip(todo, pages):
                words, lines = [], []
                for block in pg.blocks:
                    for line in block.lines:
                        idx = []
                        for wd in line.words:
                            (x0, y0), (x1, y1) = wd.geometry
                            words.append({"text": wd.value, "box": [x0, y0, x1, y1], "conf": float(wd.confidence)})
                            idx.append(len(words) - 1)
                        if idx:
                            lines.append(idx)
                with open(c[3], "w", encoding="utf-8") as f:
                    json.dump({"words": words, "lines": lines}, f)
        for r, name, img_path, cache, clean_words, tagged, sev, q in chunk:
            with open(cache, encoding="utf-8") as f:
                oc = json.load(f)
            words = transfer_tags(oc["words"], tagged)
            write_tsv(os.path.join(out_dir, "pages", name + ".tsv"), words, oc["lines"])
            out_rows.append(dict(r, image=f"pages/{name}.jpg", ocr=f"pages/{name}.tsv", noise_severity=round(sev, 3),
                                 jpeg_quality=q))
            gold, got = _spans(clean_words), _spans(words)
            for field, vals in gold.items():
                for v in vals:
                    c = coverage[(r["doc_type"], field)]
                    c[1] += 1
                    c[0] += value_found(v, [x.lower() for x in got.get(field, [])] + got.get(field, []))
                c = coverage[(r["doc_type"], field)]                 # field still tagged at all (label survived)?
                c[2] += min(len(vals), len(got.get(field, [])))
        if log and (i // batch) % 25 == 0:
            log(f"{min(i + batch, len(rows))}/{len(rows)} pages")
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in out_rows) + "\n")
    cov = {f"{dt}.{field}": {"found": a, "total": b, "rate": a / b if b else 0.0, "tagged_rate": t / b if b else 0.0}
           for (dt, field), (a, b, t) in sorted(coverage.items())}
    with open(os.path.join(out_dir, "noise_coverage.json"), "w") as f:
        json.dump(cov, f, indent=1)
    return out_rows, cov
