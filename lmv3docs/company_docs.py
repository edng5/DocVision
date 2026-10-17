"""Company Documents dataset (Kaggle: ayoubcherguelaine/company-documents-dataset, Apache-2.0)
-> page JPGs + OCR TSVs (Textract layout, word-level Tag column) + bundle manifest.

Source: 2,676 PDFs derived from the Northwind database:
  invoices/invoice_<order>.pdf                     (830, 1 page)
  PurchaseOrders/purchase_orders_<order>.pdf       (830, 1 page)
  Shipping orders/order_<order>.pdf                (809, mostly 2 pages)
  Inventory Report/monthly/StockReport_<YYYY-MM>.pdf            (23)
  Inventory Report/monthly-Category/StockReport_<YYYY-MM>_<k>.pdf (184, one per category)

Bundles: files that share their identifying number form one bundle. All documents for one order
ID make an "order packet" (invoice + purchase order + shipping order); all stock reports for one
month make a "monthly stock packet". Documents inside a bundle are shuffled (seeded), so the
classifier must label every page from its content, not its position.

Labels: the PDFs carry an exact text layer. Field values are tagged on the PDF words using the
layout (the value after "Order ID:", the cell under a table header, ...). The tags are then
transferred to the OCR words (docTR on the rendered page) by box overlap, so the training data
looks exactly like production input (image + OCR), while labels do not depend on OCR spelling.
"""
import glob
import hashlib
import json
import os
import random
import re
from collections import defaultdict

DOC_SOURCES = {
    "invoice": ["invoices/*.pdf"],
    "purchase_order": ["PurchaseOrders/*.pdf"],
    "shipping_order": ["Shipping orders/*.pdf"],
    "inventory_report": ["Inventory Report/monthly/monthly/*.pdf",
                         "Inventory Report/monthly-Category/monthly-Category/*.pdf"],
}

# key: value captions per doc type -> field (longest caption wins; matched on exact lowercase tokens)
KEY_VALUE = {
    "invoice": {"order id:": "order_id", "customer id:": "customer_id", "order date:": "order_date",
                "contact name:": "contact_name", "address:": "address", "city:": "city",
                "postal code:": "postal_code", "country:": "country", "phone:": "phone", "fax:": "fax",
                "totalprice": "total_price"},
    "shipping_order": {"order id:": "order_id", "ship name:": "ship_name", "ship address:": "ship_address",
                       "ship city:": "ship_city", "ship region:": "ship_region", "ship postal code:": "ship_postal_code",
                       "ship country:": "ship_country", "customer id:": "customer_id", "customer name:": "customer_name",
                       "employee name:": "employee_name", "shipper id:": "shipper_id", "shipper name:": "shipper_name",
                       "order date:": "order_date", "shipped date:": "shipped_date", "product:": "product_name",
                       "quantity:": "quantity", "unit price:": "unit_price", "total:": "line_total",
                       "total price:": "total_price"},
    "inventory_report": {"stock report for": "report_month", "category :": "category", "id category :": "category_id"},
    "purchase_order": {},
}

# tables: header phrase -> ordered (column header tokens, field); rows run until a stop word / blank gap
TABLES = {
    "invoice": [(["product", "id", "product", "name", "quantity", "unit", "price"],
                 [(["product", "id"], "product_id"), (["product", "name"], "product_name"),
                  (["quantity"], "quantity"), (["unit", "price"], "unit_price")])],
    "purchase_order": [(["order", "id", "order", "date", "customer", "name"],
                        [(["order", "id"], "order_id"), (["order", "date"], "order_date"),
                         (["customer", "name"], "customer_name")]),
                       (["product", "id:", "product:", "quantity:", "unit", "price:"],
                        [(["product", "id:"], "product_id"), (["product:"], "product_name"),
                         (["quantity:"], "quantity"), (["unit", "price:"], "unit_price")])],
    "inventory_report": [(["category", "product", "units", "sold", "units", "in", "stock", "unit", "price"],
                          [(["category"], "category"), (["product"], "product_name"), (["units", "sold"], "units_sold"),
                           (["units", "in", "stock"], "units_in_stock"), (["unit", "price"], "unit_price")]),
                         (["product", "units", "sold", "units", "in", "stock", "unit", "price"],
                          [(["product"], "product_name"), (["units", "sold"], "units_sold"),
                           (["units", "in", "stock"], "units_in_stock"), (["unit", "price"], "unit_price")])],
    "shipping_order": [],
}
STOP_ROW = {"totalprice", "page", "total"}

FIELDS = {   # config-ready field specs (multi = may occur many times per document)
    "invoice": ["order_id", "customer_id", "order_date", "contact_name", "address", "city", "postal_code", "country",
                "phone", "fax", {"name": "product_id", "multi": True}, {"name": "product_name", "multi": True},
                {"name": "quantity", "multi": True}, {"name": "unit_price", "multi": True}, "total_price"],
    "purchase_order": ["order_id", "order_date", "customer_name", {"name": "product_id", "multi": True},
                       {"name": "product_name", "multi": True}, {"name": "quantity", "multi": True},
                       {"name": "unit_price", "multi": True}],
    "shipping_order": ["order_id", "ship_name", "ship_address", "ship_city", "ship_region", "ship_postal_code",
                       "ship_country", "customer_id", "customer_name", "employee_name", "shipper_id", "shipper_name",
                       "order_date", "shipped_date", {"name": "product_name", "multi": True},
                       {"name": "quantity", "multi": True}, {"name": "unit_price", "multi": True},
                       {"name": "line_total", "multi": True}, "total_price"],
    "inventory_report": ["report_month", {"name": "category", "multi": True}, "category_id",
                         {"name": "product_name", "multi": True}, {"name": "units_sold", "multi": True},
                         {"name": "units_in_stock", "multi": True}, {"name": "unit_price", "multi": True}],
}


# ----------------------------------------------------------------------------- source files and bundles
def list_documents(root):
    docs = []
    for dt, pats in DOC_SOURCES.items():
        for pat in pats:
            for path in sorted(glob.glob(os.path.join(root, pat))):
                docs.append({"doc_type": dt, "pdf": path, "bundle": bundle_key(path)})
    return docs


def bundle_key(path):
    """Files sharing their identifying number belong together: order ID, or report month (YYYY-MM)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d{4})-(\d{2})", stem)
    if m:
        return f"stock_{m.group(1)}-{m.group(2)}"
    return f"order_{re.findall(r'\d+', stem)[0]}"


def split_of(key, ratios=(0.7, 0.15, 0.15)):
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < ratios[0] else "val" if u < ratios[0] + ratios[1] else "test"


# ----------------------------------------------------------------------------- PDF words (exact text layer)
def pdf_words(page):
    """Words with boxes from the PDF text layer, normalised to 0-1 (top-left origin)."""
    W, H = page.get_size()
    tp = page.get_textpage()
    words, cur = [], None
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace():
            cur = None
            continue
        l, b, r, t = tp.get_charbox(i, loose=True)          # full line-height box, like OCR boxes
        if r <= l or t <= b:
            continue
        box = [l / W, (H - t) / H, r / W, (H - b) / H]
        if cur is None or l / W - cur["box"][2] > 0.006:      # new word on a gap (some PDFs omit spaces)
            cur = {"text": ch, "box": box}
            words.append(cur)
        else:
            cur["text"] += ch
            cur["box"] = [min(cur["box"][0], box[0]), min(cur["box"][1], box[1]), max(cur["box"][2], box[2]),
                          max(cur["box"][3], box[3])]
    tp.close()
    # some PDFs (monthly stock reports) carry table rows below the page edge: invisible, so not labels
    return [w for w in words if 0.0 <= (w["box"][1] + w["box"][3]) / 2 <= 1.0]


def group_lines(words):
    lines = []
    for w in sorted(words, key=lambda w: ((w["box"][1] + w["box"][3]) / 2, w["box"][0])):
        cy, h = (w["box"][1] + w["box"][3]) / 2, w["box"][3] - w["box"][1]
        if lines and abs(cy - lines[-1]["cy"]) < 0.45 * max(h, lines[-1]["h"]):
            lines[-1]["words"].append(w)
        else:
            lines.append({"cy": cy, "h": h, "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["box"][0])
    return [ln["words"] for ln in lines]


# ----------------------------------------------------------------------------- layout tagging on PDF words
def _toks(words):
    return [w["text"].lower() for w in words]


def tag_pdf_page(words, doc_type):
    """Assign field tags to PDF words in place (w['tag']); returns {field: [values]}."""
    for w in words:
        w["tag"] = "O"
    lines = group_lines(words)
    kv = sorted(KEY_VALUE[doc_type].items(), key=lambda kv: -len(kv[0].split()))
    for line in lines:
        toks = _toks(line)
        for cap, field in kv:
            ct = cap.split()
            if toks[:len(ct)] == ct and len(toks) > len(ct):
                for w in line[len(ct):]:
                    w["tag"] = field
                break
    for header, cols in TABLES[doc_type]:
        for li, line in enumerate(lines):
            toks = _toks(line)
            if toks != header:
                continue
            starts, k = [], 0                                    # x start of each column from header words
            for ctoks, field in cols:
                starts.append((line[k]["box"][0], field))
                k += len(ctoks)
            prev_bottom = max(w["box"][3] for w in line)
            for row in lines[li + 1:]:
                top = min(w["box"][1] for w in row)
                if top - prev_bottom > 0.06 or _toks(row)[0] in STOP_ROW or len(row) < 2:
                    break
                for w in row:
                    cx = w["box"][0] + 0.002
                    field = [f for x, f in starts if x <= cx + 0.005]
                    w["tag"] = field[-1] if field else starts[0][1]
                prev_bottom = max(w["box"][3] for w in row)
    values = defaultdict(list)
    for line in lines:                                           # one value = run of same-tag words on a line
        run = []
        for w in line + [{"tag": None, "text": ""}]:
            if run and w["tag"] != run[-1]["tag"]:
                if run[0]["tag"] != "O":
                    values[run[0]["tag"]].append(" ".join(x["text"] for x in run))
                run = []
            if w["tag"] is not None:
                run.append(w)
    return dict(values)


# ----------------------------------------------------------------------------- OCR + tag transfer
def transfer_tags(ocr_words, pdf_words_tagged, min_frac=0.4):
    """Give each OCR word the tag of the PDF words it overlaps most. Overlap is measured against the
    smaller of the two boxes, so short tokens ("3", "Ln.") whose OCR box has padding still match."""
    for ow in ocr_words:
        x0, y0, x1, y1 = ow["box"]
        area = max(1e-9, (x1 - x0) * (y1 - y0))
        score = defaultdict(float)
        for pw in pdf_words_tagged:
            px0, py0, px1, py1 = pw["box"]
            ix, iy = min(x1, px1) - max(x0, px0), min(y1, py1) - max(y0, py0)
            if ix > 0 and iy > 0:
                parea = max(1e-9, (px1 - px0) * (py1 - py0))
                score[pw["tag"]] += ix * iy / min(area, parea)
        tag, frac = max(score.items(), key=lambda kv: kv[1]) if score else ("O", 0.0)
        ow["tag"] = tag if frac >= min_frac else "O"
    return ocr_words


def doctr_words(page_result):
    """docTR page -> [{'text','box'(0-1),'conf'}] plus line groups (for LINE rows)."""
    words, lines = [], []
    for block in page_result.blocks:
        for line in block.lines:
            idx = []
            for w in line.words:
                (x0, y0), (x1, y1) = w.geometry
                words.append({"text": w.value, "box": [x0, y0, x1, y1], "conf": float(w.confidence)})
                idx.append(len(words) - 1)
            if idx:
                lines.append(idx)
    return words, lines


def write_tsv(path, words, lines, page_no=1):
    def row(bt, i, text, conf, b, tag):
        return f"{bt}\t{page_no}\t{i}\t{text}\t{conf:.1f}\t{b[0]:.5f}\t{b[1]:.5f}\t{b[2] - b[0]:.5f}\t{b[3] - b[1]:.5f}\t{tag}\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("BlockType\tPage\tId\tText\tConfidence\tLeft\tTop\tWidth\tHeight\tTag\n")
        for li, idx in enumerate(lines):
            ws = [words[i] for i in idx]
            b = [min(w["box"][0] for w in ws), min(w["box"][1] for w in ws), max(w["box"][2] for w in ws), max(w["box"][3] for w in ws)]
            f.write(row("LINE", f"L{li}", " ".join(w["text"] for w in ws), 100 * min(w["conf"] for w in ws), b, ""))
        for i, w in enumerate(words):
            text = w["text"].replace("\t", " ").strip() or "."
            f.write(row("WORD", f"W{i}", text, 100 * w["conf"], w["box"], w["tag"]))


def norm_value(s):
    return re.sub(r"[^0-9a-z.]", "", s.lower())


def value_found(gold, candidates, min_ratio=0.85):
    """Gold value recovered by the OCR tags? Fuzzy, because the source PDFs contain mojibake
    ("M�ller") that OCR reads correctly ("Muller") and OCR makes rare character slips."""
    from difflib import SequenceMatcher
    g = norm_value(gold)
    return any(c == g or SequenceMatcher(None, g, c).ratio() >= min_ratio for c in candidates)


# ----------------------------------------------------------------------------- full preparation
def prepare(raw_root, out_dir, scale_ocr=2.0, scale_save=1.5, det="fast_base", reco="crnn_vgg16_bn",
            batch=8, limit_bundles=None, seed=13, log=print):
    import numpy as np
    import pypdfium2 as pdfium

    docs = list_documents(raw_root)
    by_bundle = defaultdict(list)
    for d in docs:
        by_bundle[d["bundle"]].append(d)
    keys = sorted(by_bundle)
    if limit_bundles:
        rng0 = random.Random(seed)
        stock = [k for k in keys if k.startswith("stock_")]
        order = [k for k in keys if k.startswith("order_")]
        n_stock = min(len(stock), max(2, limit_bundles // 10))
        keys = sorted(rng0.sample(order, min(len(order), limit_bundles - n_stock)) + rng0.sample(stock, n_stock))
    for sub in ("pages", "labels", "ocr_cache"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    from doctr.models import ocr_predictor
    model = None
    rows, coverage = [], defaultdict(lambda: [0, 0])
    for bi, key in enumerate(keys):
        bdocs = list(by_bundle[key])
        random.Random(f"{seed}:{key}").shuffle(bdocs)          # document order inside the packet is random
        page_num = 0
        for d in bdocs:
            stem = os.path.splitext(os.path.basename(d["pdf"]))[0]
            pdf = pdfium.PdfDocument(d["pdf"])
            gold = defaultdict(list)
            todo = []
            for pi in range(len(pdf)):
                page = pdf[pi]
                pw = pdf_words(page)
                for f, vals in tag_pdf_page(pw, d["doc_type"]).items():
                    gold[f] += vals
                name = f"{key}__{stem}_p{pi + 1}"
                img_path = os.path.join(out_dir, "pages", name + ".jpg")
                cache = os.path.join(out_dir, "ocr_cache", name + ".json")
                if not os.path.exists(img_path):
                    page.render(scale=scale_save).to_pil().convert("RGB").save(img_path, quality=88)
                todo.append((pi, name, pw, cache, page))
            missing = [t for t in todo if not os.path.exists(t[3])]
            if missing:
                if model is None:
                    model = ocr_predictor(det, reco, pretrained=True, assume_straight_pages=True)
                for i in range(0, len(missing), batch):
                    chunk = missing[i:i + batch]
                    imgs = [np.asarray(t[4].render(scale=scale_ocr).to_pil().convert("RGB")) for t in chunk]
                    for t, res in zip(chunk, model(imgs).pages):
                        w, l = doctr_words(res)
                        with open(t[3], "w", encoding="utf-8") as f:
                            json.dump({"words": w, "lines": l}, f)
            for pi, name, pw, cache, page in todo:
                with open(cache, encoding="utf-8") as f:
                    oc = json.load(f)
                words = transfer_tags(oc["words"], pw)
                write_tsv(os.path.join(out_dir, "pages", name + ".tsv"), words, oc["lines"])
                page_num += 1
                rows.append({"doc_id": key, "page_num": page_num, "image": f"pages/{name}.jpg", "ocr": f"pages/{name}.tsv",
                             "doc_type": d["doc_type"], "segment_start": pi == 0, "source_pdf": os.path.basename(d["pdf"]),
                             "split": split_of(key)})
                ocr_by_field = defaultdict(list)
                run = []
                for w in words + [{"tag": None}]:
                    if run and w["tag"] != run[-1]["tag"]:
                        if run[0]["tag"] != "O":
                            ocr_by_field[run[0]["tag"]].append(norm_value(" ".join(x["text"] for x in run)))
                        run = []
                    if w["tag"] is not None:
                        run.append(w)
                d.setdefault("_ocr_vals", defaultdict(list))
                for f, v in ocr_by_field.items():
                    d["_ocr_vals"][f] += v
            pdf.close()
            with open(os.path.join(out_dir, "labels", f"{key}__{stem}.json"), "w", encoding="utf-8") as f:
                json.dump({"doc_type": d["doc_type"], "bundle": key, "fields": gold}, f, indent=1)
            got = d.get("_ocr_vals", {})
            for field, vals in gold.items():                      # label quality: gold value found in OCR tags?
                for v in vals:
                    c = coverage[(d["doc_type"], field)]
                    c[1] += 1
                    c[0] += value_found(v, got.get(field, []))
        if log and (bi + 1) % 25 == 0:
            log(f"{bi + 1}/{len(keys)} bundles, {len(rows)} pages")
    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    cov = {f"{dt}.{field}": {"found": a, "total": b, "rate": a / b if b else 0.0} for (dt, field), (a, b) in sorted(coverage.items())}
    with open(os.path.join(out_dir, "label_coverage.json"), "w") as f:
        json.dump(cov, f, indent=1)
    return rows, cov
