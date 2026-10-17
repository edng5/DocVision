"""Synthetic bank documents: page JPG + OCR TSV (AWS Textract layout, with a word-level Tag column).

Doc types: bank_statement (1-3 pages), cheque, pay_stub, utility_bill (1-2 pages) and letters
(`other`). Documents are combined into bundles of 1-3 documents, like a real customer upload.
Every word box comes from the renderer, with realistic noise: varied captions ("Acct No." /
"Account #"), distractor numbers and dates, OCR character confusions (0/O, 1/l), blur and JPEG
compression. Use it to demo or smoke-test the pipeline; train real models on your own documents.
"""
import json
import os
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MAX_ROWS = 24   # max transaction rows per statement page

W_PAGE, H_PAGE = 850, 1100
FONT_FILES = {"regular": ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
              "bold": ["arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
              "mono": ["cour.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"]}
FONT_DIRS = ["", "C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/Library/Fonts", "/System/Library/Fonts/Supplemental"]

@lru_cache(maxsize=None)
def get_font(kind, size):
    for name in FONT_FILES[kind]:
        for d in FONT_DIRS:
            try:
                return ImageFont.truetype(os.path.join(d, name) if d else name, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)

FIRST = ["James", "Mary", "Robert", "Linda", "Ahmed", "Priya", "Chen", "Sofia", "Daniel", "Grace", "Omar", "Elena",
         "Kenji", "Fatima", "Lucas", "Amara", "Noah", "Isabel", "Arjun", "Hannah"]
LAST = ["Smith", "Johnson", "Patel", "Nguyen", "Garcia", "Okafor", "Kowalski", "Tanaka", "Haddad", "Silva",
        "Brown", "Martin", "Wong", "Rossi", "Kim", "Murphy", "Ivanova", "Mensah", "Larsen", "Cohen"]
BANKS = ["First Harbor Bank", "Northwind Savings", "Maple Trust Bank", "Summit Credit Union", "Pacific Coast Bank",
         "Granite State Bank", "Riverbend Financial"]
EMPLOYERS = ["Acme Logistics Inc", "Bluebird Health LLC", "Copperline Foods", "Delta Robotics Corp", "Evergreen Schools",
             "Fjord Software Ltd"]
UTILITIES = ["City Power & Light", "Metro Water Authority", "Clearwave Gas Co", "Horizon Telecom"]
DESCS = ["POS PURCHASE GROCERY", "ATM WITHDRAWAL", "DIRECT DEPOSIT PAYROLL", "ONLINE TRANSFER", "CARD PAYMENT FUEL",
         "UTILITY BILL PAYMENT", "CHECK DEPOSIT", "INTEREST CREDIT", "MONTHLY FEE", "RESTAURANT", "PHARMACY",
         "INSURANCE PREMIUM", "MORTGAGE PAYMENT", "SUBSCRIPTION SERVICE"]
STREETS = ["Oak St", "Maple Ave", "Harbor Rd", "Elm Dr", "Cedar Ln", "Pine St", "Lakeview Blvd"]

def rname(rng): return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
def raddr(rng): return f"{rng.randint(10, 9999)} {rng.choice(STREETS)}"
def rdate(rng, base=None):
    d = base or (np.datetime64("2023-01-01") + np.timedelta64(rng.randint(0, 900), "D"))
    y, m, dd = str(d).split("-")
    return rng.choice([f"{m}/{dd}/{y}", f"{dd}/{m}/{y}", f"{y}-{m}-{dd}"]), d
def money(v, rng, dollar=None):
    s = f"{v:,.2f}"
    return ("$" + s) if (rng.random() < 0.5 if dollar is None else dollar) else s
def acct(rng): return rng.choice([f"{rng.randint(10**9, 10**10 - 1)}", f"{rng.randint(10, 99)}-{rng.randint(1000, 9999)}-{rng.randint(100000, 999999)}"])

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
def words_num(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000: return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words_num(n % 100))
    return words_num(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + words_num(n % 1000))

class Page:
    """Renders text line by line and records every word with its exact box and tag."""
    def __init__(self, rng):
        self.rng = rng
        shade = rng.randint(242, 255)
        self.img = Image.new("RGB", (W_PAGE, H_PAGE), (shade, shade, rng.randint(236, 255)))
        self.draw = ImageDraw.Draw(self.img)
        self.words, self.lines = [], []
    def text(self, x, y, segments, size=16, kind="regular", color=(20, 20, 25)):
        f = get_font(kind, size); space = self.draw.textlength(" ", font=f)
        line = []
        for seg, tag in segments:
            for w in str(seg).split():
                l, t, r, b = self.draw.textbbox((x, y), w, font=f)
                self.draw.text((x, y), w, font=f, fill=color)
                self.words.append({"text": w, "box": (l, t, r, b), "tag": tag}); line.append(len(self.words) - 1)
                x = r + space
        if line:
            self.lines.append(line)
        return x
    def rule(self, y, x0=50, x1=W_PAGE - 50, width=1):
        self.draw.line([(x0, y), (x1, y)], fill=(90, 90, 90), width=width)
    def save(self, img_path, tsv_path, page_no=1):
        rng, img = self.rng, self.img
        if rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
        img.save(img_path, quality=rng.randint(55, 92))
        confus = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "8": "B"}
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            f.write("BlockType\tPage\tId\tText\tConfidence\tLeft\tTop\tWidth\tHeight\tTag\n")
            for li, idxs in enumerate(self.lines):         # LINE rows (ignored by the parser, like Textract)
                ws = [self.words[i] for i in idxs]
                l = min(w["box"][0] for w in ws); t = min(w["box"][1] for w in ws)
                r = max(w["box"][2] for w in ws); b = max(w["box"][3] for w in ws)
                f.write(f"LINE\t{page_no}\tL{li}\t{' '.join(w['text'] for w in ws)}\t99.0\t{l/W_PAGE:.5f}\t{t/H_PAGE:.5f}\t{(r-l)/W_PAGE:.5f}\t{(b-t)/H_PAGE:.5f}\t\n")
            for i, w in enumerate(self.words):
                text = w["text"]
                if rng.random() < 0.015:                    # simulated OCR character confusion
                    j = rng.randrange(len(text))
                    text = text[:j] + confus.get(text[j], text[j]) + text[j + 1:]
                l, t, r, b = w["box"]
                f.write(f"WORD\t{page_no}\tW{i}\t{text}\t{rng.uniform(80, 99.9):.1f}\t{l/W_PAGE:.5f}\t{t/H_PAGE:.5f}\t"
                        f"{(r-l)/W_PAGE:.5f}\t{(b-t)/H_PAGE:.5f}\t{w['tag']}\n")

def gen_statement(rng):
    bank, holder, number = rng.choice(BANKS), rname(rng), acct(rng)
    (s_txt, s_d) = rdate(rng); (e_txt, _) = rdate(rng, s_d + np.timedelta64(rng.choice([29, 30, 31]), "D"))
    bal = rng.uniform(300, 25000); opening = bal
    n_pages = rng.choice([1, 1, 2, 2, 3])
    pages = []
    for p in range(n_pages):
        pg = Page(rng); x0 = 50 + rng.randint(0, 20); y = 35 + rng.randint(0, 15)
        pg.text(x0, y, [(bank, "bank_name")], 26, "bold")
        pg.text(640, y + 8, [("Page", "O"), (f"{p + 1} of {n_pages}", "O")], 13)
        y += 50
        pg.text(x0, y, [(raddr(rng) + " Customer Service 1-800-" + str(rng.randint(100, 999)) + "-" + str(rng.randint(1000, 9999)), "O")], 12)
        y += 30
        if p == 0:
            pg.text(x0, y, [(rng.choice(["Statement of Account", "Account Statement", "Monthly Statement"]), "O")], 20, "bold"); y += 40
            pg.text(x0, y, [(rng.choice(["Account Holder:", "Name:", "Customer:"]), "O"), (holder, "account_holder_name")], 16); y += 28
            pg.text(x0, y, [(rng.choice(["Account Number:", "Acct No.", "Account #"]), "O"), (number, "account_number")], 16); y += 28
            pg.text(x0, y, [("Statement Period:", "O"), (s_txt, "statement_start_date"), ("to", "O"), (e_txt, "statement_end_date")], 16); y += 28
            pg.text(x0, y, [(rng.choice(["Opening Balance", "Beginning Balance", "Previous Balance"]), "O"), (money(opening, rng), "opening_balance")], 16); y += 40
        else:
            pg.text(x0, y, [("Account Number:", "O"), (number, "account_number"), ("(continued)", "O")], 14); y += 34
        pg.rule(y - 6)
        cols = [x0, x0 + 120, x0 + 470, x0 + 620]
        for c, h in zip(cols, ["Date", "Description", "Amount", "Balance"]):
            pg.text(c, y, [(h, "O")], 14, "bold")
        y += 28
        for _ in range(rng.randint(6, MAX_ROWS)):
            if y > H_PAGE - 130:
                break
            amt = round(rng.uniform(-900, 1500), 2); bal += amt
            d_txt, _ = rdate(rng, s_d + np.timedelta64(rng.randint(0, 29), "D"))
            pg.text(cols[0], y, [(d_txt, "transaction_date")], 13)
            pg.text(cols[1], y, [(rng.choice(DESCS) + (" " + str(rng.randint(1000, 9999)) if rng.random() < 0.4 else ""), "O")], 13)
            pg.text(cols[2], y, [(("-" if amt < 0 else "") + money(abs(amt), rng, False), "transaction_amount")], 13, "mono")
            pg.text(cols[3], y, [(money(bal, rng, False), "O")], 13, "mono")
            y += 24
        if p == n_pages - 1:
            pg.rule(y + 4); y += 18
            pg.text(x0, y, [(rng.choice(["Closing Balance", "Ending Balance"]), "O"), (money(bal, rng), "closing_balance")], 16, "bold")
        pages.append(pg)
    return pages

def gen_cheque(rng):
    pg = Page(rng); bank, payee = rng.choice(BANKS), rng.choice([rname(rng), rng.choice(EMPLOYERS)])
    amt = rng.randint(20, 9999) + rng.choice([0, 0.5, 0.25, 0.99]); cents = int(round((amt % 1) * 100))
    oy = 60 + rng.randint(0, 120)
    pg.draw.rectangle([40, oy, 810, oy + 360], outline=(60, 60, 60), width=2, fill=(rng.randint(225, 245), rng.randint(235, 250), 255))
    pg.text(70, oy + 20, [(rname(rng), "O")], 15, "bold"); pg.text(70, oy + 42, [(raddr(rng), "O")], 12)
    pg.text(680, oy + 20, [(str(rng.randint(100, 9999)), "cheque_number")], 16)
    d_txt, _ = rdate(rng)
    pg.text(520, oy + 70, [("Date", "O"), (d_txt, "cheque_date")], 15)
    pg.text(70, oy + 120, [(rng.choice(["Pay to the order of", "PAY TO THE ORDER OF", "Pay"]), "O"), (payee, "payee_name")], 16)
    pg.draw.rectangle([620, oy + 112, 790, oy + 145], outline=(60, 60, 60))
    pg.text(632, oy + 118, [(money(amt, rng, True), "amount_numeric")], 16)
    pg.text(70, oy + 170, [(f"{words_num(int(amt))} and {cents:02d}/100".capitalize(), "amount_words"), ("Dollars", "O")], 15)
    pg.text(70, oy + 220, [(bank, "O")], 14, "bold")
    pg.text(70, oy + 255, [("Memo", "O"), (rng.choice(["rent", "invoice " + str(rng.randint(100, 999)), "deposit", "services"]), "O")], 13)
    pg.rule(oy + 285, 480, 780)
    number = str(rng.randint(10**9, 10**10 - 1))
    pg.text(110, oy + 320, [(f"C{rng.randint(10**8, 10**9 - 1)}C", "O"), (number, "account_number"), (f"{rng.randint(1000, 9999)}", "O")], 15, "mono")
    return [pg]

def gen_paystub(rng):
    pg = Page(rng); emp, name = rng.choice(EMPLOYERS), rname(rng); x0 = 50 + rng.randint(0, 20); y = 40
    pg.text(x0, y, [(emp, "employer_name")], 24, "bold"); y += 36
    pg.text(x0, y, [(raddr(rng), "O")], 12); y += 34
    pg.text(x0, y, [(rng.choice(["Earnings Statement", "Pay Stub", "Payroll Advice"]), "O")], 19, "bold"); y += 40
    pg.text(x0, y, [(rng.choice(["Employee:", "Employee Name:", "Name:"]), "O"), (name, "employee_name")], 16); y += 28
    pg.text(x0, y, [("Employee ID", "O"), (str(rng.randint(10000, 99999)), "O")], 14); y += 28
    s, sd = rdate(rng); e, _ = rdate(rng, sd + np.timedelta64(13, "D")); pd_, _ = rdate(rng, sd + np.timedelta64(17, "D"))
    pg.text(x0, y, [("Pay Period:", "O"), (s, "O"), ("-", "O"), (e, "O")], 14); y += 26
    pg.text(x0, y, [("Pay Date:", "O"), (pd_, "pay_date")], 16); y += 40
    pg.rule(y - 8)
    hours, rate = rng.randint(60, 90), rng.uniform(18, 75); gross = hours * rate; ot = rng.randint(0, 10) * rate * 1.5; gross += ot
    for lbl, val in [("Regular", hours * rate), ("Overtime", ot)]:
        pg.text(x0, y, [(lbl, "O")], 14); pg.text(x0 + 300, y, [(money(val, rng, False), "O")], 14, "mono"); y += 24
    pg.text(x0, y, [("Gross Pay", "O")], 15, "bold"); pg.text(x0 + 300, y, [(money(gross, rng), "gross_pay")], 15, "mono"); y += 40
    ded = 0
    for lbl in ["Federal Tax", "State Tax", "Social Security", "Medicare", "401k"]:
        v = gross * rng.uniform(0.01, 0.12); ded += v
        pg.text(x0, y, [(lbl, "O")], 14); pg.text(x0 + 300, y, [(money(v, rng, False), "O")], 14, "mono")
        pg.text(x0 + 470, y, [("YTD", "O"), (money(v * rng.randint(3, 20), rng, False), "O")], 12); y += 24
    pg.rule(y + 2); y += 14
    pg.text(x0, y, [("Net Pay", "O")], 16, "bold"); pg.text(x0 + 300, y, [(money(gross - ded, rng), "net_pay")], 16, "mono")
    return [pg]

def gen_utility(rng):
    prov, name, number = rng.choice(UTILITIES), rname(rng), acct(rng); pages = []
    bd, bdd = rdate(rng); dd, _ = rdate(rng, bdd + np.timedelta64(21, "D")); amt = rng.uniform(25, 600)
    pg = Page(rng); x0 = 50 + rng.randint(0, 20); y = 40
    pg.text(x0, y, [(prov, "O")], 24, "bold"); y += 50
    pg.text(x0, y, [(rng.choice(["Customer Name:", "Name:", "Bill To:"]), "O"), (name, "customer_name")], 16); y += 28
    pg.text(x0, y, [(raddr(rng), "O")], 13); y += 28
    pg.text(x0, y, [(rng.choice(["Account Number:", "Account No."]), "O"), (number, "account_number")], 16); y += 28
    pg.text(x0, y, [("Bill Date:", "O"), (bd, "bill_date")], 16); y += 28
    pg.text(x0, y, [("Due Date:", "O"), (dd, "due_date")], 16); y += 44
    pg.draw.rectangle([x0 - 5, y - 6, x0 + 380, y + 34], outline=(40, 40, 40), width=2)
    pg.text(x0 + 5, y, [(rng.choice(["Amount Due", "Total Amount Due", "Please Pay"]), "O"), (money(amt, rng, True), "amount_due")], 18, "bold"); y += 60
    for _ in range(rng.randint(3, 7)):
        pg.text(x0, y, [(rng.choice(["Usage charge", "Service fee", "Taxes", "Delivery", "Previous payment"]), "O")], 13)
        pg.text(x0 + 400, y, [(money(rng.uniform(1, 200), rng, False), "O")], 13, "mono"); y += 22
    pages.append(pg)
    if rng.random() < 0.4:
        pg2 = Page(rng); y = 50
        pg2.text(60, y, [(prov + " Usage History", "O")], 18, "bold"); y += 40
        for m in range(12):
            pg2.text(60, y, [(f"Month {m + 1}", "O"), (f"{rng.randint(100, 900)} kWh", "O"), (money(rng.uniform(20, 300), rng, False), "O")], 13); y += 22
        pages.append(pg2)
    return pages

SENTENCES = ["We are writing to inform you about changes to your account terms.",
             "Please review the enclosed information carefully.", "Your reference number is {n}.",
             "If you have questions call us at 1-800-{a}-{b}.", "This notice was issued on {d}.",
             "Thank you for banking with us.", "The new fee schedule takes effect on {d}.",
             "No action is required on your part.", "Interest rates may change from time to time."]
def gen_letter(rng):
    pages = []
    for p in range(rng.choice([1, 1, 2])):
        pg = Page(rng); y = 50
        pg.text(60, y, [(rng.choice(BANKS), "O")], 22, "bold"); y += 50
        if p == 0:
            pg.text(60, y, [(rdate(rng)[0], "O")], 14); y += 30
            pg.text(60, y, [("Dear", "O"), (rname(rng) + ",", "O")], 15); y += 36
        for _ in range(rng.randint(6, 14)):
            s = rng.choice(SENTENCES).format(n=rng.randint(10**5, 10**7), a=rng.randint(100, 999), b=rng.randint(1000, 9999), d=rdate(rng)[0])
            words = s.split(); x = 60
            while words:
                chunk = words[:rng.randint(6, 11)]; words = words[len(chunk):]
                pg.text(60, y, [(" ".join(chunk), "O")], 14); y += 22
            y += 10
        pages.append(pg)
    return pages

GENERATORS = {"bank_statement": gen_statement, "cheque": gen_cheque, "pay_stub": gen_paystub,
              "utility_bill": gen_utility, "other": gen_letter}


DOC_WEIGHTS = {"bank_statement": 0.3, "cheque": 0.2, "pay_stub": 0.15, "utility_bill": 0.15, "other": 0.2}


def generate_dataset(out_dir, n_bundles=90, seed=13, split=(0.7, 0.15, 0.15), max_rows=24):
    """Writes out_dir/pages/*.jpg|*.tsv and out_dir/manifest.jsonl. Returns the manifest rows."""
    global MAX_ROWS
    MAX_ROWS = max_rows
    os.makedirs(os.path.join(out_dir, "pages"), exist_ok=True)
    rng = random.Random(seed)
    rows, types, weights = [], list(GENERATORS), [DOC_WEIGHTS[t] for t in GENERATORS]
    for bi in range(n_bundles):
        bid = f"B{bi:05d}"
        u = rng.random()
        part = "train" if u < split[0] else "val" if u < split[0] + split[1] else "test"
        page_num = 0
        for dt in rng.choices(types, weights, k=rng.choice([1, 1, 2, 2, 3])):
            for k, pg in enumerate(GENERATORS[dt](rng)):
                page_num += 1
                img, tsv = f"pages/{bid}_p{page_num}.jpg", f"pages/{bid}_p{page_num}.tsv"
                pg.save(os.path.join(out_dir, img), os.path.join(out_dir, tsv))
                rows.append({"doc_id": bid, "page_num": page_num, "image": img, "ocr": tsv, "doc_type": dt,
                             "segment_start": k == 0, "split": part})
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows
