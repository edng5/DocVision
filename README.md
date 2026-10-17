# DocVision: one LayoutLMv3 backbone, many lightweight task models

DocVision is document AI for multi-document business packets: invoices, purchase orders, shipping
orders, inventory reports, bank statements, cheques, pay stubs and utility bills. It reads page JPGs
and their OCR (a TSV in the AWS Textract layout) and works in three stages:

1. **Pretrain** LayoutLMv3, a multimodal encoder, on your own pages without labels. It uses
   **span-masked MLM** and **masked image modelling (MIM)**.
2. **Plug in small task models** on the frozen backbone. A document classifier reads whole bundles
   and labels every page. Each document type gets its own field extractor. Every task model is a
   LoRA adapter plus a small head.
3. **Serve** them all from **one backbone loaded once**. Switching tasks only changes which adapter
   is active (about 0.6 ms), so no other model is ever loaded.

No LLM is involved anywhere, so pages containing PII never leave the encoder pipeline.

> ⚠️ **Licence:** the `microsoft/layoutlmv3-base` weights are **CC BY-NC-SA 4.0 (non-commercial)**.
> Check this before any commercial use.

**Start here:** [`notebooks/layoutlmv3_document_ai.ipynb`](notebooks/layoutlmv3_document_ai.ipynb)
is the executed notebook with every result in this README. It covers the data, pretraining, the
classifier and extractors, evaluation, the memorisation audit, serving, ablations and end-to-end
accuracy.

---

## Contents

1. [Architecture](#1-architecture)
2. [Results at a glance](#2-results-at-a-glance)
3. [Experiment: noisy scans, ablations and audits](#3-experiment-noisy-scans-ablations-and-audits)
4. [Setup](#4-setup)
5. [Data](#5-data)
6. [Training](#6-training)
7. [Serving](#7-serving)
8. [Package layout](#8-package-layout)

---

## 1. Architecture

```
            page JPG ──► patch embed (16×16 px) ──┐
 OCR words + boxes ──► BPE + 1D/2D position ──┴─► LayoutLMv3 Transformer (125M)   ◄── Stage 1: pretrain
                                                   joint text + layout + image           span-masked MLM
                                                   attention                             MIM (40% patches → pixels)
                                                            │  frozen, loaded once
              ┌─────────────────────────────────────────────┼──────────────────────────────────┐
              ▼                                             ▼                                  ▼
   LoRA 'cls' + page vector                       LoRA 'ie_invoice'                 LoRA 'ie_bank_statement' ...
   [text CLS; mean text; image CLS]               + linear tag head                 + linear tag head
   → Transformer over ALL pages                   → per-word field tags             → per-word field tags
   → per-page doc type + boundary                 → invoice fields                  → bank-statement fields
```

| Stage | What trains | Data needed |
|---|---|---|
| 1. Backbone pretraining | every LayoutLMv3 weight (MLM + MIM) | unlabelled pages (image + OCR) |
| 2a. Document classifier | `cls` adapter, page projection, document Transformer and heads | `doc_type` per page |
| 2b. Field extractors | for each doc type, an `ie_<type>` adapter and a tag head | word `Tag`s in the OCR TSV |

**Pretraining objectives** (`lmv3docs/pretraining.py`). Both objectives come from one forward pass.
- **MLM uses whole-word span masking (SpanBERT style).**
  - Span lengths are drawn from Geometric(p=0.2) and capped at 10 words.
  - Spans are added until about 15% of the page's words are masked.
  - Every sub-token of a chosen word is predicted.
  - The 80% `[MASK]` / 10% random / 10% unchanged choice is made once per span.
  - A check on real pages measured 15.1% of words masked. Masked words form contiguous runs of
    about 4 words on average.
- **MIM masks about 40% of image patches in blocks.**
  - A learned mask embedding replaces each masked patch, through a hook on `patch_embed`.
  - A linear head reconstructs the patch pixels with an L1 loss (SimMIM style).

**Document classifier.**
- Each page becomes a vector made of the text CLS, the mean of the text tokens and the image CLS.
- A page-position embedding is added, then a 2-layer Transformer runs across the whole bundle.
- It outputs a doc type and a "new document starts here" flag for every page.
- First pages are weighted more heavily in the loss.

**Field extractors.**
- Words get plain tags (field names, no BIO prefixes), predicted on each word's first sub-token.
- Long pages are covered by sliding windows.
- Consecutive words with the same tag become one entity. Multi-valued table fields are split by line.

**Why this design**
- **Multimodal.** Words, their 2D positions and the pixels are attended to jointly, so a date in a
  header and a date in a table play different roles.
- **Tiny per-task cost.** A LoRA adapter (r=16 on q/k/v) is 1.7 MB in fp16. The 9 task models
  (1 classifier and 8 extractors) add 20 MB on top of a 239 MB backbone. Nine fully fine-tuned
  copies would need about 2.2 GB.
- **Decoupled.** Retraining one extractor leaves the backbone, the classifier and every other
  extractor bit-identical (verified in the notebook).

---

## 2. Results at a glance

These results are from the final model, trained and tested on **noisy scans**, on an RTX 4070.

| | |
|---|---|
| Data | 2,353 bundles, 7,064 pages, 9 classes: Company Documents (853 bundles) and synthetic bank documents (1,500 bundles). Every page was degraded and re-OCR'd with docTR. |
| Split | by bundle, 70 / 15 / 15. The test set is 390 bundles and 1,147 pages. No bundle appears in more than one split. |
| Classification (test) | page accuracy, macro-F1, boundary F1 and document exact match are all **1.000** |
| Extraction (test, entity F1 against labels on the noisy page) | **0.988–0.999** per doc type, mean **0.995** |
| End-to-end value accuracy (against the true value) | **0.821**, which is **100.0%** of the OCR ceiling of 0.821 |
| Memorisation audit | recall on values **never seen in training**: 0.990 (company), 0.997 (synthetic) |
| Ablations | full, no domain pretraining, MLM only and MIM only all land within **0.0013 entity F1** of each other |
| Latency (GPU, bf16) | classification 17.2 ms per page (p50), extraction 19.1 ms per page (p50), about 50 pages/s |
| Adapter switch | **0.63 ms**, against 0.96 s to load another LayoutLMv3 (about 1,500× faster) |
| Serving bundle | **263 MB**, every tensor fp16: backbone 239 MB, 9 adapters × 1.7 MB, heads 4.3 MB. It reloads in 1.3 s. |

---

## 3. Experiment: noisy scans, ablations and audits

### 3.1 Clean baseline, and why it scored 100%

The first run trained on clean pages, with PDFs rendered straight to JPG.
- **Scores:** every classification and extraction metric was 1.000.
- **Leakage check:** there was none.
  - Splits are made per bundle: a hash of the order ID or report month for Company Documents, and
    assigned by the generator for synthetic data.
  - 0 of 2,353 bundles appear in more than one split.
- **Why the scores were perfect:** both sources are **template-generated**.
  - The PDFs use a few fixed layouts filled from the Northwind database.
  - The synthetic bank documents come from our own generator.
  - The test bundles share those layouts. **68% of test field values also occur in training**, and
    Northwind customer names and addresses repeat 100%.
- **Memorisation:** the model was not recalling memorised values. Its extraction F1 of 1.000
  includes the 32% of test values it had never seen, so it reads them from the page.

The clean task is therefore easy, not leaked. To make the test meaningful, the experiment below
adds realistic scan noise to both the training and the test data.

### 3.2 Noise: "bad scan" versions of both datasets

Running `prepare_noisy.py` (code in `lmv3docs/noise.py`) degrades every page with a random severity
between 0.2 and 1.0 (mean 0.60). The degradations, in order:
- **Geometry:** perspective skew and rotation.
- **Light and paper:** uneven lighting, faded ink, coffee stains and fold lines.
- **Image quality:** resolution loss, blur, sensor noise, salt-and-pepper noise and JPEG compression
  (quality 25–90).

Field labels follow the known warp (a homography applied to each word box). **Every noisy page
is then OCR'd again with docTR**, so real OCR errors such as `Emst Handel` for `Ernst Handel`
become part of both training and test data.

| Dataset | Field labels kept after noise | True value still spelled correctly by OCR | Hardest field |
|---|---|---|---|
| Company Documents | 0.973 | 0.920 | `inventory_report.category_id` (0.39) |
| Synthetic bank | 0.983 | 0.808 | `bank_statement.transaction_amount` (0.65) |

Splits, bundles and doc types are the same as in the clean data.

### 3.3 Main results on noisy data

**Training settings.**

| Stage | Settings | Time |
|---|---|---|
| Pretraining | 4,000 steps × 4 pages | 9 min |
| Classifier | 5 epochs, windows of up to 5 pages | 11 min |
| Extractors | 8 epochs each, batch 4 | 12 min in total |

**Pretraining** (validation pages, before → after):

| Metric | Before | After |
|---|---|---|
| MLM loss | 14.61 | 1.28 |
| Masked-word accuracy | 0% | 75.2% |
| MIM L1 | 0.851 | 0.045 |

**Classification** (test set: 390 bundles, 1,147 pages): page accuracy, macro-F1, weighted F1,
first-page accuracy, boundary F1 and document exact match are all **1.000**. All 9 classes are at
P = R = F1 = 1.000. Noise does not hurt classification, because the layout survives degradation.

**Extraction (test set).** Entity F1 is scored against labels on the noisy page.
`doc_value_accuracy` is the share of single values returned per document that are correct.

| Doc type | Entity F1 (strict) | Doc value accuracy | Test pages |
|---|---|---|---|
| invoice | 0.997 | 0.991 | 136 |
| purchase_order | 0.999 | 1.000 | 136 |
| shipping_order | 0.996 | 0.993 | 254 |
| inventory_report | 0.998 | 0.958 | 36 |
| bank_statement | 0.999 | 0.991 | 223 |
| cheque | 0.988 | 0.989 | 90 |
| pay_stub | 0.992 | 0.990 | 61 |
| utility_bill | 0.993 | 0.989 | 105 |

### 3.4 End-to-end accuracy against the true value

Entity F1 checks whether the model **finds the right words** on the noisy page. A user cares about
something else: **is the returned value equal to the real one?**
- The real value is taken from the clean source page.
- The **OCR ceiling** is the share of real values that the noisy-page OCR still spells correctly.
  No extractor built on that OCR can do better.

| Doc type | Model trained on noisy pages | Model trained on clean pages | OCR ceiling | True values |
|---|---|---|---|---|
| invoice | 0.8781 | 0.8757 | 0.8781 | 2,872 |
| purchase_order | 0.9058 | 0.9058 | 0.9058 | 1,784 |
| shipping_order | 0.8254 | 0.8178 | 0.8254 | 3,414 |
| inventory_report | 0.8790 | 0.8772 | 0.8796 | 1,678 |
| bank_statement | 0.7644 | 0.7635 | 0.7644 | 7,978 |
| cheque | 0.8315 | 0.8185 | 0.8315 | 540 |
| pay_stub | 0.8459 | 0.8459 | 0.8459 | 305 |
| utility_bill | 0.8514 | 0.8514 | 0.8514 | 370 |
| **All** | **0.8210 (100.0% of ceiling)** | **0.8184 (99.7%)** | **0.8211** | **18,941** |

**Conclusions:**
- The extractor is effectively perfect. **All remaining error comes from OCR.**
- Training on noisy pages gives a small but consistent gain on the noisiest types (cheques
  +1.3 pts, shipping orders +0.8 pts).
- To improve end-to-end accuracy further, improve OCR or add value normalisation and validation
  (date formats, checksums, amounts against totals). Extraction training is not the bottleneck.

### 3.5 Memorisation audit (agnostic results)

The audit checks whether the model reads values or recalls them.
- Each test entity is grouped by whether its value appears in training (same doc type and field).
- Recall is the exact-span recall of the final noisy model.

| Dataset | Values **never seen** in training | Values seen in training |
|---|---|---|
| Company Documents | **0.9904** (1,763 / 1,780) | 0.9988 (7,706 / 7,715) |
| Synthetic bank | **0.9974** (5,659 / 5,674) | 0.9997 (3,408 / 3,409) |

The model is agnostic to the specific values. On values it has never encountered, it is less than
one point below its score on familiar ones, so it reads values from layout and content rather than
recalling them. The remaining gap is about what you would expect: familiar names also suffer fewer
OCR errors, because their spellings are common in training.

### 3.6 Ablations: what does each part of backbone pretraining contribute?

Four complete pipelines were trained on the same noisy pages.
- They share the same seed and identical downstream training (classifier and all 8 extractors).
- Only the backbone pretraining differs between them.

| Variant | Backbone | Page acc | Boundary F1 | Entity F1 | Doc value acc | Pretrain val |
|---|---|---|---|---|---|---|
| **full** | MLM + MIM pretraining | 1.000 | 1.000 | 0.9952 | **0.9877** | MLM acc 0.752, MIM 0.045 |
| no_pretrain | public weights, backbone **never fine-tuned** | 1.000 | 1.000 | **0.9953** | 0.9855 | — |
| mlm_only | MLM pretraining, no MIM | 1.000 | 1.000 | 0.9951 | 0.9857 | MLM acc 0.791 |
| mim_only | MIM pretraining, no MLM | 1.000 | 1.000 | 0.9939 | 0.9840 | MIM 0.040 |

Entity F1 by doc type:

| Doc type | full | no_pretrain | mlm_only | mim_only |
|---|---|---|---|---|
| invoice | 0.9968 | 0.9977 | 0.9979 | 0.9981 |
| purchase_order | 0.9991 | 1.0000 | 0.9997 | 1.0000 |
| shipping_order | 0.9958 | 0.9962 | 0.9952 | 0.9952 |
| inventory_report | 0.9984 | 0.9984 | 0.9981 | 0.9956 |
| bank_statement | 0.9991 | 0.9991 | 0.9989 | 0.9986 |
| cheque | 0.9878 | 0.9897 | 0.9850 | 0.9860 |
| pay_stub | 0.9917 | 0.9900 | 0.9917 | 0.9867 |
| utility_bill | 0.9929 | 0.9915 | 0.9943 | 0.9915 |

**Conclusions.** This is a single seed, so differences of about 0.001 are within noise.
- **On this data, domain pretraining makes no measurable difference.** All four variants classify
  perfectly and extract within 0.0013 F1 of each other.
- The public LayoutLMv3 weights plus LoRA already solve these template layouts, even with heavy
  scan noise.
- `full` has the best document value accuracy (+0.2 pts), and `mim_only` is the weakest on
  extraction. Both gaps are too small to call.
- MLM-only reaches a higher masked-word accuracy than full (0.791 vs 0.752), because it does not
  share capacity with MIM.
- **Practical reading:** skipping pretraining saves about 9 minutes per run here. Pretraining
  should pay off when the target pages differ from LayoutLMv3's pretraining data, for example in
  handwriting, unusual scripts or new layouts. Re-run this ablation on a labelled sample of your
  real documents before dropping it.

### 3.7 Serving measurements

| | |
|---|---|
| Footprint | backbone 125.3M params / 239 MB fp16; classifier adapter and head 5.9 MB; each extractor 1.7 MB |
| Adapter switch vs model load | 628 µs vs 0.96 s (1,528×) |
| Latency, RTX 4070, bf16 | classify: 20.0 ms mean, 17.2 p50, 29.2 p95; extract: 20.6 ms mean, 19.1 p50, 27.5 p95 |
| fp16 bundle | all 19 safetensors files are F16, 263 MB in total. It reloads in 1.3 s and gives the same predictions. |
| Decoupling | retraining `ie_invoice` changed only that adapter. The backbone, `cls` and the other 7 adapters stayed bit-identical, and shipping-order predictions were unchanged. |

### 3.8 Caveats

- Both sources are template-based. These scores show that the pipeline works, generalises across
  bundles and is robust to scan noise. They do not predict accuracy on real-world scans with
  varied layouts or handwriting. **Evaluate on a labelled sample of your own documents.**
- The ablations use a single seed. Use several seeds before drawing fine-grained conclusions.

---

## 4. Setup

```bash
pip install -r requirements.txt
# NVIDIA GPU: install the CUDA build of PyTorch (the default pip wheel on Windows is CPU-only)
pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
pip install "python-doctr[torch]" pypdfium2     # OCR + PDF rendering for data preparation
python tests/test_smoke.py                       # tiny random backbone, full pipeline, about 1 min on CPU
```

The first real run downloads `microsoft/layoutlmv3-base` (about 480 MB) from Hugging Face.

**GPU memory.** The notebook's `RUN = "full"` settings are sized for a 12 GB card (RTX 4070):
- pretraining batch 4, classifier windows of up to 5 pages, extractor batch 4;
- PyTorch capped at 87% of GPU memory (`GPU_MEMORY_FRACTION`), with the cache cleared between
  stages.

On Windows, running out of GPU memory doesn't raise an error. The driver silently spills into
system RAM, and training slows down about 20×. With extractor batch 8, one run spilled 13 GB.
If you change batch sizes, watch "Shared GPU memory" in Task Manager. `RUN = "quick"` runs on a
CPU.

**What is not in the repo** (see `.gitignore`):
- `data/`: datasets. Rebuild them with the commands in §5.
- `artifacts/`: trained weights. Rebuild them with the notebook or `train.py`.
- `*.safetensors`, `*.pt`, `*.bin`, `*.ckpt`: weight files in general.
- Virtual environments and caches.

---

## 5. Data

```bash
# 1. Company Documents: download the Kaggle zip and unzip to data/raw/company_docs/
python prepare_data.py            # PDFs -> page JPGs + docTR OCR TSVs (+Tags) + manifest   (data/company_docs/)
# 2. Synthetic bank documents: generated automatically on first use       (data/synthetic/)
# 3. Noisy copies of both, re-OCR'd (about 25 min on a GPU)
python prepare_noisy.py           # -> data/company_docs_noisy/, data/synthetic_noisy/
```

`CONFIG.yml` → `data.manifests` lists the manifests that are trained on together. The default is
the two `_noisy` sets. To use the clean versions, swap in the commented-out lines.

### Company Documents

The [Company Documents Dataset](https://www.kaggle.com/datasets/ayoubcherguelaine/company-documents-dataset)
(Kaggle, Apache-2.0) has 2,676 PDFs generated from the Northwind database.

| Class | Source files | Pages | Fields extracted (* = list) |
|---|---|---|---|
| `invoice` | `invoice_<order>.pdf` (830) | 1 | order_id, customer_id, order_date, contact_name, address, city, postal_code, country, phone, fax, product_id*, product_name*, quantity*, unit_price*, total_price |
| `purchase_order` | `purchase_orders_<order>.pdf` (830) | 1 | order_id, order_date, customer_name, product_id*, product_name*, quantity*, unit_price* |
| `shipping_order` | `order_<order>.pdf` (809) | mostly 2 | order_id, ship_name, ship_address, ship_city, ship_region, ship_postal_code, ship_country, customer_id, customer_name, employee_name, shipper_id, shipper_name, order_date, shipped_date, product_name*, quantity*, unit_price*, line_total*, total_price |
| `inventory_report` | `StockReport_<YYYY-MM>[_<category>].pdf` (207) | 1 | report_month, category*, category_id, product_name*, units_sold*, units_in_stock*, unit_price* |

**Bundles.** Files that share their identifying number form one bundle:
- 830 order packets: the invoice, purchase order and shipping order for one order ID.
- 23 monthly stock packets: the monthly report plus 8 per-category reports for one month.

The order of documents inside a bundle is shuffled, so the classifier learns page types and
boundaries from content, not from position.

**How `prepare_data.py` builds OCR and labels** (`lmv3docs/company_docs.py`):
1. Render each page with pypdfium2.
2. OCR the page with docTR (`fast_base` + `crnn_vgg16_bn`, or the mobile models with `--fast-ocr`).
3. Tag the PDF's exact text layer by layout.
4. Copy each tag to the OCR word that overlaps it, so labels don't depend on OCR spelling.

### Synthetic bank documents

`lmv3docs/synthetic.py` generates 1,500 bundles (3,710 pages): bank statements, cheques, pay stubs,
utility bills and `other` pages, with word-level tags.

### Data format

**OCR**: one TSV per page in the AWS Textract layout. Only `WORD` rows are used. Coordinates are
0–1 ratios. `Tag` holds a plain field name, or `O`/empty for none.

```
BlockType  Page  Text      Confidence  Left    Top     Width   Height  Tag
WORD       1     Account   99.1        0.0712  0.1023  0.0611  0.0118  O
WORD       1     12345678  99.6        0.1401  0.1023  0.0790  0.0118  account_number
```

**Manifest**: JSON Lines, one row per page. Paths are relative to the manifest.

```json
{"doc_id": "order_10248", "page_num": 2, "image": "pages/....jpg", "ocr": "pages/....tsv", "doc_type": "shipping_order", "segment_start": true, "split": "train"}
```

- `doc_id` groups pages into a bundle.
- `segment_start` marks where a new document begins.
- `split` is optional. Without it, rows are split by a hash of `doc_id`.

To use your own documents, write the same manifest and TSVs and add the manifest to
`data.manifests`.

---

## 6. Training

```bash
python train.py                                                 # pretrain → classifier → all extractors → evaluate → save
python train.py --set pretrain.steps=500 --set lora.r=32        # override any CONFIG.yml key
python train.py --from artifacts/pipeline --extractor invoice   # retrain ONE extractor; nothing else changes
python train.py --from artifacts/pipeline --classifier          # retrain only the classifier
```

The notebook runs the same stages, plus all evaluations and ablations (`RUN_ABLATIONS = True`,
about 90 extra minutes on an RTX 4070). The full notebook ran in 123 minutes.

`train.py` saves the serving bundle in **fp16** to `output_dir` (default `artifacts/pipeline/`).
Weights are upcast to fp32 on load.

```
backbone/   config.json, model.safetensors, tokenizer/
adapters/   cls.safetensors, ie_<doc_type>.safetensors
heads/      classifier.safetensors, ie_<doc_type>.safetensors
pipeline.json, metrics.json
```

### Key settings (`CONFIG.yml`)

| Section | Key settings |
|---|---|
| `data` | `manifests` (list); `split_ratios`; `cache_images`; `synthetic.n_bundles` |
| `model` | `name` (`microsoft/layoutlmv3-base`); `max_seq_len` |
| `pretrain` | `steps`, `batch_size`, `lr`; `mlm_ratio` / `span_p` / `max_span` (span masking); `mim_ratio`; `mlm_weight` / `mim_weight` (set one to 0 for the ablations) |
| `lora` | `r`, `alpha`, `dropout`, `targets` (q/k/v) |
| `classifier` | `doc_types`; `doc_encoder` (`transformer` or `bilstm`); `dim` / `layers` / `heads`; `max_pages` / `train_max_pages`; first-page weighting; `boundary_weight` |
| `extraction` | `epochs`, `batch_size`, learning rates; `window_overlap`; `fields` per doc type (`{name, multi, split_lines}`) |
| `serving` | `review_threshold`: low-confidence pages or fields are flagged `needs_review` |

---

## 7. Serving

```python
from lmv3docs import DocPipeline

pipe = DocPipeline.load("artifacts/pipeline", device="cuda")      # backbone loaded once, all adapters attached
pages = [pipe.page(image, ocr_tsv_text, {"doc_id": "abc", "page_num": i + 1})
         for i, (image, ocr_tsv_text) in enumerate(request_pages)]
result = pipe.process(pages)
```

`process` runs the classifier adapter once over the bundle, then switches to each segment's
extractor adapter. The result contains:
- `doc_type`, `confidence` and `segment_start` for each page;
- `segments`, where each field has a `value`, `confidence`, `page` and `bbox`;
- `needs_review`;
- `latency_ms`.

---

## 8. Package layout

```
CONFIG.yml                 all settings
prepare_data.py            Company Documents PDFs -> page JPGs + docTR OCR TSVs (+Tags) + manifest
prepare_noisy.py           degraded, re-OCR'd copies of the datasets (data/*_noisy)
train.py                   CLI: full training, or add/retrain one task on an existing pipeline
notebooks/layoutlmv3_document_ai.ipynb   executed showcase with every result above
lmv3docs/
  backbone.py      LayoutLMv3 + MIM hook, adapter management, fp16 save/load
  pretraining.py   whole-word span MLM + blockwise MIM (either can be switched off)
  lora.py          multi-adapter LoRA (many named adapters, one active)
  classifier.py    page vectors → Transformer/BiLSTM over pages → doc type + boundary
  extractor.py     per-doc-type tag head, sliding windows, entity grouping
  pipeline.py      DocPipeline: one backbone + adapters; process(); save/load; footprint
  data.py          pages → tokens + boxes + windows + images; manifests; bundles
  datasets.py      loads/merges the configured manifests
  noise.py         scan degradations, label warping, re-OCR, noise coverage
  experiments.py   ablation variants, split/memorisation audit, end-to-end value accuracy
  company_docs.py  Company Documents prep: bundles, PDF layout tagging, docTR OCR, tag transfer
  synthetic.py     synthetic bank bundles
  ocr.py           Textract-layout TSV reader
  metrics.py       classification / extraction metrics, latency
  viz.py           diagram and plots
tests/test_smoke.py
```
