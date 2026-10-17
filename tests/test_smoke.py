"""End-to-end smoke test with a tiny random LayoutLMv3 (no large download beyond the tokenizer).

    python tests/test_smoke.py

Covers: synthetic data -> pretrain (MLM+MIM) -> classifier -> extractors -> save -> reload the
pipeline (backbone once + adapters) -> process a bundle -> retrain one extractor in place.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run(*args):
    print("$ train.py", " ".join(args), flush=True)
    subprocess.run([sys.executable, "train.py", *args], cwd=ROOT, check=True)


def main():
    tmp = tempfile.mkdtemp(prefix="lmv3_smoke_")
    out = os.path.join(tmp, "pipeline")
    synthetic_types = "[bank_statement, cheque, pay_stub, utility_bill, other]"
    synthetic_fields = ("{bank_statement: [bank_name, account_number, {name: transaction_amount, multi: true}], "
                        "cheque: [payee_name, amount_numeric], pay_stub: [employee_name, net_pay], utility_bill: [customer_name, amount_due]}")
    common = ["--tiny", "--synthetic", "--device", "cpu",
              "--set", f"classifier.doc_types={synthetic_types}", "--set", f"extraction.fields={synthetic_fields}",
              "--set", f"data.manifests=[{os.path.join(tmp, 'synthetic', 'manifest.jsonl')!r}]",
              "--set", "data.synthetic.n_bundles=16", "--set", "data.synthetic.max_rows=8",
              "--set", "data.cache_images=true", "--set", "model.max_seq_len=128",
              "--set", "pretrain.steps=3", "--set", "pretrain.batch_size=2",
              "--set", "classifier.epochs=1", "--set", "extraction.epochs=1", "--set", "extraction.batch_size=4",
              "--set", f"output_dir={out}"]
    run(*common)
    run(*common, "--from", out, "--extractor", "cheque")

    from lmv3docs import DocPipeline
    pipe = DocPipeline.load(out, "cpu")
    assert set(pipe.backbone.adapters) >= {"cls", "ie_cheque", "ie_bank_statement"}, pipe.backbone.adapters
    rows = [json.loads(l) for l in open(os.path.join(tmp, "synthetic", "manifest.jsonl"))]
    doc = [r for r in rows if r["doc_id"] == rows[0]["doc_id"]]
    pages = [pipe.page(os.path.join(tmp, "synthetic", r["image"]), os.path.join(tmp, "synthetic", r["ocr"]),
                       {"doc_id": r["doc_id"], "page_num": r["page_num"]}) for r in doc]
    res = pipe.process(pages)
    assert len(res["pages"]) == len(doc) and res["segments"]
    print(json.dumps(res, indent=1)[:800])
    print("footprint:", {k: round(v["fp16_MB"], 2) for k, v in pipe.footprint().items()})
    print(f"adapter switch: {pipe.switch_cost_us():.1f} us")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
