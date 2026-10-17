"""DocPipeline: ONE LayoutLMv3 backbone in memory + a classifier and per-doc-type extractors on top.

  backbone (125M params, loaded once)
     ├── adapter 'cls'           + DocClassifier head   -> page types + document boundaries
     ├── adapter 'ie_cheque'     + FieldExtractor head  -> cheque fields
     ├── adapter 'ie_pay_stub'   + FieldExtractor head  -> pay-stub fields
     └── ...

Switching from classification to a doc type's extractor only changes the active LoRA adapter
(microseconds). The serving bundle is the backbone plus a few MB per task, so adding a doc type
costs a few MB of memory, not another 500 MB model.

Saved layout (pipeline.save / DocPipeline.load):
  <dir>/backbone/            config.json, model.safetensors (LoRA-free), tokenizer/
  <dir>/adapters/<name>.safetensors
  <dir>/heads/classifier.safetensors, heads/ie_<doc_type>.safetensors
  <dir>/pipeline.json        labels, field specs, LoRA + classifier config, sequence length
"""
import base64
import json
import os
import time

import torch

from .backbone import Backbone, load_tokenizer
from .classifier import ADAPTER as CLS_ADAPTER, DocClassifier, classify_bundle
from .data import Featurizer
from .extractor import FieldExtractor, aggregate, extract_page
from .lora import adapter_params, adapter_state_dict, adapter_switch_seconds, load_adapter
from .ocr import read_tsv


class DocPipeline:
    def __init__(self, backbone, featurizer, lora_cfg, device, classifier=None, extractors=None, meta=None):
        self.backbone, self.fz, self.lora_cfg, self.device = backbone, featurizer, lora_cfg, device
        self.classifier = classifier
        self.extractors = extractors or {}
        self.meta = meta or {}

    # ---------------------------------------------------------------- build for training
    @classmethod
    def create(cls, cfg, device, tiny=False):
        tok = load_tokenizer(cfg["model"]["name"])
        bb = Backbone.tiny(len(tok)) if tiny else Backbone.from_pretrained(cfg["model"]["name"])
        fz = Featurizer(tok, cfg["model"]["max_seq_len"], bb.image_size, cache_images=cfg["data"].get("cache_images", False))
        return cls(bb.to(device), fz, cfg["lora"], device, meta={"model": cfg["model"]["name"], "tiny": tiny})

    def add_classifier(self, labels, ccfg):
        self.backbone.add_adapter(CLS_ADAPTER, self.lora_cfg)
        self.classifier = DocClassifier(self.backbone.hidden, labels, ccfg).to(self.device)
        return self.classifier

    def add_extractor(self, doc_type, specs):
        ext = FieldExtractor(self.backbone.hidden, doc_type, specs).to(self.device)
        self.backbone.add_adapter(ext.adapter, self.lora_cfg)
        self.extractors[doc_type] = ext
        return ext

    # ---------------------------------------------------------------- inference
    def page(self, image, ocr_tsv, meta=None):
        """Build a page from raw inputs: image (path / bytes / PIL / base64 str) + OCR TSV (path or text)."""
        if isinstance(image, str) and not os.path.exists(image):
            image = base64.b64decode(image)
        return self.fz.page(read_tsv(ocr_tsv), image, meta)

    def classify(self, pages):
        return classify_bundle(self.backbone, self.classifier, pages, self.device)

    def extract(self, page, doc_type, overlap=64):
        return extract_page(self.backbone, self.extractors[doc_type], page, self.device, overlap)

    def process(self, pages, review_threshold=0.85):
        """Classify every page with whole-bundle context, split into sub-documents, extract fields."""
        t0 = time.perf_counter()
        probs, bprob = self.classify(pages)
        t1 = time.perf_counter()
        labels, out_pages, segments = self.classifier.labels, [], []
        for i, p in enumerate(pages):
            conf, k = probs[i].max(-1)
            dt = labels[int(k)]
            start = i == 0 or dt != out_pages[-1]["doc_type"] or bool(bprob[i] >= 0.5)
            out_pages.append({"page_num": p.meta.get("page_num", i + 1), "doc_type": dt,
                              "confidence": round(float(conf), 4), "segment_start": start})
            if start:
                segments.append({"doc_type": dt, "pages": [], "_idx": []})
            segments[-1]["pages"].append(out_pages[-1]["page_num"])
            segments[-1]["_idx"].append(i)
        for s in segments:
            idx = s.pop("_idx")
            s["confidence"] = round(min(out_pages[i]["confidence"] for i in idx), 4)
            ext = self.extractors.get(s["doc_type"])
            s["fields"] = aggregate([(out_pages[i]["page_num"], self.extract(pages[i], s["doc_type"])) for i in idx],
                                    ext.specs) if ext else None
            low = [v for v in (s["fields"] or {}).values() if isinstance(v, dict) and v["confidence"] < review_threshold]
            s["needs_review"] = s["confidence"] < review_threshold or bool(low)
        t2 = time.perf_counter()
        return {"document_id": pages[0].meta.get("doc_id"), "pages": out_pages, "segments": segments,
                "latency_ms": {"classify": round((t1 - t0) * 1000, 1), "extract": round((t2 - t1) * 1000, 1),
                               "total": round((t2 - t0) * 1000, 1)}}

    # ---------------------------------------------------------------- footprint
    def footprint(self):
        """Parameter counts and fp16 sizes: one shared backbone vs. small per-task parts."""
        mb = lambda n: n * 2 / 2 ** 20
        rows = {"backbone (shared, loaded once)": sum(v.numel() for v in self.backbone.base_state_dict().values())}
        if self.classifier is not None:
            rows["classifier: LoRA 'cls' + doc head"] = (sum(p.numel() for p in adapter_params(self.backbone.model.encoder, CLS_ADAPTER))
                                                         + sum(p.numel() for p in self.classifier.parameters()))
        for dt, e in self.extractors.items():
            rows[f"extractor '{dt}': LoRA + head"] = (sum(p.numel() for p in adapter_params(self.backbone.model.encoder, e.adapter))
                                                      + sum(p.numel() for p in e.parameters()))
        return {k: {"params_M": v / 1e6, "fp16_MB": mb(v)} for k, v in rows.items()}

    def switch_cost_us(self):
        names = self.backbone.adapters
        return adapter_switch_seconds(self.backbone.model.encoder, names) * 1e6 if names else 0.0

    # ---------------------------------------------------------------- save / load
    def save(self, out_dir, fp16=True):
        """Save backbone, adapters and heads. fp16=True stores every floating-point tensor in
        half precision (about half the size); load() upcasts to fp32 for computation."""
        from safetensors.torch import save_file
        os.makedirs(os.path.join(out_dir, "adapters"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "heads"), exist_ok=True)
        dtype = torch.float16 if fp16 else None
        cast = lambda sd: {k: (v.to(dtype) if dtype is not None and v.is_floating_point() else v).detach().cpu().contiguous()
                           for k, v in sd.items()}
        self.backbone.save(os.path.join(out_dir, "backbone"), self.fz.tok, dict(self.meta or {}, dtype=str(dtype or "float32")), dtype)
        enc = self.backbone.model.encoder
        for name in self.backbone.adapters:
            save_file(cast(adapter_state_dict(enc, name)), os.path.join(out_dir, "adapters", f"{name}.safetensors"))
        cpu = lambda m: cast(m.state_dict())
        spec = {"lora": self.lora_cfg, "max_seq_len": self.fz.max_seq_len, "meta": self.meta, "extractors": {},
                "weights_dtype": str(dtype or "float32")}
        if self.classifier is not None:
            save_file(cpu(self.classifier), os.path.join(out_dir, "heads", "classifier.safetensors"))
            spec["classifier"] = {"labels": self.classifier.labels, "cfg": self.classifier.cfg}
        for dt, e in self.extractors.items():
            save_file(cpu(e), os.path.join(out_dir, "heads", f"ie_{dt}.safetensors"))
            spec["extractors"][dt] = e.specs
        with open(os.path.join(out_dir, "pipeline.json"), "w") as f:
            json.dump(spec, f, indent=2)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load the backbone once, then attach every saved adapter and head to it."""
        from safetensors.torch import load_file
        device = torch.device(device)
        with open(os.path.join(path, "pipeline.json")) as f:
            spec = json.load(f)
        bb, tok = Backbone.load(os.path.join(path, "backbone"), device)
        bb.freeze().eval()
        lc = spec["lora"]
        enc = bb.model.encoder
        for fn in sorted(os.listdir(os.path.join(path, "adapters"))):
            load_adapter(enc, fn[:-len(".safetensors")], load_file(os.path.join(path, "adapters", fn)),
                         lc["r"], lc["alpha"], tuple(lc["targets"]))
        pipe = cls(bb, Featurizer(tok, spec["max_seq_len"], bb.image_size), lc, device, meta=spec.get("meta"))
        if "classifier" in spec:
            clf = DocClassifier(bb.hidden, spec["classifier"]["labels"], spec["classifier"]["cfg"])
            clf.load_state_dict(load_file(os.path.join(path, "heads", "classifier.safetensors")))
            pipe.classifier = clf.to(device).eval()
        for dt, specs in spec["extractors"].items():
            e = FieldExtractor(bb.hidden, dt, specs)
            e.load_state_dict(load_file(os.path.join(path, "heads", f"ie_{dt}.safetensors")))
            pipe.extractors[dt] = e.to(device).eval()
        return pipe
