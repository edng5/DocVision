"""lmv3docs: one pretrained LayoutLMv3 backbone, many lightweight task adapters.

  Stage 1  pretrain the LayoutLMv3 backbone on your own pages (span-masked MLM + MIM)
  Stage 2  freeze it and train small task models that plug into it:
             - document classifier: LoRA 'cls' + Transformer over all pages of a bundle
             - one field extractor per doc type: LoRA 'ie_<type>' + linear tag head
  Serving  load the backbone ONCE; switch tasks by activating a different adapter
"""
from .config import load_config
from .data import Featurizer, Page, build_bundles, load_manifest
from .pipeline import DocPipeline

__all__ = ["load_config", "Featurizer", "Page", "build_bundles", "load_manifest", "DocPipeline"]
