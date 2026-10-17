"""LayoutLMv3 backbone with a masked-image-modelling hook and attachable LoRA adapters.

LayoutLMv3 encodes text tokens (with 1D positions and 2D boxes) and 16x16 image patches in ONE
Transformer, so every token sees the words around it, where they are on the page, and what the
page looks like. Output sequence = [text tokens (L)] + [image CLS] + [196 patch tokens].

MIM hook: during pretraining, the embeddings of masked patches are replaced by a learned [MASK]
vector before entering the Transformer (the hook sits on `patch_embed`).
"""
import json
import os

import torch
import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3TokenizerFast

from .lora import add_adapter, adapter_names, merged_base_key, set_adapter


class Backbone(nn.Module):
    def __init__(self, model: LayoutLMv3Model):
        super().__init__()
        self.model = model
        H = model.config.hidden_size
        self.hidden = H
        self.patch = model.config.patch_size
        self.image_size = model.config.input_size
        self.n_patches = (self.image_size // self.patch) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, H))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self._patch_mask = None
        self.model.patch_embed.register_forward_hook(self._mim_hook)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_pretrained(cls, name="microsoft/layoutlmv3-base"):
        return cls(LayoutLMv3Model.from_pretrained(name))

    @classmethod
    def tiny(cls, vocab_size):
        """Small random LayoutLMv3 for fast tests (same architecture, ~6M params)."""
        cfg = LayoutLMv3Config(vocab_size=vocab_size, hidden_size=96, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=192, coordinate_size=16, shape_size=16, input_size=224, patch_size=16)
        return cls(LayoutLMv3Model(cfg))

    # ---------------------------------------------------------------- forward
    def _mim_hook(self, module, inputs, out):
        if self._patch_mask is None:
            return out
        m = self._patch_mask.unsqueeze(-1).to(out.dtype)
        return out * (1 - m) + self.mask_token.to(out.dtype) * m

    def forward(self, input_ids, bbox, attention_mask, pixel_values, patch_mask=None, **_):
        """-> (text token states (B,L,H), image CLS state (B,H), patch states (B,196,H))."""
        self._patch_mask = patch_mask
        try:
            h = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                           pixel_values=pixel_values.to(self.mask_token.dtype)).last_hidden_state
        finally:
            self._patch_mask = None
        L = input_ids.shape[1]
        return h[:, :L], h[:, L], h[:, L + 1:]

    # ---------------------------------------------------------------- adapters
    def add_adapter(self, name, lora_cfg):
        add_adapter(self.model.encoder, name, lora_cfg["r"], lora_cfg["alpha"], lora_cfg.get("dropout", 0.0),
                    tuple(lora_cfg["targets"]))

    def set_adapter(self, name):
        set_adapter(self.model.encoder, name)

    @property
    def adapters(self):
        return adapter_names(self.model.encoder)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    # ---------------------------------------------------------------- save / load
    def base_state_dict(self):
        """Backbone weights without any LoRA adapter (keys as in a plain LayoutLMv3Model)."""
        sd = {}
        for k, v in self.model.state_dict().items():
            if ".A." in k or ".B." in k:
                continue
            sd[merged_base_key(k)] = v.detach().cpu().contiguous()
        return sd

    def save(self, out_dir, tokenizer, meta=None, dtype=None):
        from safetensors.torch import save_file
        os.makedirs(out_dir, exist_ok=True)
        self.model.config.save_pretrained(out_dir)
        sd = {k: (v.to(dtype) if dtype is not None and v.is_floating_point() else v) for k, v in self.base_state_dict().items()}
        sd["__mask_token__"] = self.mask_token.detach().to(dtype or self.mask_token.dtype).cpu().contiguous()
        save_file(sd, os.path.join(out_dir, "model.safetensors"))
        tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
        with open(os.path.join(out_dir, "backbone_meta.json"), "w") as f:
            json.dump(meta or {}, f, indent=2, default=str)

    @classmethod
    def load(cls, path, device="cpu"):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, "model.safetensors"))
        mask = sd.pop("__mask_token__", None)
        bb = cls(LayoutLMv3Model(LayoutLMv3Config.from_pretrained(path)))
        bb.model.load_state_dict({k: v.float() for k, v in sd.items()})
        if mask is not None:
            bb.mask_token.data.copy_(mask.float())
        tok = LayoutLMv3TokenizerFast.from_pretrained(os.path.join(path, "tokenizer"))
        return bb.to(device), tok


def load_tokenizer(name="microsoft/layoutlmv3-base"):
    return LayoutLMv3TokenizerFast.from_pretrained(name)
