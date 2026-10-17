"""Stage 1: domain-adaptive pretraining of the LayoutLMv3 backbone (self-supervised, no labels).

MLM  ~15% of words masked in whole-word spans (length ~ Geometric(p), capped); masked tokens become
     <mask> 80% / random 10% / unchanged 10%; predict the original tokens (head tied to embeddings).
MIM  ~40% of the 14x14 image patches masked in random blocks (learned [MASK] embedding);
     reconstruct their raw pixels from the patch outputs (SimMIM-style L1).
Both losses come from one forward pass, and every backbone weight is updated.
"""
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import collate, to_device
from .training import autocast, make_optimizer, step


class PretrainHeads(nn.Module):
    def __init__(self, hidden, vocab, patch=16):
        super().__init__()
        self.mlm_dense, self.mlm_norm = nn.Linear(hidden, hidden), nn.LayerNorm(hidden, eps=1e-5)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab))
        self.mim = nn.Linear(hidden, patch * patch * 3)


class Pretrainer:
    def __init__(self, backbone, featurizer, cfg, device, seed=13):
        self.bb, self.fz, self.cfg, self.device = backbone, featurizer, cfg, device
        self.heads = PretrainHeads(backbone.hidden, backbone.model.config.vocab_size, backbone.patch).to(device)
        self.grid = backbone.image_size // backbone.patch
        self.rng = np.random.default_rng(seed)
        self.history = defaultdict(list)

    # ---------------------------------------------------------------- masking
    def span_mask(self, item, rng):
        c = self.cfg
        wids, ids = item["word_ids"], list(item["input_ids"])
        words = sorted({w for w in wids if w is not None})
        target = max(1, round(len(words) * c["mlm_ratio"])) if words else 0
        mode_of = {}                                   # word -> replacement mode, decided once per SPAN (SpanBERT)
        while len(mode_of) < target:
            span = min(c["max_span"], int(rng.geometric(c["span_p"])))
            s = int(rng.integers(len(words)))
            r = rng.random()
            mode = "mask" if r < 0.8 else "random" if r < 0.9 else "keep"
            new = [w for w in words[s:s + span] if w not in mode_of][:target - len(mode_of)]   # stop at the budget
            for w in new:
                mode_of[w] = mode
        labels = [-100] * len(ids)
        for t, w in enumerate(wids):
            if w is not None and w in mode_of:         # whole word: every sub-token of a chosen word
                labels[t] = ids[t]
                if mode_of[w] == "mask":
                    ids[t] = self.fz.mask
                elif mode_of[w] == "random":
                    ids[t] = int(rng.integers(5, self.fz.tok.vocab_size))
        return ids, labels

    def patch_mask(self, rng):
        G = self.grid
        m = np.zeros((G, G), bool)
        while m.mean() < self.cfg["mim_ratio"]:
            h, w = rng.integers(1, G // 3 + 1, size=2)
            y, x = rng.integers(0, G - h + 1), rng.integers(0, G - w + 1)
            m[y:y + h, x:x + w] = True
        return torch.from_numpy(m.reshape(-1))

    @property
    def use_mlm(self):
        return self.cfg.get("mlm_weight", 1.0) > 0

    @property
    def use_mim(self):
        return self.cfg.get("mim_weight", 1.0) > 0

    def batch(self, pages, rng):
        """Masked inputs. A disabled objective also disables its masking (ablations): the "no MLM"
        variant sees intact text, the "no MIM" variant sees intact images."""
        items = []
        for p in pages:
            wins = p.windows()
            s, e = wins[int(rng.integers(len(wins)))]
            it = p.window(s, e)
            if self.use_mlm:
                it["input_ids"], it["mlm_labels"] = self.span_mask(it, rng)
            else:
                it["mlm_labels"] = [-100] * len(it["input_ids"])
            it["pixel_values"] = p.image()
            it["patch_mask"] = self.patch_mask(rng) if self.use_mim else torch.zeros(self.grid ** 2, dtype=torch.bool)
            items.append(it)
        b = to_device(collate(items, self.fz.pad), self.device)
        if not self.use_mim:
            b["patch_mask"] = None                       # no [MASK] patches at all
        return b

    # ---------------------------------------------------------------- loss
    def patchify(self, pix):
        B, P = pix.shape[0], self.bb.patch
        return pix.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(B, self.grid ** 2, 3 * P * P)

    def loss(self, b):
        h = self.heads
        text_h, _, patch_h = self.bb(**b)
        total, stats = 0.0, {}
        if self.use_mlm:
            sel = b["mlm_labels"] != -100
            x = h.mlm_norm(F.gelu(h.mlm_dense(text_h[sel])))
            logits = F.linear(x, self.bb.model.embeddings.word_embeddings.weight) + h.mlm_bias
            mlm = F.cross_entropy(logits.float(), b["mlm_labels"][sel])
            total = total + self.cfg["mlm_weight"] * mlm
            stats.update(mlm=mlm.item(), mlm_acc=(logits.argmax(-1) == b["mlm_labels"][sel]).float().mean().item())
        if self.use_mim:
            mim = (h.mim(patch_h).float() - self.patchify(b["pixel_values"].float())).abs()[b["patch_mask"]].mean()
            total = total + self.cfg["mim_weight"] * mim
            stats["mim"] = mim.item()
        return total, stats

    # ---------------------------------------------------------------- train / eval
    @torch.no_grad()
    def evaluate(self, pages, n=32):
        self.bb.eval()
        rng, stats = np.random.default_rng(0), defaultdict(list)
        bs = self.cfg["batch_size"]
        for i in range(0, min(n, len(pages)), bs):
            with autocast(self.device):
                _, s = self.loss(self.batch(pages[i:i + bs], rng))
            for k, v in s.items():
                stats[k].append(v)
        self.bb.train()
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train(self, pages, steps=None, log_every=None, log=print):
        c = self.cfg
        steps = steps or c["steps"]
        params = list(self.bb.parameters()) + list(self.heads.parameters())
        opt, sched = make_optimizer([{"params": params, "lr": c["lr"]}], steps, c.get("warmup", 0.06))
        self.bb.train()
        t0 = time.time()
        for i in range(steps):
            idx = self.rng.choice(len(pages), min(c["batch_size"], len(pages)), replace=False)
            with autocast(self.device):
                loss, s = self.loss(self.batch([pages[j] for j in idx], self.rng))
            step(loss, params, opt, sched)
            for k, v in s.items():
                self.history[k].append(v)
            if log and (i + 1) % (log_every or max(1, steps // 6)) == 0:
                log(f"pretrain step {i + 1}/{steps}  " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()) + f"  ({time.time() - t0:.0f}s)")
        self.seconds = time.time() - t0
        return self.history
