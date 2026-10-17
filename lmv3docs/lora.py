"""Multi-adapter LoRA: many named adapters share one frozen backbone.

Each wrapped Linear keeps its frozen base weight W and a small (A, B) pair per adapter:
    y = W x + B A x * (alpha / r)
Only one adapter is active at a time. Switching tasks just changes which pair is used, with no
weight copying or reloading, so one backbone in memory serves every task.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.A, self.B, self.drop = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        self.scale = {}
        self.active = None

    def add(self, name, r, alpha, dropout):
        w = self.base.weight
        self.A[name] = nn.Linear(self.base.in_features, r, bias=False).to(w.device, w.dtype)
        self.B[name] = nn.Linear(r, self.base.out_features, bias=False).to(w.device, w.dtype)
        nn.init.kaiming_uniform_(self.A[name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.B[name].weight)            # new adapter starts as a no-op
        self.drop[name] = nn.Dropout(dropout)
        self.scale[name] = alpha / r

    def remove(self, name):
        for d in (self.A, self.B, self.drop):
            if name in d:
                del d[name]
        self.scale.pop(name, None)
        if self.active == name:
            self.active = None

    def forward(self, x):
        y = self.base(x)
        a = self.active
        if a is not None and a in self.A:
            y = y + self.B[a](self.A[a](self.drop[a](x))) * self.scale[a]
        return y


def _lora_modules(root):
    return [(n, m) for n, m in root.named_modules() if isinstance(m, LoRALinear)]


def add_adapter(root, name, r=16, alpha=32, dropout=0.05, targets=("query", "key", "value")):
    n = 0
    for parent in list(root.modules()):
        for cname, child in list(parent.named_children()):
            if cname in targets:
                if isinstance(child, nn.Linear):
                    child = LoRALinear(child)
                    setattr(parent, cname, child)
                if isinstance(child, LoRALinear) and name not in child.A:
                    child.add(name, r, alpha, dropout)
                    n += 1
    if n == 0:
        raise ValueError(f"No LoRA targets {targets} found")
    return n


def set_adapter(root, name):
    for _, m in _lora_modules(root):
        m.active = name


def remove_adapter(root, name):
    for _, m in _lora_modules(root):
        m.remove(name)


def adapter_names(root):
    mods = _lora_modules(root)
    return sorted(mods[0][1].A.keys()) if mods else []


def adapter_params(root, name):
    return [p for _, m in _lora_modules(root) if name in m.A
            for p in list(m.A[name].parameters()) + list(m.B[name].parameters())]


def adapter_state_dict(root, name):
    return {f"{path}.{ab}": getattr(m, ab)[name].weight.detach().cpu().contiguous()
            for path, m in _lora_modules(root) if name in m.A for ab in ("A", "B")}


def load_adapter(root, name, state, r, alpha, targets):
    add_adapter(root, name, r, alpha, 0.0, targets)
    mods = dict(_lora_modules(root))
    for key, t in state.items():
        path, ab = key.rsplit(".", 1)
        layer = getattr(mods[path], ab)[name]
        layer.weight.data.copy_(t.to(layer.weight.dtype))


def merged_base_key(key):
    """State-dict key of a wrapped model -> original key (drops the LoRALinear '.base' level)."""
    return key.replace(".base.", ".")


@torch.no_grad()
def adapter_switch_seconds(root, names, repeats=200):
    """Average time to switch the active adapter (the cost of 'swapping models' at serving)."""
    import time
    t = time.perf_counter()
    for i in range(repeats):
        set_adapter(root, names[i % len(names)])
    return (time.perf_counter() - t) / repeats
