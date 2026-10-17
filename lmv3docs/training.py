"""Small training helpers shared by all stages."""
from contextlib import nullcontext

import torch


def get_device(name="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def autocast(device):
    """bf16 autocast on GPU, no-op on CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def make_optimizer(groups, total_steps, warmup=0.06, weight_decay=0.01):
    """AdamW + linear warmup then linear decay."""
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    w = max(1, int(warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / w, max(0.0, (total_steps - s) / max(1, total_steps - w))))
    return opt, sched


def step(loss, params, opt, sched, clip=1.0):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, clip)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
