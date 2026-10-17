"""CONFIG.yml loading with dotted CLI overrides, and field-spec normalisation."""
import copy

import yaml


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path="CONFIG.yml", overrides=()):
    """Load YAML config. overrides: ["pretrain.steps=100", ...] (values parsed as YAML)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or ():
        key, raw = ov.split("=", 1)
        value = yaml.safe_load(raw)
        if isinstance(value, str):          # YAML 1.1 reads "2e-3" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        _set(cfg, key.strip(), value)
    return cfg


def field_specs(fields):
    """[name | {name, multi, split_lines}] -> list of dicts. split_lines defaults to multi."""
    out = []
    for f in fields:
        f = {"name": f} if isinstance(f, str) else copy.deepcopy(f)
        f.setdefault("multi", False)
        f.setdefault("split_lines", f["multi"])
        out.append(f)
    return out


def all_field_specs(cfg):
    return {dt: field_specs(fs) for dt, fs in cfg["extraction"]["fields"].items()}
