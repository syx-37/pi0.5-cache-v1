#!/usr/bin/env python3
"""Model assessment entry point with optional VLA-Cache metric printing."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _config_utils import auto_prefix_config_name

auto_prefix_config_name("pipeline/eval/")

import hydra
from omegaconf import DictConfig, open_dict


def _apply_backend_vendor_from_cfg(cfg: DictConfig) -> None:
    if os.environ.get("EMBODIED_DEVICE_VENDOR") or os.environ.get("EMBODIED_BACKEND_VENDOR"):
        return
    vendor = cfg.get("backend_vendor")
    precision_cfg = cfg.get("precision")
    if vendor is None and hasattr(precision_cfg, "get"):
        vendor = precision_cfg.get("vendor")
    if vendor:
        os.environ["EMBODIED_DEVICE_VENDOR"] = str(vendor)


def _find_vla_cache_stats(obj, *, max_depth: int = 6):
    """Best-effort search for adapter/model VLA-Cache eval stats."""
    seen = set()

    def visit(node, depth):
        if node is None or depth > max_depth:
            return None
        node_id = id(node)
        if node_id in seen:
            return None
        seen.add(node_id)

        stats_fn = getattr(node, "get_vla_cache_eval_stats", None)
        if callable(stats_fn):
            try:
                stats = stats_fn()
            except Exception:
                stats = None
            if stats:
                return stats

        if isinstance(node, dict):
            values = node.values()
        elif isinstance(node, (list, tuple)):
            values = node
        elif hasattr(node, "__dict__"):
            values = vars(node).values()
        else:
            return None

        for value in values:
            stats = visit(value, depth + 1)
            if stats:
                return stats
        return None

    return visit(obj, 0) or {}


@hydra.main(
    config_path="../configs",
    config_name="pipeline/eval/pi0_eval_libero",
    version_base=None,
)
def main(cfg: DictConfig):
    _apply_backend_vendor_from_cfg(cfg)

    from utils.logging import setup_logging

    setup_logging()

    from pipelines.eval.config_resolver import resolve_eval_config

    cfg = resolve_eval_config(cfg)

    from utils import device as device_backend

    backend_type = device_backend.get_device_type()
    num_gpus = cfg.get("num_gpus", "auto")
    if num_gpus == "auto":
        num_gpus = device_backend.device_count()
        if backend_type != "cuda":
            num_gpus = min(num_gpus, 1)

    model_cfg = cfg.get("model", {})
    model_device = model_cfg.get("device") if model_cfg is not None else None
    if backend_type != "cuda" and (not model_device or str(model_device).startswith("cuda")):
        with open_dict(cfg):
            if "model" not in cfg or cfg.model is None:
                cfg.model = {}
            cfg.model.device = str(device_backend.get_device(0))

    runner = None
    if num_gpus > 1:
        from pipelines.eval.evaluator import run_parallel_eval

        results = run_parallel_eval(cfg, num_gpus)
        vla_cache_stats = {}
    else:
        from pipelines.eval.evaluator import VLAEvaluator

        runner = VLAEvaluator(cfg)
        runner.setup()
        results = runner.run()
        vla_cache_stats = _find_vla_cache_stats(runner)

    task_metrics = results.get("task_metrics", {})
    success_rate = task_metrics.get("success_rate", 0.0)
    print(f"\nFinal success rate: {success_rate:.2%}")

    if vla_cache_stats:
        print("VLA_CACHE_EVAL_STATS_JSON: " + json.dumps(vla_cache_stats, sort_keys=True))

    return success_rate


if __name__ == "__main__":
    main()
