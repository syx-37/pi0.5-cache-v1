#!/usr/bin/env python3
"""Build a small comparison table from baseline/cache evaluation logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SUCCESS_RE = re.compile(r"Final success rate:\s*([0-9.]+)%")
STATS_RE = re.compile(r"VLA_CACHE_EVAL_STATS_JSON:\s*(\{.*\})")


def parse_log(path: Path, setting: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    success_matches = SUCCESS_RE.findall(text)
    stats_matches = STATS_RE.findall(text)

    success_rate = float(success_matches[-1]) / 100.0 if success_matches else 0.0
    stats = json.loads(stats_matches[-1]) if stats_matches else {}

    baseline = int(stats.get("baseline_visual_tokens_total", 0))
    reused = int(stats.get("reused_visual_tokens_total", 0))
    effective = int(stats.get("effective_visual_tokens_total", baseline))
    compression = float(stats.get("visual_compression_rate", 0.0))

    return {
        "setting": setting,
        "success_rate": success_rate,
        "baseline_visual_tokens_total": baseline,
        "effective_visual_tokens_total": effective,
        "reused_visual_tokens_total": reused,
        "visual_compression_rate": compression,
        "avg_baseline_visual_tokens_per_step": float(stats.get("avg_baseline_visual_tokens_per_step", 0.0)),
        "avg_effective_visual_tokens_per_step": float(stats.get("avg_effective_visual_tokens_per_step", baseline)),
        "avg_reused_visual_tokens_per_step": float(stats.get("avg_reused_visual_tokens_per_step", 0.0)),
    }


def print_markdown(rows: list[dict]) -> None:
    columns = [
        "setting",
        "success_rate",
        "avg_baseline_visual_tokens_per_step",
        "avg_effective_visual_tokens_per_step",
        "avg_reused_visual_tokens_per_step",
        "visual_compression_rate",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if col in {"success_rate", "visual_compression_rate"}:
                values.append(f"{value:.2%}")
            elif isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--cache-log", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        parse_log(args.baseline_log, "baseline"),
        parse_log(args.cache_log, "vla_cache_static_selection"),
    ]
    if rows[0]["baseline_visual_tokens_total"] == 0 and rows[1]["baseline_visual_tokens_total"] > 0:
        rows[0]["baseline_visual_tokens_total"] = rows[1]["baseline_visual_tokens_total"]
        rows[0]["effective_visual_tokens_total"] = rows[1]["baseline_visual_tokens_total"]
        rows[0]["avg_baseline_visual_tokens_per_step"] = rows[1]["avg_baseline_visual_tokens_per_step"]
        rows[0]["avg_effective_visual_tokens_per_step"] = rows[1]["avg_baseline_visual_tokens_per_step"]
    print_markdown(rows)

    if args.csv_out is not None:
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
