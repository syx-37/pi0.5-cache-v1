"""Runtime state and metrics for pi0.5 VLA-Cache static selection.

This first integration stage is intentionally non-invasive: it estimates how
many visual tokens are reusable between consecutive frames, but it does not
modify embeddings, attention masks, KV cache, or action prediction outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VLACacheState:
    enabled: bool = False
    stage: str = "token_stats"
    sim_threshold: float = 0.996
    log_interval: int = 50

    valid: bool = False
    step_id: int = 0
    prev_prefix_embs: Optional[Any] = None
    prev_prefix_pad_masks: Optional[Any] = None
    prev_visual_token_mask: Optional[Any] = None
    prev_env_ids: Optional[Any] = None

    last_step_stats: dict = field(default_factory=dict)
    last_reset_reason: str = ""

    eval_num_steps: int = 0
    eval_baseline_visual_tokens_total: int = 0
    eval_reused_visual_tokens_total: int = 0
    eval_effective_visual_tokens_total: int = 0

    def reset(self, reason: str = "") -> None:
        self.valid = False
        self.step_id = 0
        self.prev_prefix_embs = None
        self.prev_prefix_pad_masks = None
        self.prev_visual_token_mask = None
        self.prev_env_ids = None
        self.last_step_stats = {}
        self.last_reset_reason = reason

    def reset_eval_stats(self, reason: str = "") -> None:
        self.eval_num_steps = 0
        self.eval_baseline_visual_tokens_total = 0
        self.eval_reused_visual_tokens_total = 0
        self.eval_effective_visual_tokens_total = 0
        self.last_step_stats = {}
        self.last_reset_reason = reason

    def mark_step(self) -> None:
        self.step_id += 1

    def _build_visual_token_mask(self, prefix_pad_masks, img_masks, image_token_counts):
        import torch

        batch_size = int(prefix_pad_masks.shape[0])
        visual_mask = torch.zeros_like(prefix_pad_masks, dtype=torch.bool)
        camera_ranges = []
        cursor = 0

        for camera_idx, token_count in enumerate(image_token_counts):
            token_count = int(token_count)
            start, end = cursor, cursor + token_count
            cursor = end
            if token_count <= 0 or start >= visual_mask.shape[1]:
                continue

            end = min(end, visual_mask.shape[1])
            width = end - start
            if camera_idx < len(img_masks):
                camera_valid = img_masks[camera_idx].to(device=prefix_pad_masks.device, dtype=torch.bool)
                camera_valid = camera_valid.reshape(batch_size, 1).expand(batch_size, width)
            else:
                camera_valid = torch.ones((batch_size, width), device=prefix_pad_masks.device, dtype=torch.bool)

            token_valid = prefix_pad_masks[:, start:end].to(dtype=torch.bool)
            visual_mask[:, start:end] = camera_valid & token_valid
            camera_ranges.append(
                {
                    "camera_idx": int(camera_idx),
                    "start": int(start),
                    "end": int(end),
                    "token_count": int(width),
                    "valid_batch_count": int(camera_valid[:, 0].sum().item()) if width > 0 else 0,
                }
            )

        return visual_mask, camera_ranges

    def record_prefix_step(
        self,
        *,
        prefix_embs,
        prefix_pad_masks,
        img_masks,
        image_token_counts,
        sim_threshold: float | None = None,
    ) -> dict:
        """Record one inference step and estimate reusable visual tokens.

        The returned metrics are theoretical static-selection metrics. They are
        safe to collect during evaluation because this method only reads tensors
        and stores detached copies for the next environment step.
        """
        import torch

        threshold = float(self.sim_threshold if sim_threshold is None else sim_threshold)

        with torch.no_grad():
            visual_mask, camera_ranges = self._build_visual_token_mask(prefix_pad_masks, img_masks, image_token_counts)
            baseline_visual_tokens = int(visual_mask.sum().item())

            reused_visual_tokens = 0
            similarity_mean = None
            similarity_min = None
            similarity_q10 = None
            status = "first_step_no_prev_cache"

            if (
                self.prev_prefix_embs is not None
                and self.prev_visual_token_mask is not None
                and tuple(self.prev_prefix_embs.shape) == tuple(prefix_embs.shape)
                and tuple(self.prev_visual_token_mask.shape) == tuple(visual_mask.shape)
            ):
                prev = self.prev_prefix_embs.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
                prev_visual_mask = self.prev_visual_token_mask.to(device=visual_mask.device, dtype=torch.bool)
                comparable_mask = visual_mask & prev_visual_mask

                if bool(comparable_mask.any().item()):
                    cur_f = torch.nn.functional.normalize(prefix_embs.float(), dim=-1)
                    prev_f = torch.nn.functional.normalize(prev.float(), dim=-1)
                    sim = (cur_f * prev_f).sum(dim=-1)
                    selected = comparable_mask & (sim >= threshold)
                    reused_visual_tokens = int(selected.sum().item())
                    comparable_sim = sim[comparable_mask]
                    similarity_mean = float(comparable_sim.mean().item())
                    similarity_min = float(comparable_sim.min().item())
                    similarity_q10 = float(torch.quantile(comparable_sim.float(), 0.10).item())
                    status = "ok_static_selection"
                else:
                    status = "no_comparable_visual_tokens"
            elif self.prev_prefix_embs is not None:
                status = "shape_mismatch_reset_selection"

            effective_visual_tokens = max(baseline_visual_tokens - reused_visual_tokens, 0)
            compression_rate = float(reused_visual_tokens) / float(max(baseline_visual_tokens, 1))

            self.eval_num_steps += 1
            self.eval_baseline_visual_tokens_total += baseline_visual_tokens
            self.eval_reused_visual_tokens_total += reused_visual_tokens
            self.eval_effective_visual_tokens_total += effective_visual_tokens

            self.last_step_stats = {
                "enabled": bool(self.enabled),
                "stage": str(self.stage),
                "status": status,
                "step_id": int(self.step_id),
                "baseline_visual_tokens": int(baseline_visual_tokens),
                "reused_visual_tokens": int(reused_visual_tokens),
                "effective_visual_tokens": int(effective_visual_tokens),
                "visual_compression_rate": float(compression_rate),
                "similarity_threshold": float(threshold),
                "similarity_mean": similarity_mean,
                "similarity_min": similarity_min,
                "similarity_q10": similarity_q10,
                "camera_ranges": camera_ranges,
            }

            self.prev_prefix_embs = prefix_embs.detach().clone()
            self.prev_prefix_pad_masks = prefix_pad_masks.detach().clone()
            self.prev_visual_token_mask = visual_mask.detach().clone()
            self.valid = True
            self.mark_step()

            return dict(self.last_step_stats)

    def get_eval_stats(self) -> dict:
        baseline = int(self.eval_baseline_visual_tokens_total)
        reused = int(self.eval_reused_visual_tokens_total)
        effective = int(self.eval_effective_visual_tokens_total)
        return {
            "enabled": bool(self.enabled),
            "stage": str(self.stage),
            "eval_num_steps": int(self.eval_num_steps),
            "baseline_visual_tokens_total": baseline,
            "reused_visual_tokens_total": reused,
            "effective_visual_tokens_total": effective,
            "visual_compression_rate": float(reused) / float(max(baseline, 1)),
            "avg_baseline_visual_tokens_per_step": float(baseline) / float(max(self.eval_num_steps, 1)),
            "avg_reused_visual_tokens_per_step": float(reused) / float(max(self.eval_num_steps, 1)),
            "avg_effective_visual_tokens_per_step": float(effective) / float(max(self.eval_num_steps, 1)),
            "last_step_stats": dict(self.last_step_stats),
            "last_reset_reason": str(self.last_reset_reason),
        }
