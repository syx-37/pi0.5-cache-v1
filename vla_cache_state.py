"""Runtime state and metrics for pi0.5 VLA-Cache static selection.

This first integration stage is intentionally non-invasive: it estimates how
many visual tokens are reusable between consecutive frames, but it does not
modify embeddings, attention masks, KV cache, or action prediction outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class VLAOverwriteDynamicCache:
    """Dynamic-cache compatible object with position-based KV overwrite.

    Stage 2A keeps the sequence length unchanged. It reuses previous K/V only
    for visual token positions that were selected as static for the whole
    vectorized batch, and overwrites all other positions with current-frame K/V.
    """

    def __init__(self, *, reusable_token_positions=None, has_previous_cache: bool = False):
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
        self.reusable_token_positions = reusable_token_positions
        self.has_previous_cache = bool(has_previous_cache)
        self.vla_cache_last_update_stats = {}

    def __getitem__(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __iter__(self):
        for layer_idx in range(len(self.key_cache)):
            yield self[layer_idx]

    def __len__(self) -> int:
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        layer_cache = self.key_cache[layer_idx]
        if not hasattr(layer_cache, "numel") or not layer_cache.numel():
            return 0
        return int(layer_cache.shape[-2])

    @classmethod
    def from_cache(cls, cache, *, reusable_token_positions=None):
        new_cache = cls(
            reusable_token_positions=reusable_token_positions,
            has_previous_cache=cache is not None,
        )
        if cache is None:
            return new_cache

        for key_states in getattr(cache, "key_cache", []):
            new_cache.key_cache.append(key_states.detach().clone() if hasattr(key_states, "detach") else key_states)
        for value_states in getattr(cache, "value_cache", []):
            new_cache.value_cache.append(
                value_states.detach().clone() if hasattr(value_states, "detach") else value_states
            )
        if new_cache.key_cache and hasattr(new_cache.key_cache[0], "shape") and new_cache.key_cache[0].numel():
            new_cache._seen_tokens = int(new_cache.key_cache[0].shape[-2])
        else:
            new_cache._seen_tokens = int(getattr(cache, "_seen_tokens", 0))
        return new_cache

    def _ensure_layer(self, layer_idx: int):
        import torch

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.tensor([]))
            self.value_cache.append(torch.tensor([]))

    def _positions_to_keep_from_previous(self, cache_position, key_states, existing_key):
        import torch

        if (
            not self.has_previous_cache
            or self.reusable_token_positions is None
            or cache_position is None
            or cache_position.numel() <= 1
            or existing_key is None
            or not hasattr(existing_key, "numel")
            or not existing_key.numel()
        ):
            return None

        reusable = self.reusable_token_positions.to(device=cache_position.device, dtype=cache_position.dtype)
        if not reusable.numel():
            return None
        reusable = reusable[(reusable >= 0) & (reusable < existing_key.shape[-2])]
        if not reusable.numel():
            return None

        keep_previous = torch.isin(cache_position, reusable)
        if not bool(keep_previous.any().item()):
            return None
        return keep_previous

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
        import torch

        self._ensure_layer(layer_idx)

        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        if cache_position is not None:
            cache_position = cache_position.to(device=key_states.device, dtype=torch.long).flatten()

        existing_key = self.key_cache[layer_idx]
        existing_value = self.value_cache[layer_idx]
        has_existing = hasattr(existing_key, "numel") and bool(existing_key.numel())

        if cache_position is None:
            if not has_existing:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([existing_key, key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([existing_value, value_states], dim=-2)
            if layer_idx == 0:
                self._seen_tokens = self.get_seq_length(0)
                self.vla_cache_last_update_stats = {
                    "cache_hit": bool(has_existing),
                    "prefix_token_positions": int(key_states.shape[-2]),
                    "reusable_token_positions": 0,
                    "written_token_positions": int(key_states.shape[-2]),
                    "batch_size": int(key_states.shape[0]),
                }
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        needs_full_replace = (
            not has_existing
            or int(cache_position.max().item()) >= int(existing_key.shape[-2])
            or tuple(existing_key.shape[:2]) != tuple(key_states.shape[:2])
            or tuple(existing_key.shape[-1:]) != tuple(key_states.shape[-1:])
        )

        if needs_full_replace:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            written_positions = int(key_states.shape[-2])
            reusable_positions = 0
            cache_hit = False
        else:
            keep_previous = self._positions_to_keep_from_previous(cache_position, key_states, existing_key)
            if keep_previous is None:
                write_mask = torch.ones_like(cache_position, dtype=torch.bool)
            else:
                write_mask = ~keep_previous

            write_positions = cache_position[write_mask]
            write_source_indices = torch.arange(cache_position.numel(), device=key_states.device)[write_mask]
            key_out = existing_key
            value_out = existing_value
            if write_positions.numel():
                key_out = key_out.index_copy(2, write_positions, key_states.index_select(2, write_source_indices))
                value_out = value_out.index_copy(2, write_positions, value_states.index_select(2, write_source_indices))
            self.key_cache[layer_idx] = key_out
            self.value_cache[layer_idx] = value_out
            written_positions = int(write_positions.numel())
            reusable_positions = int(cache_position.numel() - write_positions.numel())
            cache_hit = bool(self.has_previous_cache)

        if layer_idx == 0:
            self._seen_tokens = max(self._seen_tokens, self.get_seq_length(0))
            batch_size = int(key_states.shape[0])
            self.vla_cache_last_update_stats = {
                "cache_hit": bool(cache_hit),
                "prefix_token_positions": int(cache_position.numel()),
                "reusable_token_positions": int(reusable_positions),
                "written_token_positions": int(written_positions),
                "reused_tokens_all_batch": int(reusable_positions * batch_size),
                "written_tokens_all_batch": int(written_positions * batch_size),
                "batch_size": batch_size,
                "real_skipped_visual_tokens": 0,
            }

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


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
    prev_past_key_values: Optional[Any] = None
    last_reusable_token_positions: Optional[Any] = None

    last_step_stats: dict = field(default_factory=dict)
    last_reset_reason: str = ""

    eval_num_steps: int = 0
    eval_baseline_visual_tokens_total: int = 0
    eval_reused_visual_tokens_total: int = 0
    eval_effective_visual_tokens_total: int = 0
    eval_real_kv_steps: int = 0
    eval_real_kv_cache_hit_steps: int = 0
    eval_real_kv_reused_tokens_total: int = 0
    eval_real_kv_written_tokens_total: int = 0

    def reset(self, reason: str = "") -> None:
        self.valid = False
        self.step_id = 0
        self.prev_prefix_embs = None
        self.prev_prefix_pad_masks = None
        self.prev_visual_token_mask = None
        self.prev_env_ids = None
        self.prev_past_key_values = None
        self.last_reusable_token_positions = None
        self.last_step_stats = {}
        self.last_reset_reason = reason

    def reset_eval_stats(self, reason: str = "") -> None:
        self.eval_num_steps = 0
        self.eval_baseline_visual_tokens_total = 0
        self.eval_reused_visual_tokens_total = 0
        self.eval_effective_visual_tokens_total = 0
        self.eval_real_kv_steps = 0
        self.eval_real_kv_cache_hit_steps = 0
        self.eval_real_kv_reused_tokens_total = 0
        self.eval_real_kv_written_tokens_total = 0
        self.valid = False
        self.step_id = 0
        self.prev_prefix_embs = None
        self.prev_prefix_pad_masks = None
        self.prev_visual_token_mask = None
        self.prev_env_ids = None
        self.prev_past_key_values = None
        self.last_reusable_token_positions = None
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
            reusable_token_positions = None
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
                    all_batch_selected = selected.all(dim=0) & visual_mask.any(dim=0)
                    if bool(all_batch_selected.any().item()):
                        reusable_token_positions = torch.nonzero(all_batch_selected, as_tuple=False).flatten()
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
            reusable_position_count = int(reusable_token_positions.numel()) if reusable_token_positions is not None else 0

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
                "real_kv_candidate_token_positions": int(reusable_position_count),
                "real_kv_candidate_tokens_all_batch": int(reusable_position_count * int(prefix_embs.shape[0])),
            }

            self.prev_prefix_embs = prefix_embs.detach().clone()
            self.prev_prefix_pad_masks = prefix_pad_masks.detach().clone()
            self.prev_visual_token_mask = visual_mask.detach().clone()
            self.last_reusable_token_positions = (
                reusable_token_positions.detach().clone() if reusable_token_positions is not None else None
            )
            self.valid = True
            self.mark_step()

            return dict(self.last_step_stats)

    def _cache_seq_len(self, cache) -> int:
        if cache is None:
            return 0
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        key_cache = getattr(cache, "key_cache", [])
        if key_cache and hasattr(key_cache[0], "shape") and key_cache[0].numel():
            return int(key_cache[0].shape[-2])
        return 0

    def prepare_real_kv_cache(self, *, expected_seq_len: int):
        prev_seq_len = self._cache_seq_len(self.prev_past_key_values)
        if self.prev_past_key_values is None or prev_seq_len != int(expected_seq_len):
            return VLAOverwriteDynamicCache()
        return VLAOverwriteDynamicCache.from_cache(
            self.prev_past_key_values,
            reusable_token_positions=self.last_reusable_token_positions,
        )

    def store_real_kv_cache(self, past_key_values) -> dict:
        update_stats = dict(getattr(past_key_values, "vla_cache_last_update_stats", {}) or {})
        self.prev_past_key_values = VLAOverwriteDynamicCache.from_cache(past_key_values)

        self.eval_real_kv_steps += 1
        if bool(update_stats.get("cache_hit", False)):
            self.eval_real_kv_cache_hit_steps += 1
        self.eval_real_kv_reused_tokens_total += int(update_stats.get("reused_tokens_all_batch", 0))
        self.eval_real_kv_written_tokens_total += int(update_stats.get("written_tokens_all_batch", 0))

        real_kv_stats = {
            "real_kv_enabled": True,
            "real_kv_mode": "overwrite_no_token_skip",
            "real_kv_cache_hit": bool(update_stats.get("cache_hit", False)),
            "real_kv_prefix_token_positions": int(update_stats.get("prefix_token_positions", 0)),
            "real_kv_reused_token_positions": int(update_stats.get("reusable_token_positions", 0)),
            "real_kv_written_token_positions": int(update_stats.get("written_token_positions", 0)),
            "real_kv_reused_tokens_all_batch": int(update_stats.get("reused_tokens_all_batch", 0)),
            "real_kv_written_tokens_all_batch": int(update_stats.get("written_tokens_all_batch", 0)),
            "real_skipped_visual_tokens": 0,
        }
        self.last_step_stats.update(real_kv_stats)
        return real_kv_stats

    def get_eval_stats(self) -> dict:
        baseline = int(self.eval_baseline_visual_tokens_total)
        reused = int(self.eval_reused_visual_tokens_total)
        effective = int(self.eval_effective_visual_tokens_total)
        real_reused = int(self.eval_real_kv_reused_tokens_total)
        real_written = int(self.eval_real_kv_written_tokens_total)
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
            "real_kv_steps": int(self.eval_real_kv_steps),
            "real_kv_cache_hit_steps": int(self.eval_real_kv_cache_hit_steps),
            "real_kv_reused_tokens_total": real_reused,
            "real_kv_written_tokens_total": real_written,
            "real_kv_reuse_rate": float(real_reused) / float(max(baseline, 1)),
            "real_skipped_visual_tokens_total": 0,
            "last_step_stats": dict(self.last_step_stats),
            "last_reset_reason": str(self.last_reset_reason),
        }
