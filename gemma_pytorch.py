import logging
from typing import Literal

import torch
from torch import nn
# Post-V1b: import patched classes from vendor (the openpi.models_pytorch.transformers_replace package)
# instead of from globally-patched transformers. This eliminates the cp -r
# install step. CONFIG_MAPPING / DynamicCache come from transformers (no patch).
# Also ensures transformers_replace/__init__.py runs (registers patched classes
# with transformers.AutoModel) before any downstream code calls AutoModel.from_config.
import openpi.models_pytorch.transformers_replace  # noqa: F401 — side-effect: AutoModel.register
from openpi.models_pytorch.transformers_replace.models.gemma.modeling_gemma import GemmaForCausalLM
from openpi.models_pytorch.transformers_replace.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from openpi.models_pytorch.transformers_replace.models.gemma import modeling_gemma
from transformers.cache_utils import DynamicCache
from transformers.models.auto import CONFIG_MAPPING


logger = logging.getLogger("valor")
_FORWARD_PATH_LOGGED = set()


def _log_forward_path_once(path_name: str, detail: str) -> None:
    if path_name in _FORWARD_PATH_LOGGED:
        return
    _FORWARD_PATH_LOGGED.add(path_name)
    logger.info(f"[OpenPiForward] {path_name}: {detail}")


# ── transformers 4.49.0 compatibility ──────────────────────────────────────────
# openpi's transformers_replace patches add AdaRMS support to GemmaRMSNorm
# (accepts cond= param, returns (output, gate) tuple) and _gated_residual.
# Standard transformers 4.49.0 GemmaRMSNorm only takes (x) and returns a tensor.
# These helpers normalize the API so gemma_pytorch works with both versions.
#
# _single_model_forward 替代 GemmaModel.forward() 手动迭代 decoder layers，因为：
#   1. openpi 注释掉了 embedding normalizer (hidden_states *= √hidden_size)
#   2. _update_causal_mask() 对 openpi 的 4D attention mask 处理不正确
#
# 所有 decoder layer 调用走标准 GemmaDecoderLayer.forward()（对齐 RLinf），
# 配合 config._attn_implementation="sdpa" 实现 F.scaled_dot_product_attention。
# 不再使用自定义 SDPA 函数，从而兼容 FSDP use_orig_params=False。


def _get_lm_inner_model(language_model):
    """Get the inner GemmaModel regardless of transformers version.

    New transformers (≥4.44): language_model is GemmaForCausalLM → has .model (GemmaModel)
    Old transformers (<4.44): language_model is GemmaModel directly → no .model attribute
    """
    return getattr(language_model, "model", language_model)


def _gated_residual(x, y, gate):
    """Gated residual: x + y when gate is None, x + y * gate otherwise."""
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def _compat_layernorm(layernorm, x, cond=None):
    """Call layernorm with optional AdaRMS cond, returning (output, gate) tuple."""
    if hasattr(layernorm, 'dense') and layernorm.dense is not None and cond is not None:
        return layernorm(x, cond=cond)
    result = layernorm(x)
    if isinstance(result, tuple):
        return result
    return result, None


def _compact_vla_cache_queries(hidden_states, attention_mask, position_ids, skip_plan):
    """Compact per-sample query tokens for Stage 2B visual-token skipping."""
    if not skip_plan or not skip_plan.get("enabled", False):
        return hidden_states, attention_mask, position_ids, None, None, None

    keep_mask = skip_plan.get("keep_token_mask")
    if keep_mask is None:
        return hidden_states, attention_mask, position_ids, None, None, None

    batch_size, seq_len, hidden_dim = hidden_states.shape
    keep_mask = keep_mask.to(device=hidden_states.device, dtype=torch.bool)
    if tuple(keep_mask.shape) != (batch_size, seq_len):
        return hidden_states, attention_mask, position_ids, None, None, None
    if not bool((~keep_mask).any().item()):
        return hidden_states, attention_mask, position_ids, None, None, None

    kept_counts = keep_mask.sum(dim=1)
    max_kept = int(kept_counts.max().item())
    if max_kept <= 0:
        return hidden_states, attention_mask, position_ids, None, None, None

    full_indices = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
    gather_indices = torch.zeros((batch_size, max_kept), device=hidden_states.device, dtype=torch.long)
    valid_mask = torch.zeros((batch_size, max_kept), device=hidden_states.device, dtype=torch.bool)
    for batch_idx in range(batch_size):
        indices = full_indices[keep_mask[batch_idx]]
        count = int(indices.numel())
        gather_indices[batch_idx, :count] = indices
        valid_mask[batch_idx, :count] = True

    compact_hidden = hidden_states.gather(
        1,
        gather_indices[:, :, None].expand(batch_size, max_kept, hidden_dim),
    )
    compact_position_ids = position_ids.gather(1, gather_indices).masked_fill(~valid_mask, 0)
    compact_cache_position = gather_indices.masked_fill(~valid_mask, 0)

    compact_attention_mask = attention_mask
    if attention_mask is not None and attention_mask.dim() == 4:
        gather_index = gather_indices[:, None, :, None].expand(
            batch_size,
            attention_mask.shape[1],
            max_kept,
            attention_mask.shape[-1],
        )
        compact_attention_mask = attention_mask.gather(2, gather_index)
        compact_attention_mask = torch.where(
            valid_mask[:, None, :, None],
            compact_attention_mask,
            torch.zeros_like(compact_attention_mask),
        )

    restore_state = {
        "gather_indices": gather_indices,
        "valid_mask": valid_mask,
        "prev_prefix_output": skip_plan.get("prev_prefix_output"),
        "full_seq_len": int(seq_len),
    }
    skip_stats = dict(skip_plan.get("stats", {}) or {})
    skip_stats.update(
        {
            "skip_start_layer": int(skip_plan.get("skip_start_layer", -1)),
            "max_kept_token_positions": int(kept_counts.max().item()),
            "min_kept_token_positions": int(kept_counts.min().item()),
        }
    )
    return compact_hidden, compact_attention_mask, compact_position_ids, compact_cache_position, valid_mask, (
        restore_state,
        skip_stats,
    )


def _restore_vla_cache_queries(hidden_states, restore_state):
    """Restore compacted Stage 2B hidden states to full prefix length."""
    if restore_state is None:
        return hidden_states

    gather_indices = restore_state["gather_indices"]
    valid_mask = restore_state["valid_mask"]
    prev_prefix_output = restore_state.get("prev_prefix_output")
    batch_size = hidden_states.shape[0]
    full_seq_len = int(restore_state["full_seq_len"])
    hidden_dim = hidden_states.shape[-1]

    if (
        prev_prefix_output is not None
        and tuple(prev_prefix_output.shape[:2]) == (batch_size, full_seq_len)
        and int(prev_prefix_output.shape[-1]) == hidden_dim
    ):
        full_hidden = prev_prefix_output.to(device=hidden_states.device, dtype=hidden_states.dtype).clone()
    else:
        full_hidden = hidden_states.new_zeros((batch_size, full_seq_len, hidden_dim))

    for batch_idx in range(batch_size):
        valid = valid_mask[batch_idx]
        if not bool(valid.any().item()):
            continue
        full_hidden[batch_idx, gather_indices[batch_idx, valid]] = hidden_states[batch_idx, valid]
    return full_hidden


def _single_model_forward(model, hidden_states, attention_mask, position_ids,
                          past_key_values, use_cache, adarms_cond=None,
                          gradient_checkpointing=False, vla_cache_skip_plan=None):
    """Process hidden_states through a GemmaModel's decoder layers.

    统一使用标准 GemmaDecoderLayer.forward()（对齐 RLinf），不再手动拆解 attention。
    前提：模型 config._attn_implementation 已设为 "sdpa"，标准 forward 内部使用
    F.scaled_dot_product_attention，内存效率与手写 SDPA 等价。

    Replaces GemmaModel.forward() to avoid:
    - Embedding normalizer (√hidden_size scaling, commented out in openpi patches)
    - _update_causal_mask() which transforms 4D masks incorrectly for openpi
    - Cache mutation when use_cache=False (standard GemmaAttention 已正确处理)

    gradient_checkpointing: when True, applies per-layer activation checkpointing.
      KV-cache generation (use_cache=True) is excluded — cache.update() 在 GC
      recomputation 时会重复 append，需避免。
    """
    # Position embeddings (shared across all layers)
    position_embeddings = model.rotary_emb(hidden_states, position_ids)

    # Create cache for storing KV if use_cache=True
    cache = None
    if use_cache:
        cache = DynamicCache() if past_key_values is None else past_key_values
        if hasattr(cache, "vla_cache_update_mask"):
            cache.vla_cache_update_mask = None
        if hasattr(cache, "vla_cache_skip_stats"):
            cache.vla_cache_skip_stats = {}
    cache_position = None
    if use_cache:
        cache_position = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        )

    # kv_for_layers: cache (prefix write), past_key_values (suffix read-only), or None
    kv_for_layers = cache if use_cache else past_key_values

    # GC only for non-cache, no-external-KV paths.
    #   - use_cache=True: cache.update() 不能被 recompute (会写两次)
    #   - past_key_values is not None (suffix read-only path): transformers
    #     ≥4.53 GemmaAttention.forward 无条件调 past_key_value.update(),
    #     即使 use_cache=False。GC recompute 会让 layer 的 update() 在
    #     forward+backward 内被调用两次, 第二次 cat 让 key_states 长度从
    #     prefix+suffix 变成 prefix+2*suffix, 与预先构造的 attention_mask
    #     维度不匹配 → "tensor a (prefix+2*suffix) vs tensor b (prefix+suffix)"。
    #     suffix 一般只有几十 token, GC 省的 activation 也小, 直接禁掉。
    apply_gc = (
        gradient_checkpointing
        and not use_cache
        and past_key_values is None
    )
    skip_start_layer = None
    if use_cache and vla_cache_skip_plan and vla_cache_skip_plan.get("enabled", False):
        skip_start_layer = int(vla_cache_skip_plan.get("skip_start_layer", 2))
    restore_state = None
    compacted = False

    for idx, decoder_layer in enumerate(model.layers):
        if use_cache and skip_start_layer is not None and not compacted and idx >= skip_start_layer:
            (
                hidden_states,
                attention_mask,
                position_ids,
                cache_position,
                update_mask,
                restore_bundle,
            ) = _compact_vla_cache_queries(hidden_states, attention_mask, position_ids, vla_cache_skip_plan)
            if restore_bundle is not None:
                restore_state, skip_stats = restore_bundle
                compacted = True
                position_embeddings = model.rotary_emb(hidden_states, position_ids)
                if hasattr(cache, "vla_cache_update_mask"):
                    cache.vla_cache_update_mask = update_mask
                if hasattr(cache, "vla_cache_skip_stats"):
                    cache.vla_cache_skip_stats = skip_stats

        if apply_gc:
            # Per-layer gradient checkpointing.
            # _idx=idx 冻结循环变量，避免 Python closure-in-loop 捕获 bug。
            def _ckpt_layer(hs, am, _idx=idx):
                layer = model.layers[_idx]
                return layer(
                    hs,
                    attention_mask=am,
                    position_ids=position_ids,
                    past_key_value=kv_for_layers,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                    adarms_cond=adarms_cond,
                )[0]

            hidden_states = torch.utils.checkpoint.checkpoint(
                _ckpt_layer,
                hidden_states, attention_mask,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=kv_for_layers,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
            )
            hidden_states = layer_outputs[0]

    # Final norm
    hidden_states, _ = _compat_layernorm(model.norm, hidden_states, cond=adarms_cond)
    if compacted:
        hidden_states = _restore_vla_cache_queries(hidden_states, restore_state)

    return hidden_states, cache
# ───────────────────────────────────────────────────────────────────────────────


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        # 统一使用 SDPA attention（对齐 RLinf），标准 decoder_layer.forward() 内部调用
        # F.scaled_dot_product_attention，与手写 SDPA 等价但兼容 FSDP use_orig_params=False。
        self.paligemma.language_model.config._attn_implementation = "sdpa"  # noqa: SLF001
        self.gemma_expert.model.config._attn_implementation = "sdpa"  # noqa: SLF001

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    @property
    def _vlm_model(self):
        """Inner GemmaModel, compatible with all transformers versions.

        New transformers (≥4.44): paligemma.language_model is GemmaForCausalLM → .model
        Old transformers (<4.44): paligemma.language_model is GemmaModel directly
        """
        return _get_lm_inner_model(self.paligemma.language_model)

    def embed_image(self, image: torch.Tensor):
        # transformers 4.49.0: get_image_features 直接在 PaliGemma 实例上
        # autocast 处理 vision tower 内部的 float32/bfloat16 混合精度
        device_type = image.device.type if image.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self._vlm_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | DynamicCache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        vla_cache_skip_plan: dict | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        # 确保 inputs_embeds 与模型权重 dtype 一致
        target_dtype = self.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        inputs_embeds = [
            emb.to(target_dtype) if emb is not None else None
            for emb in inputs_embeds
        ]
        # 获取 Expert 的 GC 标志（用户配置），传递给所有路径以决定 SDPA vs decoder_layer
        _expert_gc = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        )

        if inputs_embeds[1] is None:
            # Prefix-only: process through VLM language model layers
            vlm_model = self._vlm_model
            prefix_output, prefix_past_key_values = _single_model_forward(
                vlm_model, inputs_embeds[0], attention_mask, position_ids,
                past_key_values, use_cache, adarms_cond=adarms_cond[0],
                gradient_checkpointing=_expert_gc,
                vla_cache_skip_plan=vla_cache_skip_plan,
            )
            suffix_output = None
        elif inputs_embeds[0] is None:
            # Suffix-only: process through Expert model layers (read-only past KV)
            expert_model = self.gemma_expert.model
            suffix_output, _ = _single_model_forward(
                expert_model, inputs_embeds[1], attention_mask, position_ids,
                past_key_values, use_cache=False, adarms_cond=adarms_cond[1],
                gradient_checkpointing=_expert_gc,
            )
            prefix_output = None
            prefix_past_key_values = None
        else:
            # Dual-expert: interleaved processing of both models
            models = [self._vlm_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled (respect user config, do not force-enable)
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self._vlm_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = _compat_layernorm(layer.input_layernorm, hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                    device=query_states.device, dtype=query_states.dtype,
                )
                cos, sin = self._vlm_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self._vlm_model.layers[layer_idx].self_attn.scaling

                att_output, _ = modeling_gemma.eager_attention_forward(
                    self._vlm_model.layers[layer_idx].self_attn,
                    query_states, key_states, value_states,
                    attention_mask, scaling,
                )
                head_dim = self._vlm_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    out_emb = _gated_residual(hidden_states, out_emb, gates[i])
                    after_first_residual = out_emb.clone()
                    out_emb, gate = _compat_layernorm(layer.post_attention_layernorm, out_emb, cond=adarms_cond[i])
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    out_emb = _gated_residual(after_first_residual, out_emb, gate)
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete, layer_idx, inputs_embeds,
                        attention_mask, position_ids, adarms_cond,
                        use_reentrant=False, preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = _compat_layernorm(models[i].norm, hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
