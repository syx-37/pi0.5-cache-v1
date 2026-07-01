"""Pi0 / Pi0.5 VLA model adapter.

Importing this module registers "pi0" into VLA_MODEL_REGISTRY so that
get_vla_model("pi0", config) works as expected.
"""
from .pi0_model import load_pi0_model, OpenPi0ForRLActionPrediction
from .vla_cache_state import VLACacheState

from models.base import VLAModelBase
from models.registry import register_vla_model

import torch
import torch.nn as nn


@register_vla_model("pi0")
class Pi0Adapter(VLAModelBase):
    """Adapter that bridges the Pi0 model to the VLAModelBase interface.

    Responsibilities:
      - load_model(): call load_pi0_model(config) to build the model and load
        pretrained weights from disk; return the nn.Module for FSDP wrapping
      - compute_sft_loss(): delegate to model.sft_forward() for SFT denoising loss
      - forward_policy(): delegate to model.default_forward() for PPO update
        logprob/value/entropy recomputation
      - predict_action_batch(): delegate to model.predict_action_batch() for
        action sampling during rollout

    Registry registration is triggered automatically when `import models.pi0`
    is executed (e.g., at the start of a training worker).
    """

    def __init__(self, config):
        # Store config; actual model loading is deferred to load_model()
        # to keep import-time cost near zero (weights are GB-scale).
        self.config = config
        self._model: OpenPi0ForRLActionPrediction | None = None
        self._vla_cache_state = VLACacheState(
            enabled=bool(self.config.get("use_vla_cache", False)),
            stage=str(self.config.get("vla_cache_stage", "token_stats")),
            sim_threshold=float(self.config.get("vla_cache_sim_threshold", 0.996)),
            log_interval=int(self.config.get("vla_cache_log_interval", 50)),
        )
        self._vla_cache_logged = False

    # ------------------------------------------------------------------
    # Required: loading, adapter, and inference
    # ------------------------------------------------------------------

    def load_model(self, config=None) -> nn.Module:
        """Load Pi0 weights and return the nn.Module for FSDP wrapping."""
        cfg = config if config is not None else self.config
        self._model = load_pi0_model(cfg)
        return self._model

    def get_model_adapter(self):
        """Return the Pi0ModelAdapter for UnifiedObs ↔ Pi0 format conversion."""
        from models.pi0.pi0_model_adapter import Pi0ModelAdapter
        assert self._model is not None, "call load_model() before get_model_adapter()"
        return Pi0ModelAdapter(self._model)

    def reset_vla_cache(self, reason: str = "") -> None:
        if hasattr(self, "_vla_cache_state") and self._vla_cache_state is not None:
            self._vla_cache_state.reset(reason=reason)
            self._vla_cache_state.enabled = bool(self.config.get("use_vla_cache", False))
            self._vla_cache_state.stage = str(self.config.get("vla_cache_stage", "token_stats"))
            self._vla_cache_state.sim_threshold = float(self.config.get("vla_cache_sim_threshold", 0.996))

    def reset_vla_cache_eval_stats(self, reason: str = "") -> None:
        if hasattr(self, "_vla_cache_state") and self._vla_cache_state is not None:
            self._vla_cache_state.reset_eval_stats(reason=reason)

    def get_vla_cache_eval_stats(self) -> dict:
        if not hasattr(self, "_vla_cache_state") or self._vla_cache_state is None:
            return {}
        return self._vla_cache_state.get_eval_stats()

    def reset_episode(self, *args, **kwargs) -> None:
        self.reset_vla_cache(reason="episode_boundary")

    def predict_action_batch(self, env_obs: dict, profile=None, **kwargs) -> tuple:
        """Rollout phase: sample actions from unified observations.

        If ``profile`` is provided, ``env_obs`` is treated as BatchedUnifiedObs
        and format conversion (UnifiedObs → Pi0 internal dict) is done
        internally via Pi0ModelAdapter.  This keeps rollout workers free of
        Pi0-specific adapter calls.

        If ``profile`` is None, ``env_obs`` is passed directly to the model
        (legacy path / already-converted input).
        """
        assert self._model is not None, "call load_model() before predict_action_batch()"

        cfg_use_vla_cache = bool(self.config.get("use_vla_cache", False))
        use_vla_cache = bool(kwargs.pop("use_vla_cache", cfg_use_vla_cache))
        vla_cache_state = kwargs.pop("vla_cache_state", None)

        if vla_cache_state is None:
            vla_cache_state = self._vla_cache_state

        vla_cache_state.enabled = use_vla_cache
        vla_cache_state.stage = str(self.config.get("vla_cache_stage", "token_stats"))
        vla_cache_state.sim_threshold = float(self.config.get("vla_cache_sim_threshold", 0.996))

        if use_vla_cache:
            if not self._vla_cache_logged:
                import logging
                stage = str(self.config.get("vla_cache_stage", "token_stats"))
                real_kv_enabled = bool(stage == "real_kv" or self.config.get("vla_cache_real_kv", False))
                if real_kv_enabled:
                    message = "[VLA-Cache] Stage 2A real KV overwrite reuse enabled; token skipping is disabled."
                else:
                    message = "[VLA-Cache] visual-token compression statistics enabled; model outputs are unchanged."
                logging.getLogger("embodied_ai").info(
                    message
                )
                self._vla_cache_logged = True

        if profile is not None:
            openpi_obs = self.get_model_adapter().unified_obs_to_openpi_dict(env_obs, profile)
        else:
            openpi_obs = env_obs

        return self._model.predict_action_batch(
            openpi_obs=openpi_obs,
            use_vla_cache=use_vla_cache,
            vla_cache_state=vla_cache_state,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Optional: experiment setup (copy norm_stats to output dir)
    # ------------------------------------------------------------------

    def setup_experiment(self, output_dir: str) -> None:
        """Copy Pi0 normalization statistics into the experiment output directory.

        This makes RL checkpoints self-contained for evaluation: the evaluator
        can find norm_stats under ``<output_dir>/norm_stats/<asset_id>/``
        without needing to access the original SFT weight directory.

        Skips silently when ``model_path`` is unset or the norm_stats source
        directory does not exist (e.g. first-run without a pretrained model).

        Args:
            output_dir: Experiment root directory (e.g. ``outputs/exp_name``).
        """
        import logging
        import os
        import shutil

        _log = logging.getLogger("embodied_ai")

        model_path = self.config.get("model_path", "") or ""
        if not model_path:
            return

        # Resolve asset_id (e.g. "physical-intelligence/libero") from openpi config.
        try:
            from models.pi0.config import get_openpi_config
            config_name = self.config.get("openpi", {}).get("config_name", "pi0_libero")
            train_cfg = get_openpi_config(config_name, model_path=model_path)
            data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
            asset_id = data_cfg.asset_id
        except Exception as exc:
            _log.warning(
                f"[Pi0] setup_experiment: could not resolve asset_id, skipping norm_stats copy: {exc}"
            )
            return

        src = os.path.join(str(model_path), *asset_id.split("/"))
        dst = os.path.join(output_dir, "norm_stats", *asset_id.split("/"))

        if not os.path.isdir(src):
            _log.warning(
                f"[Pi0] setup_experiment: norm_stats source not found at '{src}', skipping."
            )
            return

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        _log.info(f"[Pi0] Copied norm_stats: {src} → {dst}")

    # ------------------------------------------------------------------
    # Optional: SFT training
    # ------------------------------------------------------------------

    def compute_sft_loss(self, model_input: dict, actions: torch.Tensor) -> torch.Tensor:
        """SFT loss via Pi0 flow-matching denoising objective.

        Delegates to OpenPi0ForRLActionPrediction.sft_forward(), which computes
        the denoising MSE loss over the full action_horizon sequence.
        """
        from openpi.models.model import Observation
        assert self._model is not None, "call load_model() before compute_sft_loss()"
        observation = Observation.from_dict(model_input)
        return self._model.sft_forward(observation, actions)

    # ------------------------------------------------------------------
    # Optional: PPO training
    # ------------------------------------------------------------------

    def forward_policy(self, forward_inputs: dict, **kwargs) -> dict:
        """PPO update phase: recompute logprobs, values, entropy.

        Delegates to OpenPi0ForRLActionPrediction.default_forward().
        """
        assert self._model is not None, "call load_model() before forward_policy()"
        return self._model.default_forward(forward_inputs, **kwargs)
