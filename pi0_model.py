"""Pi0 model implementation (VLA-RL framework core) -- adapted from RLinf.

This module contains two core components:
  1. OpenPi0ForRLActionPrediction:
     Inherits from openpi's PI0Pytorch and extends it with RL training
     capabilities:
       - ValueHead: state value estimation (PPO Critic)
       - ExploreNoiseNet: learnable exploration noise (flow-noise mode)
       - sample_actions(): action sampling with logprob computation
         (rollout phase)
       - get_log_prob_value(): compute logprob and value given action chains
         (PPO update phase)
       - default_forward(): unified entry point for PPO training forward pass

  2. load_pi0_model():
     Loads Pi0 model weights and normalization statistics from disk, and
     configures the complete transform pipeline. This function is the sole
     model initialization entry point, called by VLARolloutWorker and
     VLAPolicyWorker.

Key design decisions:
  - Flow-matching SDE sampling: action generation uses multi-step denoising,
    with random noise injected at each step. This maps to the stochasticity
    concept in RL policies; log-probability is computed via Gaussian
    distribution
  - denoise_inds mechanism: during rollout, one denoising step is randomly
    selected for logprob computation, avoiding gradient computation on all
    10 denoising steps (saves VRAM). During PPO update, the same index is
    used for recomputation
  - FSDP compatibility: get_fsdp_hints() declares wrap_classes /
    wrap_named_modules / extra_module_classes so the engine knows which
    boundaries not to split; large submodules stay on one rank to avoid
    cross-GPU communication overhead
"""

import contextlib
import glob
import logging
import math
import os
import random
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Vendored openpi: insert models/pi0/openpi into sys.path head so that
# `import openpi` resolves to the local copy without needing
# pip install git+...openpi.git.
# Multiple strategies are tried to find the directory, ensuring correct
# resolution in Ray worker processes.
# ---------------------------------------------------------------------------
def _find_vendor_openpi() -> str:
    """Find the vendored openpi directory, trying multiple paths by priority."""
    candidates = [
        # 1. Relative to current file (most common: local execution)
        os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "openpi")
        ),
    ]
    # 2. EMBODIED_AI_ROOT environment variable (Ray workers may change cwd)
    project_root = os.environ.get("EMBODIED_AI_ROOT")
    if project_root:
        candidates.append(os.path.join(project_root, "models", "pi0", "openpi"))
    # 3. Infer from package installation path
    try:
        import models as _models_pkg
        pkg_dir = os.path.dirname(os.path.abspath(_models_pkg.__file__))
        candidates.append(os.path.join(pkg_dir, "pi0", "openpi"))
    except Exception:
        pass
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError(
        f"Cannot find vendored openpi directory. Tried: {candidates}\n"
        f"Set EMBODIED_AI_ROOT env var to the project root, or ensure "
        f"models/pi0/openpi/ exists."
    )

_VENDOR_OPENPI_PATH = _find_vendor_openpi()
if _VENDOR_OPENPI_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_OPENPI_PATH)

import numpy as np
import torch
from openpi import transforms as _transforms


from models.pi0.pi0_transforms import tree_map as _tree_map
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from models.modules.dapc_mixin import DAPCMixin, DAPCModelConfig
from models.modules.explore_noise_net import ExploreNoiseNet
from models.modules.prefix_cache_mixin import PrefixCacheMixin
from models.modules.value_head import ValueHead

logger = logging.getLogger("embodied_ai")


class ForwardType(Enum):
    """Forward pass type enum, distinguishing RL training mode from SFT
    supervised fine-tuning mode.

    DEFAULT: RL training forward pass, calls default_forward(),
             computes log-prob, value, and entropy for PPO loss
    SFT: supervised fine-tuning forward pass, calls sft_forward(),
         directly computes denoising loss via parent PI0Pytorch.forward()
    """
    DEFAULT = "default"
    SFT = "sft"


class BasePolicy(ABC):
    """Abstract base interface for all policies.

    Defines the minimal interface that policy objects must implement, providing
    type safety when interacting with other framework components (e.g.
    VLARolloutWorker). In practice, OpenPi0ForRLActionPrediction implements
    both PI0Pytorch and BasePolicy through multiple inheritance.
    """

    @abstractmethod
    def default_forward(self, **kwargs): ...

    @abstractmethod
    def predict_action_batch(self, **kwargs): ...


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config, DAPCModelConfig):
    """RL-extended configuration for the Pi0 model, adding RL training-specific
    parameters on top of the original Pi0Config.

    Inherits all Pi0Config fields (model structure parameters) and extends
    with the following RL-related fields:

    Exploration noise configuration:
      - noise_method: "flow_sde" (default, schedule-based random noise),
                      "flow_cps" (cosine-sine decomposition noise),
                      "flow_noise" (learnable noise using ExploreNoiseNet)
      - noise_level: noise intensity scalar, controls exploration degree
      - noise_anneal: whether to anneal noise over training steps
                      (from high exploration to low)
      - noise_params: [start, end, steps], annealing schedule parameters

    Model structure configuration:
      - add_value_head: whether to add a ValueHead (needed for PPO, not SFT)
      - value_after_vlm: whether value head extracts features from VLM
                         (PaliGemma) output instead of expert output
      - value_vlm_mode: VLM value mode token aggregation method
                        ("mean_token"/"last_token"/"first_token")

    Log-prob computation configuration:
      - joint_logprob: whether to compute joint logprob over all denoising
                       steps (more accurate but higher VRAM usage)
      - safe_get_logprob: use simplified logprob computation (-|x-mu|^2) to
                          avoid numerical instability
      - ignore_last: exclude last step when randomly selecting logprob step

    Critic input configuration:
      - detach_critic_input: whether to detach critic input gradients
                             (prevent value loss from affecting actor)
      - chunk_critic_input: whether to use only the action chunk portion of
                            suffix_out as critic input
    """
    config_name: str = "pi0_libero"           # Corresponds to openpi_config name
    num_images_in_input: int = 2              # Number of effective cameras (LIBERO: 2 = base + wrist)
    noise_method: str = "flow_sde"            # Exploration noise method
    noise_level: float = 0.5                  # Noise intensity
    noise_anneal: bool = False                # Whether to anneal
    noise_params: list = field(default_factory=lambda: [0.7, 0.3, 400])  # Annealing params
    noise_logvar_range: list = field(default_factory=lambda: [0.08, 0.16])  # flow_noise range
    action_chunk: int = 10                    # Actual action chunk length used (<= action_horizon)
    action_env_dim: int = 7                   # Environment action dimension (override via YAML model.action_dim)
    camera_map: dict = field(default_factory=dict)  # {pi0_slot: unified_cam} override (empty = default)
    use_vla_cache: bool = False               # Enable VLA-Cache visual-token statistics / later KV reuse
    vla_cache_stage: str = "token_stats"      # token_stats/static_selection; both are stats-only in this patch
    vla_cache_sim_threshold: float = 0.996    # Cosine threshold for theoretical visual-token reuse
    vla_cache_log_interval: int = 50          # Print/cache-stat aggregation interval
    num_steps: int = 10                       # Flow-matching denoising steps
    train_expert_only: bool = False           # Whether to only train the expert network (freeze VLM)
    safe_get_logprob: bool = False            # Use simplified logprob computation
    joint_logprob: bool = False               # Whether to compute joint logprob
    ignore_last: bool = False                 # Exclude last step when sampling logprob step
    detach_critic_input: bool = False         # Whether to detach critic input gradients
    chunk_critic_input: bool = False          # Whether to use only chunk portion for value computation
    add_value_head: bool = False              # Whether to add a value head
    # DAPC-specific fields (add_denoise_value_head, tau_embed_dim, tau_condition_mode,
    # dapc_action_input, dapc_action_embed_dim) are inherited from DAPCModelConfig.
    value_after_vlm: bool = False             # Whether value head extracts from VLM output
    value_vlm_mode: str = "mean_token"        # VLM token aggregation mode
    cache_frozen_backbone: bool = True        # Enable prefix KV cache when train_expert_only=True
    cache_offload: bool = False               # Offload KV cache to CPU (saves VRAM, adds ~50ms/micro-batch)


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy, DAPCMixin, PrefixCacheMixin):
    """RL-capable Pi0 action prediction model.

    This class is the most central model class in the VLA-RL framework,
    extending openpi's PI0Pytorch into an RL policy network supporting PPO
    training through multiple inheritance:

    Architecture overview (PI0Pytorch base structure):
      PI0Pytorch internally contains:
        - paligemma_with_expert: wraps PaliGemma VLM + Gemma Expert joint model
          - PaliGemma (prefix processing): processes images and language,
            generates semantic features, outputs KV cache
          - Gemma Expert (suffix processing): processes action sequence
            denoising, leveraging prefix KV cache
        - action_out_proj: projects expert output to action velocity field
          (flow velocity)

    RL extensions (added by this class):
      - value_head (optional): extracts features from expert output to predict
        state value V(s)
      - noise_head (optional): ExploreNoiseNet, dynamically predicts
        exploration noise std
      - sample_actions(): overrides parent method to record logprob and
        denoise_inds during sampling
      - get_log_prob_value(): recomputes logprob and value given action chains

    Framework interactions:
      - VLARolloutWorker calls predict_action_batch()
        -> calls sample_actions() to generate actions and logprob
      - VLAPolicyWorker calls forward_policy() (via model.forward(DEFAULT, ...))
        -> calls default_forward()
        -> calls get_log_prob_value() to recompute logprob
        -> outputs logprobs/values/entropy for PPO loss computation
    """

    config: OpenPi0Config

    # Parameter-name prefixes that exist only in RL-trained checkpoints
    # (not in the pretrained SFT base). When an inference-only consumer loads
    # an RL checkpoint, these keys are expected extras and must be silently
    # ignored rather than rejected. Eval loader reads this attribute via
    # ``getattr(model, "RL_ONLY_PARAM_PREFIXES", ())`` so the knowledge lives
    # here (with the model) and not in the loader's hardcoded list.
    RL_ONLY_PARAM_PREFIXES: tuple[str, ...] = (
        "value_head.",          # PPO critic (always present when add_value_head=True)
        "denoise_value_head.",  # DAPC V_denoise head
        "tau_embedding.",       # DAPC noise-step conditioning
        "tau_film_proj.",       # DAPC FiLM projection
    )

    def get_fsdp_extra_module_classes(self) -> list[type]:
        """Return extra module classes for FSDP individual wrapping."""
        if hasattr(self, "value_head"):
            from models.modules.value_head import ValueHead
            return [ValueHead]
        return []

    def get_fsdp_hints(self) -> dict:
        """Unified public API the FSDP engine reads to plan wrapping.

        Consolidates under one public method what used to live as scattered
        underscore properties read directly by the FSDP engine.

        Returns a dict with keys:
          - ``wrap_classes``: transformer layer class names to wrap as FSDP
            units (same list as the ``_no_split_modules`` property — the
            property stays defined so HuggingFace ``device_map=auto`` and
            similar tools that read the HF convention keep working).
          - ``wrap_named_modules``: submodule tags (matched against the
            submodule's ``_fsdp_wrap_name`` runtime attribute) that must not
            be split across ranks — critical projection layers with small
            parameter counts where cross-GPU fragmentation would require
            all-gather on every forward pass.
          - ``extra_module_classes``: auxiliary module types (ValueHead,
            ExploreNoiseNet) that should remain standalone FSDP units.
        """
        # Pi0's own trainable submodules that need standalone FSDP wrapping
        # (small projections that would otherwise be split across ranks).
        wrap_named_modules = [
            "action_in_proj",      # Action input projection
            "action_out_proj",     # Action output projection
            "lm_head",             # Language model output head
            "state_proj",          # Robot state encoding projection
            "action_time_mlp_in",  # Time-conditioning input MLP
            "action_time_mlp_out", # Time-conditioning output MLP
            "time_mlp_in",         # Global time encoding MLP input
            "time_mlp_out",        # Global time encoding MLP output
        ]
        # Mixin-contributed wrap names: each mixin that introduces trainable
        # submodules owns its own FSDP wrap-name list. Pi0 just merges them
        # in — no need to know mixin field names directly. Adding a future
        # trainable mixin field happens in one file (the mixin), not here.
        wrap_named_modules.extend(DAPCMixin.dapc_fsdp_wrap_names())

        return {
            "wrap_classes": self._no_split_modules,
            "wrap_named_modules": wrap_named_modules,
            "extra_module_classes": self.get_fsdp_extra_module_classes(),
        }

    @property
    def _no_split_modules(self) -> list[str]:
        """FSDP no-split module list: these module types cannot be split
        across GPUs by FSDP.

        Why certain modules cannot be split:
          - Splitting GemmaDecoderLayer would cause huge cross-GPU
            communication overhead for attention computation
          - Splitting SiglipVisionEmbeddings would break spatial relationships
            between image patches
          - ValueHead and ExploreNoiseNet have small parameter counts;
            splitting them offers no benefit and adds sync overhead

        Different granularities are chosen based on train_expert_only:
          - expert only mode: GemmaDecoderLayer (entire Transformer layer)
            as minimum unit, ensuring VLM layers are not accidentally split
          - full parameter training: GemmaMLP (MLP sublayer) as minimum unit,
            allowing finer-grained memory allocation
        """
        if self.config.train_expert_only:
            # When training only the expert, VLM layers are fully frozen;
            # use layer as FSDP unit
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            # Full parameter training: use MLP as minimum FSDP unit,
            # balancing memory and communication efficiency
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        # ValueHead / ExploreNoiseNet have small parameter counts and must
        # be kept as independent FSDP units.
        # Aligns with RLinf fsdp/utils.py:186-192 ValueHead wrap policy:
        # with use_orig_params=False, FSDP flattens parameters into
        # FlatParameter; only independently wrapped modules retain
        # "value_head" in their FlatParameter name, which build_optimizer()
        # relies on to distinguish actor/critic param groups and lr.
        if hasattr(self, "value_head"):
            no_split_modules.append("ValueHead")
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    def get_model_adapter(self):
        """Return a Pi0ModelAdapter for this model.

        Allows pipelines to obtain the correct ModelAdapter without
        needing to know the model type or import Pi0ModelAdapter directly.
        Camera mapping is read from ``config.camera_map`` (set via YAML).
        """
        from models.pi0.pi0_model_adapter import Pi0ModelAdapter
        cam_map = self.config.camera_map or None  # empty dict → use default
        return Pi0ModelAdapter(self, camera_map=cam_map)

    def _vla_cache_image_token_counts(self, prefix_embs, lang_tokens, img_masks):
        """Return per-image visual-token counts in prefix order.

        This mirrors PI0Pytorch.embed_prefix(): every valid image contributes
        one SigLIP patch sequence, while the language prefix is placed after
        image tokens. The returned counts allow VLACacheState to locate visual
        token ranges without changing model outputs.
        """
        if not isinstance(img_masks, Sequence):
            return []

        counts = []
        lang_len = int(lang_tokens.shape[1]) if lang_tokens is not None and hasattr(lang_tokens, "shape") else 0
        total_prefix_len = int(prefix_embs.shape[1])
        visual_total = max(total_prefix_len - lang_len, 0)

        valid_images = 0
        for mask in img_masks:
            if mask is None:
                continue
            if torch.is_tensor(mask):
                if bool(mask.detach().bool().any().item()):
                    valid_images += 1
            elif bool(mask):
                valid_images += 1

        if valid_images <= 0:
            return counts

        per_image = visual_total // valid_images
        remainder = visual_total % valid_images
        for mask in img_masks:
            is_valid = False
            if mask is not None:
                if torch.is_tensor(mask):
                    is_valid = bool(mask.detach().bool().any().item())
                else:
                    is_valid = bool(mask)
            if is_valid:
                counts.append(per_image + (1 if remainder > 0 else 0))
                remainder = max(remainder - 1, 0)
            else:
                counts.append(0)
        return counts

    def __init__(self, config: OpenPi0Config):
        """Initialize the RL version of the Pi0 model, adding value head
        and noise network.

        Args:
            config: OpenPi0Config instance containing all model and RL
                    training hyperparameters
        """
        super().__init__(config)
        self.global_step = 0  # Current global training step, used for noise annealing
        # Resolve the attention backend ONCE (it's a process-constant env var). Default
        # to "eager": torch 2.2 mem-efficient SDPA produces NaN actions on our custom 4D
        # float attention mask (see scripts/bench_sdpa_vs_eager.py); PI0_ATTN_IMPL=sdpa is
        # reserved for benchmarking on newer torch. The forward paths still WRITE this onto
        # the sub-module configs every call (defensive against FSDP/state-dict resets) —
        # only the repeated os.environ lookup is hoisted here.
        self._attn_impl = os.environ.get("PI0_ATTN_IMPL", "eager")

        # Determine projection dimension based on value head input source
        # VLM output (prefix) has larger dimension (2048), expert output
        # (suffix) has smaller dimension (1024)
        if self.config.value_after_vlm:
            proj_width = 2048  # PaliGemma VLM output dimension
        else:
            proj_width = 1024  # Gemma Expert output dimension

        if self.config.add_value_head:
            # Choose different ValueHead sizes based on model variant
            # (Pi0 vs Pi0.5); Pi0.5 (larger model) needs wider value network
            if self.config.config_name in ["pi05_maniskill", "pi05_libero"]:
                value_head_hidden_sizes = (1024, 512, 256)  # Pi0.5 wider value network
            else:
                value_head_hidden_sizes = (512, 256, 128)   # Standard Pi0 value network
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,       # Output scalar value
                activation="relu",  # ReLU (faster than GELU, sufficient for value regression)
                bias_last=True,     # Output layer with bias
            )

        # Initialize DAPC components when denoise value head is enabled.
        # DAPCMixin is shape-agnostic — we honor its contract by computing
        # the flattened size of Pi0's x_τ (shape [B, action_horizon, action_dim])
        # and passing that int. compute_denoise_value() will then receive x_τ
        # and reshape it to [B, action_horizon*action_dim] before the Linear layer.
        if getattr(self.config, "add_denoise_value_head", False):
            action_input_dim = None
            if getattr(self.config, "dapc_action_input", False):
                # Pi0-specific: x_τ.shape[1:] == (action_horizon, action_dim)
                action_input_dim = int(self.config.action_horizon) * int(self.config.action_dim)
            self.init_dapc(
                num_steps=self.config.num_steps,
                feature_dim=proj_width,
                tau_embed_dim=getattr(self.config, "tau_embed_dim", 32),
                tau_condition_mode=getattr(self.config, "tau_condition_mode", "concat"),
                action_input_dim=action_input_dim,
                action_embed_dim=getattr(self.config, "dapc_action_embed_dim", 64),
            )

        # Record whether VLM value mode is used (requires both value_after_vlm
        # and add_value_head)
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )

        if self.config.noise_method == "flow_noise":
            # Learnable noise mode: add ExploreNoiseNet
            # Input dim 1024 = expert suffix_out dimension
            # Output dim = action_dim (Pi0 internal action dim, typically 8)
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )

        # Set _fsdp_wrap_name attribute for each submodule
        # The FSDP engine uses this attribute to identify module hierarchy
        # and decide parameter grouping strategy
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        """Update the global training step, used for noise annealing schedule.

        At the start of each PPO update round, VLATrainRunner calls this method
        to sync the step count, causing noise level to gradually decrease
        according to the preset annealing schedule (noise_params).
        """
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        """Set up the model's input/output transform pipeline.

        Called in load_pi0_model(), combining the following transforms into
        a sequential pipeline:
          inputs:  Pi0Inputs -> [DeltaActions] -> Normalize -> Tokenize -> ImageEncode
          outputs: ImageDecode -> Unnormalize -> [AbsoluteActions] -> Pi0Outputs

        Once set, predict_action_batch() automatically applies these transforms
        via input_transform()/output_transform(), converting raw environment
        observations to model input format and converting model output back
        to environment action format.

        Args:
            transforms: input-direction transform sequence (composed in order)
            output_transforms: output-direction transform sequence
        """
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform_sample(self, sample: dict) -> dict:
        """Apply the input transform compose chain to a single sample dict.

        Public per-sample entry point used by Pi0ModelAdapter, which converts
        a unified observation into Pi0's internal tensor format one sample
        at a time and then stacks the results into a batch.

        Higher-level code that starts from a batch observation should prefer
        ``input_transform(obs)`` which handles tokenization and batching
        automatically. This method is the lower-level building block.
        """
        return self._input_transform(sample)

    def output_transform_sample(self, sample: dict) -> dict:
        """Apply the output transform compose chain to a single sample dict.

        Public per-sample entry point mirroring ``input_transform_sample``.
        ``output_transform(outputs)`` is the batched counterpart.
        """
        return self._output_transform(sample)

    def input_transform(self, obs: dict, transpose=True):
        """Apply the input transform pipeline to batch observations,
        converting to model-acceptable tensor format.

        This method handles two calling scenarios (determined by whether
        "prompt" key is present):
          1. First-time processing (predict_action_batch call): obs contains
             "prompt", requiring image tokenization and tokenized_prompt
             generation
          2. Second-time processing (default_forward call): obs contains
             "tokenized_prompt", skipping language tokenization and directly
             using pre-processed tokens

        Batching mechanism: openpi transform functions are designed to process
        single samples, so transforms are applied individually to each sample
        in the batch, then results are stacked into batch tensors using
        tree_map.

        Args:
            obs: batch observation dict, batch size determined by
                 values' .shape[0]
            transpose: whether to transpose images from CHW->HWC
                       (environment output is typically CHW, openpi expects HWC)

        Returns:
            Processed batch input dict with all tensors converted to PyTorch
            Tensor format
        """
        inputs = _tree_map(lambda x: x, obs)  # Shallow copy

        # Determine if this is "first-time" processing (contains raw language
        # prompt needing tokenization)
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            # Second-time processing: keep only environment keys with "/"
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # Convert all torch Tensors to numpy (openpi transforms expect numpy).
        # bfloat16 is PyTorch-specific; convert to float32 first.
        def _to_numpy(x):
            if not torch.is_tensor(x):
                return x
            t = x.detach().cpu()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            return np.asarray(t)
        inputs = _tree_map(_to_numpy, inputs)

        # Infer batch size from any value (handle numpy, torch, or list)
        batch_size = None
        for v in inputs.values():
            if hasattr(v, "shape") and len(v.shape) > 0:
                batch_size = v.shape[0]
                break
            if isinstance(v, (list, tuple)):
                batch_size = len(v)
                break
        if batch_size is None:
            raise ValueError(
                f"Cannot infer batch size from input_transform inputs. "
                f"Keys: {list(inputs.keys())}, "
                f"Types: {[(k, type(v).__name__) for k, v in inputs.items()]}"
            )

        # Apply transforms to each sample individually
        transformed_samples = []
        for i in range(batch_size):
            sample = _tree_map(lambda x: x[i], inputs)

            if transpose:
                sample = _tree_map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )

            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                # PPO recompute path: forward_inputs doesn't carry raw prompt
                # text (only tokenized_prompt / tokenized_prompt_mask are kept).
                # The actual prompt embedding is already in the prefix KV cache;
                # this placeholder is tokenized but masked out by prefix_pad_masks.
                sample["prompt"] = "xxxx"

            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)

        # Stack all sample transform results into batch tensors
        inputs = _tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )

        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        """Apply inverse transforms to batch model outputs, converting
        normalized actions back to environment coordinate system.

        Args:
            outputs: batch dict containing "actions" and "state"

        Returns:
            Processed output dict with "actions" shaped [B, action_chunk, action_env_dim]
        """
        batch_size = outputs["actions"].shape[0]
        cpu_outputs = _tree_map(lambda x: np.asarray(x.detach().cpu()), outputs)
        transformed_samples = []
        for i in range(batch_size):
            sample = _tree_map(lambda x: x[i], cpu_outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)

        outputs = _tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, :self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """Unified forward pass entry point, dispatching based on type.

        Args:
            forward_type: ForwardType enum value
            **kwargs: parameters passed to the specific forward function
        """
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, observation, actions, **kwargs):
        """Supervised fine-tuning forward pass, computing denoising loss.

        Expects pre-processed inputs (converted via Pi0ModelAdapter):
          - observation: ``Observation`` dataclass with ``.images``,
            ``.image_masks``, ``.state``, ``.tokenized_prompt``
          - actions: ``(B, action_horizon, action_dim)`` float32 tensor

        Returns:
            Per-element MSE loss tensor of shape ``(B, action_horizon, action_dim)``.
        """
        return super().forward(observation, actions)

    @property
    def supports_prefix_cache(self) -> bool:
        # Requires frozen backbone (train_expert_only) AND not explicitly disabled
        if not getattr(self.config, 'train_expert_only', False):
            return False
        return getattr(self.config, 'cache_frozen_backbone', True)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Core PPO training forward pass: recompute logprob, value, and
        entropy.

        During the PPO update phase, given action trajectories (chains) and
        denoising indices (denoise_inds) collected from rollout, recompute
        log-probabilities under the current policy.

        Args:
            forward_inputs: dict containing:
              - "chains": action denoising trajectory,
                shape [B, num_steps+1, action_horizon, action_dim]
              - "denoise_inds": selected denoising step indices,
                shape [B, num_steps]
              - observation-related keys
            **kwargs:
              - compute_values: whether to compute values (default False)
              - prefix_kv_cache: cached VLM prefix KV from a previous epoch
                (skip prefix recomputation when frozen backbone)

        Returns:
            dict containing:
              - "logprobs": shape [B, action_chunk, action_env_dim]
              - "values": shape [B]
              - "entropy": shape [B, 1]
              - "prefix_kv_cache": VLM prefix cache (when backbone is frozen)
        """
        compute_values = kwargs.get("compute_values", False)
        prefix_kv_cache = kwargs.get("prefix_kv_cache", None)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]

        # Convert environment observations to model input format
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = _model.Observation.from_dict(observation)

        # Preprocess observation (image encoding, language tokenization)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        # Ensure all tensors are on the correct device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)

        # Core call: given action chains, compute logprobs, values, and entropy
        log_probs, value_t, entropy, denoise_vals, new_prefix_cache = (
            self.get_log_prob_value(
                images, img_masks, lang_tokens, lang_masks, state,
                chains, denoise_inds, compute_values,
                prefix_kv_cache=prefix_kv_cache,
            )
        )

        # Truncate to the actual action dimension range used
        log_probs = log_probs[:, :, :self.config.action_chunk, :self.config.action_env_dim]
        entropy = entropy[:, :, :self.config.action_chunk, :self.config.action_env_dim]
        dapc_results = self.pack_dapc_training_results(
            log_probs=log_probs,
            denoise_values=denoise_vals,
            joint_logprob=self.config.joint_logprob,
        )

        # Average over denoising step dimension (dim=1)
        log_probs = log_probs.mean(dim=1)
        # Average over all dimensions to get scalar entropy
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        # Average value over last dimension
        value_t = value_t.mean(dim=-1, keepdim=False)

        result = {"logprobs": log_probs, "values": value_t, "entropy": entropy}
        if new_prefix_cache is not None:
            result["prefix_kv_cache"] = new_prefix_cache
        result.update(dapc_results)
        # Surface DAPC old per-step logprobs from forward_inputs into output_dict,
        # so policy_worker reads from output_dict (not from forward_inputs internals)
        prev_psl = forward_inputs.get("prev_per_step_logprobs")
        if prev_psl is not None:
            result["per_step_old_logprobs"] = prev_psl
        return result

    def precision_processor(self, processed_obs):
        """Move all tensors in observations to the current model device and
        ensure memory contiguity.

        Args:
            processed_obs: observation dict containing various value types

        Returns:
            Processed observation dict with tensors on correct device
        """
        device = next(self.parameters()).device

        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item) else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(device=device).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        *,
        openpi_obs: dict,
        mode: Literal["train", "inference"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Batch action prediction, called during the rollout phase.

        This method is the rollout worker's core interface, responsible for:
          1. Converting observations to model input format
          2. Calling sample_actions() to sample actions and record logprobs
          3. Inverse-transforming model output actions to environment coords
          4. Packaging forward_inputs for PPO update phase

        Args:
            openpi_obs: openpi-keyed dict from
                ``Pi0ModelAdapter.unified_obs_to_openpi_dict()``.
            mode: "train" (exploratory sampling with noise) or
                  "inference" (deterministic, no noise)
            compute_values: whether to compute state values
            **kwargs: additional parameters

        Returns:
            (actions, result) tuple:
              - actions: np.ndarray, shape [B, action_chunk, action_env_dim]
              - result: dict with prev_logprobs, prev_values, forward_inputs
        """
        to_process_obs = openpi_obs

        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)

        observation = _model.Observation.from_dict(processed_obs)

        best_of_n = getattr(self.config, "best_of_n", 1)
        if best_of_n > 1 and mode != "train":
            outputs = self._best_of_n_sample(observation, n=best_of_n, compute_values=compute_values)
        else:
            use_vla_cache = bool(kwargs.pop("use_vla_cache", getattr(self.config, "use_vla_cache", False)))
            vla_cache_state = kwargs.pop("vla_cache_state", None)
            outputs = self.sample_actions(
                observation,
                mode=mode,
                compute_values=compute_values,
                use_vla_cache=use_vla_cache,
                vla_cache_state=vla_cache_state,
            )

        out_state = observation.state
        if isinstance(out_state, torch.Tensor):
            out_state = out_state.float()
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": out_state}
        )["actions"].numpy()

        # Package forward_inputs for PPO update (model-internal keys only,
        # no raw env-specific keys). to_process_obs has observation/*
        # keys from model_adapter.
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        # Add observation/* keys from pre-transform dict.
        # Ensure values are torch.Tensor (required by trajectory batch assembly
        # which only concatenates tensors, skipping numpy arrays).
        for k, v in to_process_obs.items():
            if k != "prompt":
                if isinstance(v, np.ndarray):
                    forward_inputs[k] = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    forward_inputs[k] = v
                # skip non-array values (e.g., list[str])

        if outputs.get("prev_per_step_logprobs") is not None:
            forward_inputs["prev_per_step_logprobs"] = outputs["prev_per_step_logprobs"]

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        if outputs.get("vla_cache_stats") is not None:
            result["vla_cache_stats"] = outputs["vla_cache_stats"]
        return actions, result

    @torch.no_grad()
    def _best_of_n_sample(self, observation, n=4, compute_values=True):
        """Sample N action candidates with different noise seeds, select best by V_denoise.

        Shares the same observation but uses different random noise for each candidate.
        Selection criterion: mean V_denoise across all denoising steps (higher = better).
        Falls back to first candidate if V_denoise is unavailable.
        """
        bon_vtime_weight = float(getattr(self.config, "best_of_n_vtime_weight", 0.0))
        need_values = compute_values or bon_vtime_weight > 0
        candidates = []
        for _ in range(n):
            result = self.sample_actions(observation, mode="inference", compute_values=need_values)
            candidates.append(result)

        # Score each candidate by V_denoise (configurable reduction over τ-axis),
        # optionally blended with V_time for robustness when V_denoise is unreliable.
        reduce = getattr(self.config, "best_of_n_score_reduce", "mean")
        scores = []
        for c in candidates:
            dv = c.get("prev_denoise_values")
            vt = c.get("prev_values")
            if dv is not None and dv.numel() > 0:
                if reduce == "mean":
                    s = dv.mean(dim=-1)                          # [B]
                elif reduce == "last":
                    s = dv[..., -1]                              # [B] (τ=K-1)
                elif reduce == "max":
                    s = dv.max(dim=-1).values                    # [B]
                else:
                    raise ValueError(
                        f"Unknown best_of_n_score_reduce='{reduce}'. "
                        f"Expected one of: mean, last, max."
                    )
                if bon_vtime_weight > 0 and vt is not None and vt.numel() > 0:
                    vt_flat = vt.view(-1) if vt.dim() > 1 else vt
                    s = (1.0 - bon_vtime_weight) * s + bon_vtime_weight * vt_flat
                scores.append(s)
            else:
                scores.append(torch.zeros(c["actions"].shape[0], device=c["actions"].device))

        scores_stacked = torch.stack(scores, dim=0)  # [N, B]
        best_idx = scores_stacked.argmax(dim=0)  # [B]
        batch_size = best_idx.shape[0]
        batch_arange = torch.arange(batch_size, device=best_idx.device)

        # Select best candidate per batch element using advanced indexing
        best_result = {}
        for key in candidates[0]:
            vals = [c[key] for c in candidates]
            if isinstance(vals[0], torch.Tensor):
                stacked = torch.stack(vals, dim=0)  # [N, B, ...]
                best_result[key] = stacked[best_idx, batch_arange]  # [B, ...]
            else:
                best_result[key] = vals[0]  # non-tensor: use first

        return best_result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
        use_vla_cache: bool = False,
        vla_cache_state=None,
    ) -> torch.Tensor:
        """Sample actions via flow-matching denoising, recording logprobs
        and values.

        This is the most central sampling function in the RL framework:
          1. Starting from pure noise x_T, generate actions x_0 through
             num_steps denoising steps
          2. Compute logprob at each denoising step
          3. Use denoise_inds mechanism to select one step's logprob as
             the policy's logprob

        Args:
            observation: Observation object with state, images, lang_tokens
            noise: optional initial noise tensor; Gaussian if None
            mode: "train" (random exploration) or other (deterministic)
            compute_values: whether to compute state values

        Returns:
            Dict with "actions", "chains", "prev_logprobs", "prev_values",
            "denoise_inds"
        """
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Prefix processing phase (PaliGemma VLM)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Default to "eager" attention: torch 2.2 mem_efficient SDPA backend produces
        # NaN actions on our custom 4D float attention mask (confirmed bug; see
        # scripts/bench_sdpa_vs_eager.py). PI0_ATTN_IMPL=sdpa is reserved for
        # benchmarking on newer torch where this may be fixed.
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = self._attn_impl

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        vla_cache_stats = None
        if use_vla_cache and vla_cache_state is not None:
            stage = getattr(vla_cache_state, "stage", getattr(self.config, "vla_cache_stage", "token_stats"))
            if stage in {"token_stats", "static_selection"}:
                image_token_counts = self._vla_cache_image_token_counts(prefix_embs, lang_tokens, img_masks)
                sim_threshold = float(
                    getattr(vla_cache_state, "sim_threshold", getattr(self.config, "vla_cache_sim_threshold", 0.996))
                )
                vla_cache_stats = vla_cache_state.record_prefix_step(
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    img_masks=img_masks,
                    image_token_counts=image_token_counts,
                    sim_threshold=sim_threshold,
                )

        # Denoising phase: step by step from x_T to x_0
        x_t = noise
        chains = []
        log_probs = []
        values = []
        self.reset_dapc_collector()
        chains.append(x_t)

        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)

        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # Determine denoise_inds
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor([random.randint(0, num_steps - 2)] * num_steps)
                else:
                    denoise_inds = torch.tensor([random.randint(0, num_steps - 1)] * num_steps)
        else:
            denoise_inds = torch.tensor([-1] * num_steps)

        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # Step-by-step denoising loop
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "not_train"

            x_t_mean, x_t_std, value_t, denoise_value_t = self.sample_mean_var_val(
                x_t, idx, state, prefix_pad_masks, past_key_values,
                sample_mode, num_steps, compute_values,
            )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            values.append(value_t)
            self.collect_dapc_step(denoise_value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)

        log_probs_stacked = torch.stack(log_probs, dim=1)[
            :, :, :self.config.action_chunk, :self.config.action_env_dim
        ]

        if self.config.joint_logprob:
            log_probs = log_probs_stacked.mean(dim=1)
        else:
            log_probs = log_probs_stacked[torch.arange(log_probs_stacked.shape[0]), denoise_inds[:, 0]]

        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        result = {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }
        result.update(
            self.pack_dapc_rollout_results(
                log_probs_stacked=log_probs_stacked,
                joint_logprob=self.config.joint_logprob,
            )
        )
        if vla_cache_stats is not None:
            result["vla_cache_stats"] = vla_cache_stats
        return result

    def sample_mean_var_val(
        self, x_t, idx, state, prefix_pad_masks, past_key_values,
        mode, denoise_steps, compute_values=True,
    ):
        """Compute mean, standard deviation, and value estimate for a single
        denoising step.

        Args:
            x_t: action state at current timestep
            idx: current denoising step index
            state: robot proprioceptive state
            prefix_pad_masks: VLM prefix padding mask
            past_key_values: VLM prefix KV cache
            mode: "not_train" (deterministic) or "train" (stochastic)
            denoise_steps: total denoising steps
            compute_values: whether to compute value estimates

        Returns:
            (x_t_mean, x_t_std, value_t, denoise_value_t) tuple
        """
        bsize = state.shape[0]
        device = state.device

        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        # Compute noise intensity (with annealing schedule support)
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps) / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.config.noise_level).to(device)

        # Build time schedule: from t=1 (pure noise) to t=0 (clean action)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])

        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        # Expert network forward pass: predict flow velocity v_t
        suffix_out = self.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_input)
        v_t = self.action_out_proj(suffix_out)

        # Value estimation (if needed). V_time and V_denoise are independent:
        # eval-time BoN uses V_denoise alone, so the two heads must be gated
        # separately (the original combined gate silently zeroed V_denoise
        # whenever V_time was absent, which broke BoN scoring at inference).
        need_features = compute_values and not self.config.value_after_vlm and (
            self.config.add_value_head or getattr(self.config, "add_denoise_value_head", False)
        )
        if need_features:
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(suffix_out[:, :self.config.action_chunk], dim=1, keepdim=False)
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
        else:
            suffix_out_value = None

        if self.config.add_value_head and suffix_out_value is not None:
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)

        if getattr(self.config, "add_denoise_value_head", False) and suffix_out_value is not None:
            denoise_value_t = self.compute_denoise_value(suffix_out_value, idx, x_tau=x_t)
        else:
            denoise_value_t = torch.zeros((bsize), device=device)

        # Compute predicted x_0 and x_1
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)

        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        # Compute next step mean and std based on noise method
        if mode != "train":
            # Deterministic: ODE flow, no random noise
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)

        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                # Flow SDE: DDPM-style noise schedule
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)

                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)

                x_t_std = torch.sqrt(delta) * sigma_i

            elif self.config.noise_method == "flow_cps":
                # Cosine-Phase SDE noise
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)

                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term

            elif self.config.noise_method == "flow_noise":
                # Learnable noise via ExploreNoiseNet
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)

            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t, denoise_value_t

    def get_suffix_out(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """Compute Gemma Expert network output features (suffix processing).

        Args:
            state: robot proprioceptive state, shape [B, state_dim]
            prefix_pad_masks: VLM prefix padding mask
            past_key_values: cached prefix KV
            x_t: action state at current step
            timestep: current timestep, shape [B]

        Returns:
            suffix_out: Expert network action token output,
                        shape [B, action_horizon, hidden_dim]
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Default "eager"; see prefix forward comment for SDPA NaN bug rationale.
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = self._attn_impl

        # Shallow-clone KV cache to prevent DynamicCache mutation:
        # transformers ≥4.53 GemmaAttention unconditionally calls
        # past_key_value.update() even when use_cache=False, which would
        # grow the shared cache on every denoising step.
        # We clone only the tensor lists (not the tensors themselves) so
        # .update() appends to the clone, leaving the original intact.
        if past_key_values is not None:
            from transformers.cache_utils import DynamicCache
            _kv_snapshot = DynamicCache()
            _kv_snapshot.key_cache = list(past_key_values.key_cache)
            _kv_snapshot.value_cache = list(past_key_values.value_cache)
        else:
            _kv_snapshot = None

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=_kv_snapshot,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon:]

        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)

        return suffix_out

    def get_logprob_norm(self, sample, mu, sigma):
        """Compute log-probability density of sample under Gaussian N(mu, sigma^2).

        Args:
            sample: actual sampled value, arbitrary shape
            mu: distribution mean, same shape as sample
            sigma: distribution std; zero tensor in deterministic mode

        Returns:
            log_prob: log-probability density, same shape as sample
        """
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)

            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)

        return log_prob

    def get_log_prob_value(
        self, images, img_masks, lang_tokens, lang_masks, state,
        chains, denoise_inds, compute_values=False,
        prefix_kv_cache=None,
    ):
        """Given action chains, recompute logprobs, values, and entropy
        (core computation for PPO updates).

        Args:
            images: image feature list
            img_masks: image validity mask list
            lang_tokens: language token tensor
            lang_masks: language token mask tensor
            state: robot state, shape [B, state_dim]
            chains: complete denoising trajectory,
                    shape [B, num_steps+1, action_horizon, action_dim]
            denoise_inds: selected denoising step indices, shape [B, num_steps]
            compute_values: whether to compute values
            prefix_kv_cache: cached (prefix_output, past_key_values, prefix_pad_masks)
                from a previous epoch. When provided and backbone is frozen, skips
                the expensive VLM prefix forward pass entirely.

        Returns:
            (chains_log_probs, chains_values, chains_entropy,
             chains_denoise_values, prefix_kv_cache_out) tuple.
            prefix_kv_cache_out is non-None when backbone is frozen (for caller
            to cache and reuse across PPO epochs).
        """
        bsize = state.shape[0]
        prefix_cache_out = None

        if prefix_kv_cache is not None and self.config.train_expert_only:
            # Reuse cached prefix — skip the entire VLM forward pass
            prefix_output, past_key_values, prefix_pad_masks = prefix_kv_cache
        else:
            # Compute VLM prefix output
            prefix_ctx = (
                torch.no_grad() if self.config.train_expert_only else contextlib.nullcontext()
            )
            with prefix_ctx:
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
                # Default "eager"; see sample_actions prefix comment for SDPA NaN bug rationale.
                self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = self._attn_impl

                [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

            # When backbone is frozen, expose cache for caller to reuse
            if self.config.train_expert_only:
                prefix_cache_out = (prefix_output, past_key_values, prefix_pad_masks)

        chains_log_probs = []
        chains_values = []
        chains_entropy = []
        self.reset_dapc_collector()

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        # Selective gradient checkpointing for DAPC's multi-step loop.
        # When joint_logprob=True (num_steps>1), all forward passes retain
        # their computation graphs simultaneously until backward, causing
        # ~num_steps × activation_size VRAM.  Wrapping sample_mean_var_val
        # in checkpoint discards intermediate activations and recomputes
        # them during backward, reducing peak from N× to ~1× activations.
        # PPO (num_steps=1) and inference are unaffected.
        _use_ckpt = self.training and num_steps > 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]

            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]

            _args = (chains_pre, denoise_ind, state,
                     prefix_pad_masks, past_key_values,
                     "train", self.config.num_steps, compute_values)
            if _use_ckpt:
                x_t_mean, x_t_std, value_t, denoise_value_t = (
                    torch.utils.checkpoint.checkpoint(
                        self.sample_mean_var_val, *_args,
                        use_reentrant=False,
                    )
                )
            else:
                x_t_mean, x_t_std, value_t, denoise_value_t = (
                    self.sample_mean_var_val(*_args)
                )

            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)

            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            self.collect_dapc_step(denoise_value_t)

            if not self.use_vlm_value:
                chains_values.append(value_t)

        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        if getattr(self, '_dapc_collected_values', []):
            chains_denoise_values = torch.stack(self._dapc_collected_values, dim=1)
        else:
            chains_denoise_values = torch.zeros((chains_log_probs.shape[0], 0),
                                                device=chains_log_probs.device)

        return chains_log_probs, chains_values, chains_entropy, chains_denoise_values, prefix_cache_out

    def get_value_from_vlm(self, prefix_output):
        """Extract value estimate from VLM (PaliGemma) prefix output.

        Args:
            prefix_output: PaliGemma VLM output tensor,
                           shape [B, prefix_len, hidden_dim]

        Returns:
            values_vlm: state value estimates, shape [B]
        """
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True]
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] + [False] * (all_token_length - 1)

        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)

        prefix_out_value = prefix_out_value.to(dtype=torch.float32)

        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        """Compute Gaussian differential entropy.

        H(N(mu, sigma^2)) = 0.5 * log(2 * pi * e * sigma^2)

        Args:
            sigma: Gaussian std tensor; sigma=0 positions set to 0

        Returns:
            entropy: Gaussian differential entropy, same shape as sigma
        """
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))

        return entropy

    def freeze_vlm(self):
        """Freeze PaliGemma VLM parameters (train only the Expert network).

        Called in load_pi0_model() when train_expert_only=True.
        """
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

    def post_set_train_mode(self, train: bool) -> None:
        """Keep the frozen PaliGemma VLM in inference mode after model.train().

        nn.Module.train() recursively flips every submodule to train mode,
        which re-enables dropout in the VLM. With train_expert_only=True, the
        VLM's parameters don't receive gradients, so dropout would only inject
        noise into the prefix outputs without any learning benefit. This hook
        — invoked by models.capabilities.set_train_mode — switches the VLM
        back whenever the outer model enters train mode, while leaving the
        trainable Gemma Expert untouched.
        """
        if train and self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.train(False)

    def get_forward_module_for_compile(self):
        """Return the submodule suitable for torch.compile and its forward
        method.

        Returns:
            (module, bound_method) tuple
        """
        module = self.paligemma_with_expert
        return module, module.forward


def _resolve_norm_stats_dir(
    asset_id: str,
    *,
    model_path: str,
    rl_checkpoint: "str | None",
):
    """Return the etils epath.Path to the norm_stats directory.

    Search order (first directory that exists wins):

    1. ``<experiment_dir>/norm_stats/<asset_id>/``
       Inferred from *rl_checkpoint* by walking up to the experiment root:
       ``<exp>/checkpoints/global_step_N/ckpt_file`` → ``<exp>/norm_stats/``
       Works for both file paths and directory (``global_step_N/``) paths.

    2. ``<model_path>/<asset_id>/``
       Classic location inside the SFT weight directory (existing behaviour).

    Raises:
        RuntimeError: when neither location exists.
    """
    import etils.epath as epath
    from pathlib import Path

    candidates = []

    # --- Candidate 1: norm_stats copied next to RL checkpoints ---
    if rl_checkpoint:
        ckpt = Path(str(rl_checkpoint))
        # Walk up: file → step_dir → checkpoints/ → experiment_root
        #          dir  → checkpoints/ → experiment_root
        step_dir = ckpt if ckpt.is_dir() else ckpt.parent
        checkpoints_dir = step_dir.parent
        experiment_dir = checkpoints_dir.parent
        candidate = experiment_dir / "norm_stats" / asset_id
        candidates.append(candidate)

    # --- Candidate 2: classic SFT model_path location ---
    if model_path:
        candidates.append(Path(str(model_path)) / asset_id)

    for candidate in candidates:
        ep = epath.Path(str(candidate))
        try:
            if ep.exists():
                return ep
        except Exception:
            pass  # etils may raise on invalid paths; treat as non-existent

    # Build a helpful error message
    checked = "\n  ".join(str(c) for c in candidates) if candidates else "(none — model_path and checkpoint are both unset)"
    raise RuntimeError(
        f"[Pi0] Could not find normalization statistics for asset '{asset_id}'.\n"
        f"Searched:\n  {checked}\n\n"
        f"To fix, either:\n"
        f"  • Ensure the SFT model directory (model_path) is accessible and contains '{asset_id}/', or\n"
        f"  • Run RL training once so norm_stats are copied to the experiment output directory."
    )


def load_pi0_model(cfg) -> OpenPi0ForRLActionPrediction:
    """Load and initialize the complete Pi0 RL model from configuration.

    This function is the sole initialization entry point for Pi0 models,
    responsible for:
      1. Parsing config to obtain openpi model configuration (OpenPi0Config)
      2. Loading safetensors-format pretrained weights
      3. Loading normalization statistics (norm_stats)
      4. Building the complete transform pipeline and injecting into model

    Args:
        cfg: Hydra config object supporting dict-like access (.get() method)

    Returns:
        Fully initialized OpenPi0ForRLActionPrediction instance
    """
    import safetensors
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.shared import normalize as _normalize

    # Import local openpi config module (avoids importing
    # openpi.training.checkpoints which would pull in lerobot)
    from models.pi0.config import get_openpi_config

    config_name = cfg.get("openpi", {}).get("config_name", "pi0_libero")
    model_path = cfg.get("model_path", "")

    actor_train_config = get_openpi_config(config_name, model_path=model_path)
    actor_model_config = actor_train_config.model

    # Upgrade base Pi0Config to OpenPi0Config (adding RL-related fields)
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)

    # Override OpenPi0Config fields with YAML openpi sub-node config
    openpi_overrides = cfg.get("openpi", {})
    if openpi_overrides:
        for key, val in openpi_overrides.items():
            if hasattr(actor_model_config, key):
                actor_model_config.__dict__[key] = val

    # Sync fields from top-level model config to openpi model config
    if cfg.get("action_chunk") is not None:
        actor_model_config.__dict__["action_chunk"] = cfg["action_chunk"]
    if cfg.get("action_dim") is not None:
        actor_model_config.__dict__["action_env_dim"] = cfg["action_dim"]
    if cfg.get("add_value_head") is not None:
        actor_model_config.__dict__["add_value_head"] = cfg["add_value_head"]
    if cfg.get("train_expert_only") is not None:
        actor_model_config.__dict__["train_expert_only"] = cfg["train_expert_only"]
    if cfg.get("cache_frozen_backbone") is not None:
        actor_model_config.__dict__["cache_frozen_backbone"] = cfg["cache_frozen_backbone"]
    if cfg.get("cache_offload") is not None:
        actor_model_config.__dict__["cache_offload"] = cfg["cache_offload"]
    if cfg.get("camera_map"):
        actor_model_config.__dict__["camera_map"] = dict(cfg["camera_map"])
    for key in (
        "use_vla_cache",
        "vla_cache_stage",
        "vla_cache_sim_threshold",
        "vla_cache_log_interval",
    ):
        if cfg.get(key) is not None:
            actor_model_config.__dict__[key] = cfg[key]

    # ---- Initialize model architecture ----
    model = OpenPi0ForRLActionPrediction(actor_model_config)

    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Detect PaliGemma inner-model structure once; used both for the key-remap
    # rules below and for weight tying after weight loading.
    _pali_lm = model.paligemma_with_expert.paligemma.language_model
    _lm_has_inner = hasattr(_pali_lm, "model")

    # ---- Load pretrained weights ----
    # When an RL checkpoint is provided the caller (model_loader._load_checkpoint)
    # will overlay the complete model state from that checkpoint, so loading SFT
    # weights here would be immediately overwritten — skip to save time and remove
    # the dependency on the SFT weight directory during eval.
    rl_checkpoint = cfg.get("checkpoint", None)
    if rl_checkpoint:
        logger.info(
            f"[Pi0] Skipping SFT weight loading: RL checkpoint supplied by caller "
            f"({rl_checkpoint})"
        )
    else:
        checkpoint_dir = download.maybe_download(str(model_path))

        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

        # Key name remapping for PaliGemma weight compatibility
        _PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma."
        if not _lm_has_inner:
            _KEY_REMAP_RULES = []
        else:
            _KEY_REMAP_RULES = [
                ("model.language_model.", "language_model.model."),
                ("lm_head.", "language_model.lm_head."),
                ("model.multi_modal_projector.", "multi_modal_projector."),
                ("model.vision_tower.", "vision_tower."),
            ]

        def _remap_ckpt_key(key: str) -> str:
            """Map old PaliGemma checkpoint key names to current model expected key names."""
            if not key.startswith(_PALIGEMMA_PREFIX):
                return key
            suffix = key[len(_PALIGEMMA_PREFIX):]
            for old, new in _KEY_REMAP_RULES:
                if suffix.startswith(old):
                    return _PALIGEMMA_PREFIX + new + suffix[len(old):]
            return key

        model_state = model.state_dict()
        for weight_path in weight_paths:
            ckpt_state = safetensors.torch.load_file(weight_path)
            remapped = {}
            n_remapped = 0
            for k, v in ckpt_state.items():
                new_k = _remap_ckpt_key(k)
                if new_k != k:
                    n_remapped += 1
                if new_k in model_state:
                    remapped[new_k] = v
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            logger.info(
                f"Weight loading: {weight_path} -- "
                f"loaded {len(remapped)} parameters (remapped {n_remapped} keys), "
                f"missing {len(missing)}, unexpected {len(unexpected)}"
            )

    # ---- PaliGemma weight tying ----
    pali = model.paligemma_with_expert.paligemma
    inner_lm = model.paligemma_with_expert._vlm_model
    if _lm_has_inner:
        lm_head_w = pali.language_model.lm_head.weight
    elif hasattr(pali, "lm_head"):
        lm_head_w = pali.lm_head.weight
    else:
        lm_head_w = None

    if lm_head_w is not None:
        embed_w = inner_lm.embed_tokens.weight
        if lm_head_w.data_ptr() != embed_w.data_ptr():
            inner_lm.embed_tokens.weight = lm_head_w
            logger.info("Tied embed_tokens.weight <- lm_head.weight (PaliGemma weight tying)")

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # ---- Load normalization statistics ----
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )

    norm_stats_dir = _resolve_norm_stats_dir(
        data_config.asset_id,
        model_path=model_path,
        rl_checkpoint=cfg.get("checkpoint", None),
    )
    norm_stats = _normalize.load(norm_stats_dir)
    logger.info(f"Loaded normalization statistics from {norm_stats_dir}")

    # ---- Build complete transform pipeline and inject into model ----
    # Uses generic Pi0Inputs / Pi0Outputs (env-agnostic) instead of the
    # legacy LiberoInputs / LiberoOutputs.  action_dim is read from config
    # (set via YAML ``model.action_dim``), not hard-coded.
    #
    # data_config.data_transforms may contain DeltaActions / AbsoluteActions
    # (model variant config, e.g. pi0 uses delta, pi0.5 uses absolute).
    # These are model-intrinsic and must be preserved in the pipeline.
    from models.pi0.pi0_transforms import Pi0Inputs, Pi0Outputs

    model.setup_wrappers(
        transforms=[
            transforms.InjectDefaultPrompt(None),
            Pi0Inputs(model_type=actor_model_config.model_type),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            Pi0Outputs(action_dim=actor_model_config.action_env_dim),
        ],
    )

    logger.info(f"Pi0 model loaded from {model_path}")
    return model
