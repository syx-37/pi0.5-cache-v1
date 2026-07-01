"""Vendored transformers_replace package.

This package contains modified versions of transformers' Gemma / PaliGemma /
SigLIP modeling files used by Pi0 (adds AdaRMS, custom 4D mask handling, etc).

Pre-V1b layout: these files were copied into the pip-installed transformers
via ``cp -r transformers_replace/* $SITE_PACKAGES/transformers/`` at install time.

Post-V1b layout (this file's existence makes the directory an importable
package): openpi imports patched classes from here directly, AND we
re-register the patched classes with ``transformers.AutoModel`` so that
``AutoModel.from_config(GemmaConfig)`` returns OUR patched ``GemmaModel``
instead of upstream's.

The registry override fires at import time. Importing openpi has a global
side effect on the in-process transformers AutoModel registry — analogous to
the cp install step's global filesystem side effect, but contained to the
process rather than the filesystem.
"""
from __future__ import annotations

from transformers import AutoModel
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from openpi.models_pytorch.transformers_replace.models.gemma.modeling_gemma import (
    GemmaModel,
)
from openpi.models_pytorch.transformers_replace.models.siglip.modeling_siglip import (
    SiglipVisionModel,
)


def _register_patched(config_class, model_class) -> None:
    """Register a patched modeling class for AutoModel.from_config(config_class).

    Uses ``exist_ok=True`` semantics when available — if the config is already
    registered (e.g. on re-import or because cp left the patched version in
    transformers already), we silently overwrite. AutoModel.register raises
    ValueError on duplicate without exist_ok, so we catch it for older
    transformers versions that lack the kwarg.
    """
    try:
        AutoModel.register(config_class, model_class, exist_ok=True)
    except TypeError:
        try:
            AutoModel.register(config_class, model_class)
        except ValueError:
            pass
    except ValueError:
        pass


_register_patched(GemmaConfig, GemmaModel)
_register_patched(SiglipVisionConfig, SiglipVisionModel)

__all__ = ["GemmaModel", "SiglipVisionModel"]
