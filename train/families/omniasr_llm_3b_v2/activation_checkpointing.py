"""Packet-owned activation-checkpoint registration for OmniASR LLM models.

Omnilingual ASR 0.2.0 registers its Wav2Vec2-Llama FSDP splitter but omits
the matching fairseq2 layerwise activation-checkpoint applier.  Keep the
repair local to the frozen WAXAL3 packet and fail closed if the pinned
upstream extension changes.
"""

from __future__ import annotations

from torch.nn import Module

from fairseq2.composition.models import register_model_family
from fairseq2.models.utils.ac import apply_layerwise_ac
from fairseq2.runtime.dependency import DependencyContainer

from omnilingual_asr.models.wav2vec2_asr.config import (
    register_omnilingual_asr_wav2vec2_asr_configs,
)
from omnilingual_asr.models.wav2vec2_llama import (
    WAV2VEC2_LLAMA_FAMILY,
    Wav2Vec2LlamaConfig,
    Wav2Vec2LlamaModel,
    apply_fsdp_to_wav2vec2_llama,
    convert_wav2vec2_llama_state_dict,
    create_wav2vec2_llama_model,
    register_wav2vec2_llama_configs,
)
from omnilingual_asr.models.wav2vec2_ssl.config import (
    register_omnilingual_asr_wav2vec2_ssl_configs,
)


EXPECTED_OMNILINGUAL_ASR_VERSION = "0.2.0"
REPAIR_ID = "waxal3_wav2vec2_llama_layerwise_ac_v1"


def apply_ac_to_wav2vec2_llama(
    model: Wav2Vec2LlamaModel, every_nth_layer: int
) -> Module:
    """Checkpoint the 60 speech and 12 Llama layers before FSDP wrapping."""

    if not isinstance(model, Wav2Vec2LlamaModel):
        raise TypeError(
            "model must be an omnilingual_asr Wav2Vec2LlamaModel, "
            f"but is {type(model)}"
        )
    if every_nth_layer <= 0:
        raise ValueError("every_nth_layer must be positive")
    apply_layerwise_ac(model.encoder.layers, every_nth_layer)
    apply_layerwise_ac(model.llama_decoder.layers, every_nth_layer)
    return model


def _register_models_with_layerwise_ac(container: DependencyContainer) -> None:
    """Reproduce the pinned upstream registration with one missing callback."""

    register_omnilingual_asr_wav2vec2_ssl_configs(container)
    register_omnilingual_asr_wav2vec2_asr_configs(container)
    register_model_family(
        container,
        WAV2VEC2_LLAMA_FAMILY,
        kls=Wav2Vec2LlamaModel,
        config_kls=Wav2Vec2LlamaConfig,
        factory=create_wav2vec2_llama_model,
        fsdp_applier=apply_fsdp_to_wav2vec2_llama,
        layerwise_ac_applier=apply_ac_to_wav2vec2_llama,
        state_dict_converter=convert_wav2vec2_llama_state_dict,
    )
    register_wav2vec2_llama_configs(container)


def install_wav2vec2_llama_layerwise_ac() -> dict[str, object]:
    """Patch the not-yet-initialized pinned extension, idempotently."""

    import omnilingual_asr

    version = str(getattr(omnilingual_asr, "__version__", ""))
    if version != EXPECTED_OMNILINGUAL_ASR_VERSION:
        raise RuntimeError(
            "Omnilingual ASR version changed; re-audit the activation-checkpoint "
            f"repair before training: {version!r}"
        )
    current = getattr(omnilingual_asr, "_register_models", None)
    if current is _register_models_with_layerwise_ac:
        return {
            "status": "PASS",
            "repair_id": REPAIR_ID,
            "upstream_version": version,
            "mode": "already_installed",
        }
    if (
        current is None
        or getattr(current, "__module__", None) != "omnilingual_asr"
        or getattr(current, "__name__", None) != "_register_models"
    ):
        raise RuntimeError(
            "Omnilingual ASR model registration changed; refusing an unsafe patch"
        )
    omnilingual_asr._register_models = _register_models_with_layerwise_ac
    return {
        "status": "PASS",
        "repair_id": REPAIR_ID,
        "upstream_version": version,
        "mode": "installed_before_fairseq2_container_initialization",
    }
