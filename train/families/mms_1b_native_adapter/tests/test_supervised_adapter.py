from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from supervised.mms_adapter import (
    adapter_l2sp_penalty,
    build_head_overlap_mapping,
    export_native_specialist,
    initialize_native_adapter_specialist,
    mapping_sha256,
    two_group_optimizer_parameters,
)
from supervised.model import validate_loading_info


class FakeAdapterLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(1280, 16)
        self.linear_2 = torch.nn.Linear(16, 1280)
        self.norm = torch.nn.LayerNorm(1280)


class FakeEncoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_layer = FakeAdapterLayer()
        self.frozen_base_probe = torch.nn.Parameter(torch.zeros(1))


class FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeEncoderLayer() for _ in range(48)])


class FakeWav2Vec2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()


class FakeMms(torch.nn.Module):
    def __init__(self, target_rows: int) -> None:
        super().__init__()
        self.wav2vec2 = FakeWav2Vec2()
        self.lm_head = torch.nn.Linear(1280, target_rows)

    def init_adapter_layers(self) -> None:
        return None

    def freeze_base_model(self) -> None:
        for parameter in self.wav2vec2.parameters():
            parameter.requires_grad = False


def test_native_adapter_only_init_fixed_head_l2sp_and_export(tmp_path: Path) -> None:
    source_vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}
    target_vocab = {"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "c": 4}
    source_vocab_path = tmp_path / "source_vocab.json"
    target_vocab_path = tmp_path / "target_vocab.json"
    source_vocab_path.write_text(json.dumps({"lin": source_vocab}), encoding="utf-8")
    target_vocab_path.write_text(json.dumps(target_vocab), encoding="utf-8")
    model = FakeMms(len(target_vocab))
    adapters = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if "adapter_layer" in name
    }
    assert len(adapters) == 288
    native = {
        **adapters,
        "lm_head.weight": torch.randn(len(source_vocab), 1280),
        "lm_head.bias": torch.randn(len(source_vocab)),
    }
    adapter_only = tmp_path / "cpt.safetensors"
    native_path = tmp_path / "adapter.lin.safetensors"
    head_init = tmp_path / "head.safetensors"
    save_file(adapters, str(adapter_only))
    save_file(native, str(native_path))
    fixed_head = {
        "lm_head.weight": torch.randn(len(target_vocab), 1280),
        "lm_head.bias": torch.randn(len(target_vocab)),
    }
    save_file(fixed_head, str(head_init))
    mapping = build_head_overlap_mapping(source_vocab, target_vocab)
    adapter_parameters = sum(value.numel() for value in adapters.values())
    expected = {
        "expected_source_head_rows": len(source_vocab),
        "expected_target_head_rows": len(target_vocab),
        "expected_mapped_head_rows": len(mapping),
        "expected_fresh_head_rows": len(target_vocab) - len(mapping),
        "expected_mapping_sha256": mapping_sha256(mapping),
        "expected_adapter_tensors": 288,
        "expected_adapter_parameters": adapter_parameters,
        "expected_head_parameters": len(target_vocab) * 1281,
        "expected_trainable_parameters": adapter_parameters
        + len(target_vocab) * 1281,
    }
    report, reference = initialize_native_adapter_specialist(
        model,
        adapter_path=adapter_only,
        native_package_path=native_path,
        source_vocab_path=source_vocab_path,
        target_vocab_path=target_vocab_path,
        language="lin",
        head_init_path=head_init,
        expected=expected,
    )
    assert report["trainable_parameters"] == expected["expected_trainable_parameters"]
    assert torch.equal(model.lm_head.weight, fixed_head["lm_head.weight"])
    assert torch.equal(model.lm_head.bias, fixed_head["lm_head.bias"])
    assert adapter_l2sp_penalty(model, reference).item() == 0.0
    adapter_group, head_group = two_group_optimizer_parameters(model)
    assert len(adapter_group) == 288
    assert len(head_group) == 2
    first_name = sorted(reference)[0]
    with torch.no_grad():
        dict(model.named_parameters())[first_name].view(-1)[0].add_(2.0)
    assert torch.isclose(adapter_l2sp_penalty(model, reference), torch.tensor(4.0))
    output = tmp_path / "export.safetensors"
    exported = export_native_specialist(model, output, metadata={"language": "lin"})
    assert exported["tensor_count"] == 290


def test_loading_info_allows_only_resized_linear_head() -> None:
    validate_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [
                ("lm_head.weight", (154, 1280), (44, 1280)),
                ("lm_head.bias", (154,), (44,)),
            ],
            "error_msgs": [],
        }
    )
