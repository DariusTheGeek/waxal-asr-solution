#!/usr/bin/env python3
"""Fresh-process strict reload and LLM-forward parity gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


EXPECTED_PARAMETERS = 4_380_578_432


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    export_dir = args.export_dir.resolve()
    export_record = json.loads((export_dir / "EXPORT.json").read_text(encoding="utf-8"))
    if (
        not isinstance(export_record, dict)
        or export_record.get("schema_version") != 2
        or export_record.get("status") != "PASS"
        or int(export_record.get("step", -1))
        != int(export_record.get("source_checkpoint_step", -2))
        or int(export_record.get("world_size", -1)) != 8
    ):
        raise RuntimeError("export record identity contract drift")
    required_environment = (
        "WAXAL3_REPO_ROOT",
        "WAXAL3_TRAINER_OUTPUT_DIR",
        "WAXAL3_REMOTE_STORE",
        "WAXAL3_EXPERIMENT_ID",
        "WAXAL3_PACKET_DIGEST",
        "WAXAL3_PROFILE",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"strict reload identity environment is incomplete: {missing}")
    from checkpoint_contract import export_source_identity

    observed_identity = export_source_identity(
        Path(os.environ["WAXAL3_TRAINER_OUTPUT_DIR"]),
        remote_store=Path(os.environ["WAXAL3_REMOTE_STORE"]),
        experiment_id=os.environ["WAXAL3_EXPERIMENT_ID"],
        packet_digest=os.environ["WAXAL3_PACKET_DIGEST"],
        step=int(export_record["source_checkpoint_step"]),
        world_size=int(export_record["source_checkpoint_world_size"]),
        profile=os.environ["WAXAL3_PROFILE"],
    )
    if any(export_record.get(key) != value for key, value in observed_identity.items()):
        raise RuntimeError("export/source checkpoint identity changed before strict reload")
    from early_stopping import validate_export_validation_state
    from fsdp_export import canonical_json_sha256
    from runtime_config import runtime_geometry_from_environment

    geometry = runtime_geometry_from_environment(
        Path(os.environ["WAXAL3_REPO_ROOT"])
    )

    validation_evidence = validate_export_validation_state(
        trainer_output_dir=Path(os.environ["WAXAL3_TRAINER_OUTPUT_DIR"]),
        rank_map_path=geometry.manifest_dir / "dev.rank_map.world8.csv",
        checkpoint_step=int(export_record["source_checkpoint_step"]),
        world_size=int(export_record["source_checkpoint_world_size"]),
        profile=os.environ["WAXAL3_PROFILE"],
        updates_per_epoch=geometry.updates_per_epoch,
    )
    if (
        export_record.get("validation_evidence") != validation_evidence
        or export_record.get("validation_evidence_digest")
        != canonical_json_sha256(validation_evidence)
    ):
        raise RuntimeError("export validation evidence changed before strict reload")
    checkpoint = export_dir / str(export_record["model_file"])
    probe_path = export_dir / str(export_record["probe_file"])
    if (
        sha256_file(checkpoint) != export_record["model_sha256"]
        or sha256_file(probe_path) != export_record["probe_sha256"]
    ):
        raise RuntimeError("export hash drift before strict reload")
    output = export_dir / "VERIFY.json"
    if output.exists():
        raise FileExistsError(output)
    root = Path(os.environ["WAXAL3_REPO_ROOT"]).resolve()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "OmniASR LLM strict reload requires exactly one visible CUDA device"
        )
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    properties = torch.cuda.get_device_properties(device)
    experiment_id = os.environ["WAXAL3_EXPERIMENT_ID"].lower()
    with tempfile.TemporaryDirectory(
        prefix=f"{experiment_id}-export-card-"
    ) as temporary_raw:
        temporary = Path(temporary_raw)
        card = temporary / "export.yaml"
        card.write_text(
            f"name: waxal3_{experiment_id}_consolidated_verify\n"
            "model_family: wav2vec2_llama\n"
            "model_arch: 3b_v2\n"
            f"checkpoint: {checkpoint.as_posix()}\n"
            f"tokenizer_ref: waxal3_{experiment_id}_export_tokenizer\n"
            "\n---\n\n"
            f"name: waxal3_{experiment_id}_export_tokenizer\n"
            "tokenizer_family: char_tokenizer\n"
            f"tokenizer: {(root / 'models/omniasr-llm-3b-v2/omniASR_tokenizer_written_v2.model').as_posix()}\n",
            encoding="utf-8",
        )
        os.environ["FAIRSEQ2_ASSET_DIR"] = str(temporary)
        import omnilingual_asr  # noqa: F401
        from fairseq2.models.hub import load_model
        from fsdp_export import parity_forward, parity_input

        model = load_model(
            f"waxal3_{experiment_id}_consolidated_verify",
            device=device,
            dtype=dtype,
            mmap=True,
            progress=False,
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"strictly reloaded parameter count drift: {parameters} != {EXPECTED_PARAMETERS}"
            )
        model.eval()
        parity_batch = parity_input(device, language_code=geometry.language_code)
        loss, logits, output_layout = parity_forward(
            model, parity_batch, device=device
        )
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    from fsdp_export import PARITY_TARGET_TOKENS

    if (
        probe.get("language_code") != geometry.language_code
        or tuple(probe.get("target_tokens", [])) != PARITY_TARGET_TOKENS
        or int(probe.get("input_samples", -1)) != 3_200
    ):
        raise RuntimeError("export parity probe input contract drift")
    expected_logits = probe["logits"].float()
    expected_loss = probe["loss"].float()
    observed_logits = logits.detach().float().cpu()
    observed_loss = loss.detach().float().cpu()
    if observed_logits.shape != expected_logits.shape:
        raise RuntimeError(
            "strict unsharded reload parity shape mismatch: "
            f"{tuple(observed_logits.shape)} != {tuple(expected_logits.shape)}"
        )
    maximum_delta = float((observed_logits - expected_logits).abs().max())
    mean_delta = float((observed_logits - expected_logits).abs().mean())
    loss_delta = float((observed_loss - expected_loss).abs().max())
    output_lengths = [int(value) for value in output_layout.seq_lens]
    if (
        output_lengths != list(probe["output_seq_lens"])
        or not bool(torch.isfinite(observed_logits).all())
        or not bool(torch.isfinite(observed_loss).all())
        or maximum_delta > 0.01
        or loss_delta > 0.01
    ):
        raise RuntimeError(
            "strict unsharded reload parity failed: "
            f"shape={tuple(observed_logits.shape)}/{tuple(expected_logits.shape)} "
            f"lengths={output_lengths}/{probe['output_seq_lens']} "
            f"max_delta={maximum_delta} loss_delta={loss_delta}"
        )
    record = {
        "schema_version": 2,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": export_record["experiment_id"],
        "packet_digest": export_record["packet_digest"],
        "profile": export_record["profile"],
        "source_checkpoint_step": export_record["source_checkpoint_step"],
        "checkpoint_inventory_digest": export_record[
            "checkpoint_inventory_digest"
        ],
        "source_local_marker_sha256": export_record[
            "source_local_marker_sha256"
        ],
        "target_evidence_sha256": export_record["target_evidence_sha256"],
        "export_record_sha256": sha256_file(export_dir / "EXPORT.json"),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": properties.name,
        "cuda_compute_capability": [properties.major, properties.minor],
        "cuda_total_memory_bytes": properties.total_memory,
        "dtype": str(dtype),
        "strict_load": True,
        "model_sha256": export_record["model_sha256"],
        "model_parameters": parameters,
        "logits_shape": list(observed_logits.shape),
        "output_seq_lens": output_lengths,
        "maximum_absolute_logit_delta": maximum_delta,
        "mean_absolute_logit_delta": mean_delta,
        "absolute_loss_delta": loss_delta,
        "tolerance": 0.01,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
