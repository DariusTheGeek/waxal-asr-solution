#!/usr/bin/env python3
"""Build an atomic, verified average from three explicit full CTC states.

This entry point is for post-training checkpoint selection.  It deliberately
does not infer sources from a training run: a frozen config pins the three
export/verification records, model bytes, hashes, validation scores, and the
already-audited averaging implementation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import ModuleType
from typing import Any

import torch
from torch import Tensor


ALGORITHM = "uniform-parameter-mean-float64-v1"


class ExplicitAverageError(RuntimeError):
    """Raised when a source or publication contract is not exact."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def lexical_repo_path(value: str, root: Path) -> Path:
    if not value.startswith("repo://"):
        raise ExplicitAverageError(f"repo URI required: {value}")
    candidate = Path(os.path.abspath(root / value.removeprefix("repo://")))
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExplicitAverageError(f"repo URI escapes repository: {value}") from error
    return candidate


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExplicitAverageError(f"JSON object required: {path}")
    return value


def validate_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("schema_version") != 1:
        raise ExplicitAverageError("unsupported config schema")
    if config.get("algorithm") != ALGORITHM:
        raise ExplicitAverageError("averaging algorithm drift")
    if config.get("source_experiment_id") not in {"X0019", "X0044"}:
        raise ExplicitAverageError(
            "source experiment must be a 3B-family control (X0019 or X0044)"
        )
    if not isinstance(config.get("source_packet_digest"), str) or len(
        str(config["source_packet_digest"])
    ) != 64:
        raise ExplicitAverageError("invalid source packet digest")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ExplicitAverageError("exactly three explicit sources are required")
    if any(not isinstance(item, dict) for item in sources):
        raise ExplicitAverageError("source records must be objects")
    steps = [int(item.get("step", -1)) for item in sources]
    if len(set(steps)) != 3 or any(step <= 0 for step in steps):
        raise ExplicitAverageError("source steps must be three unique positive values")
    scores = [float(item.get("target_weighted_q", "nan")) for item in sources]
    if not all(score == score for score in scores) or scores != sorted(
        scores, reverse=True
    ):
        raise ExplicitAverageError("sources must be ordered by descending validation Q")
    for item in sources:
        for key in (
            "run_id",
            "model",
            "model_sha256",
            "export_record",
            "export_record_sha256",
            "verify_record",
            "verify_record_sha256",
        ):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ExplicitAverageError(f"source field is absent: {key}")
        if int(item.get("model_bytes", -1)) <= 0:
            raise ExplicitAverageError("source model byte count must be positive")
    return sources


def _verify_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExplicitAverageError(f"{label} must be a regular file: {path}")


def verify_source(
    item: Mapping[str, Any],
    *,
    root: Path,
    experiment_id: str,
    packet_digest: str,
) -> dict[str, Any]:
    model_path = lexical_repo_path(str(item["model"]), root)
    export_path = lexical_repo_path(str(item["export_record"]), root)
    verify_path = lexical_repo_path(str(item["verify_record"]), root)
    for path, label in (
        (model_path, "model"),
        (export_path, "export record"),
        (verify_path, "verification record"),
    ):
        _verify_regular_file(path, label=label)

    observed_hashes = {
        "model_sha256": sha256_file(model_path),
        "export_record_sha256": sha256_file(export_path),
        "verify_record_sha256": sha256_file(verify_path),
    }
    for key, observed in observed_hashes.items():
        if observed != str(item[key]):
            raise ExplicitAverageError(f"source {item['step']} {key} drift")
    if model_path.stat().st_size != int(item["model_bytes"]):
        raise ExplicitAverageError(f"source {item['step']} model byte-count drift")

    export = read_json_object(export_path)
    verify = read_json_object(verify_path)
    required_identity = {
        "experiment_id": experiment_id,
        "packet_digest": packet_digest,
        "source_checkpoint_step": int(item["step"]),
        "model_sha256": str(item["model_sha256"]),
    }
    for name, record in (("export", export), ("verify", verify)):
        for key, expected in required_identity.items():
            if record.get(key) != expected:
                raise ExplicitAverageError(
                    f"source {item['step']} {name} identity drift for {key}"
                )
    if verify.get("status") != "PASS" or verify.get("strict_load") is not True:
        raise ExplicitAverageError(f"source {item['step']} strict reload did not pass")
    if float(verify.get("maximum_absolute_logit_delta", float("inf"))) != 0.0:
        raise ExplicitAverageError(f"source {item['step']} reload logits changed")
    return {
        **dict(item),
        "model_path": model_path,
        "observed_hashes": observed_hashes,
    }


def load_core(path: Path, expected_sha256: str) -> ModuleType:
    _verify_regular_file(path, label="averaging core")
    if sha256_file(path) != expected_sha256:
        raise ExplicitAverageError("averaging core implementation hash drift")
    spec = importlib.util.spec_from_file_location("waxal3_frozen_checkpoint_average", path)
    if spec is None or spec.loader is None:
        raise ExplicitAverageError(f"cannot load averaging core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "load_checkpoint",
        "extract_model_state",
        "average_states",
        "tensor_schema",
        "tensor_content_sha256",
    ):
        if not callable(getattr(module, name, None)):
            raise ExplicitAverageError(f"averaging core lacks callable {name}")
    return module


def _safe_cleanup_temporary(path: Path, parent: Path) -> None:
    if (
        path.parent == parent
        and path.name.startswith(".parameter_average.")
        and path.is_dir()
        and not path.is_symlink()
    ):
        shutil.rmtree(path)


def build_average(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ExplicitAverageError(f"repository root must be a real directory: {root}")
    config_path = config_path.resolve(strict=True)
    config_path.relative_to(root)
    _verify_regular_file(config_path, label="frozen average config")
    config = read_json_object(config_path)
    sources = validate_config(config)
    experiment_id = str(config["source_experiment_id"])
    packet_digest = str(config["source_packet_digest"])
    verified = [
        verify_source(
            item,
            root=root,
            experiment_id=experiment_id,
            packet_digest=packet_digest,
        )
        for item in sources
    ]

    core_record = config.get("averaging_core")
    if not isinstance(core_record, dict):
        raise ExplicitAverageError("averaging core record is absent")
    core_path = lexical_repo_path(str(core_record.get("path", "")), root)
    core = load_core(core_path, str(core_record.get("sha256", "")))

    checkpoints = [core.load_checkpoint(item["model_path"]) for item in verified]
    states: Sequence[Mapping[str, Tensor]] = [
        core.extract_model_state(checkpoint) for checkpoint in checkpoints
    ]
    averaged, contract = core.average_states(states)
    tensor_content_sha256 = core.tensor_content_sha256(averaged)

    output = lexical_repo_path(str(config.get("output", "")), root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ExplicitAverageError(f"unsafe output parent: {output.parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=".parameter_average.", dir=output.parent)
    )
    try:
        checkpoint_path = temporary / "model.pt"
        torch.save({"model": averaged, "fs2": True}, checkpoint_path)
        reloaded = core.extract_model_state(core.load_checkpoint(checkpoint_path))
        if core.tensor_schema(reloaded) != contract["schema"]:
            raise ExplicitAverageError("averaged tensor schema changed on reload")
        for name in averaged:
            if not torch.equal(averaged[name], reloaded[name]):
                raise ExplicitAverageError(f"averaged tensor changed on reload: {name}")
        if core.tensor_content_sha256(reloaded) != tensor_content_sha256:
            raise ExplicitAverageError("averaged tensor-content hash changed on reload")

        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "kind": "explicit_ctc_checkpoint_parameter_average",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": ALGORITHM,
            "experiment_id": str(config.get("experiment_id")),
            "source_experiment_id": experiment_id,
            "source_packet_digest": packet_digest,
            "source_count": 3,
            "uniform_weight": 1 / 3,
            "source_steps_score_ranked": [int(item["step"]) for item in verified],
            "source_target_weighted_q": [
                float(item["target_weighted_q"]) for item in verified
            ],
            "sources": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"model_path", "observed_hashes"}
                }
                for item in verified
            ],
            "frozen_config": config_path.relative_to(root).as_posix(),
            "frozen_config_sha256": sha256_file(config_path),
            "averaging_core": {
                "path": core_path.relative_to(root).as_posix(),
                "sha256": sha256_file(core_path),
            },
            "accumulator_dtype": "torch.float64",
            "output_dtype_policy": "cast_each_average_to_source_dtype",
            "non_floating_policy": "require_exact_match",
            "tensor_schema_sha256": contract["schema_sha256"],
            "tensor_count": contract["tensor_count"],
            "parameter_count": contract["parameter_count"],
            "floating_tensor_count": contract["floating_tensor_count"],
            "non_floating_tensor_count": contract["non_floating_tensor_count"],
            "output_checkpoint": "model.pt",
            "output_checkpoint_bytes": checkpoint_path.stat().st_size,
            "output_checkpoint_sha256": sha256_file(checkpoint_path),
            "output_tensor_content_sha256": tensor_content_sha256,
            "strict_reload_tensor_equal": True,
            "finite_sources_and_output": True,
            "builder_sha256": sha256_file(Path(__file__)),
        }
        (temporary / "AVERAGE.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, output)
        observed = read_json_object(output / "AVERAGE.json")
        if observed != manifest or sha256_file(output / "model.pt") != manifest[
            "output_checkpoint_sha256"
        ]:
            raise ExplicitAverageError("published average verification failed")
        return manifest
    except BaseException:
        _safe_cleanup_temporary(temporary, output.parent)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_average(args.config, args.repository_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
