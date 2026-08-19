from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from build_explicit_parameter_average import (
    ALGORITHM,
    ExplicitAverageError,
    build_average,
    lexical_repo_path,
    sha256_file,
    validate_config,
)


def config() -> dict[str, object]:
    sources = []
    for rank, (step, score) in enumerate(
        ((3507, 0.7599), (2004, 0.7567), (3006, 0.7548)), start=1
    ):
        sources.append(
            {
                "rank": rank,
                "step": step,
                "run_id": f"RUN{rank:04d}",
                "target_weighted_q": score,
                "model": f"repo://run{rank}/model.pt",
                "model_bytes": 123,
                "model_sha256": str(rank) * 64,
                "export_record": f"repo://run{rank}/EXPORT.json",
                "export_record_sha256": str(rank + 3) * 64,
                "verify_record": f"repo://run{rank}/VERIFY.json",
                "verify_record_sha256": str(rank + 6) * 64,
            }
        )
    return {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "source_experiment_id": "X0019",
        "source_packet_digest": "a" * 64,
        "sources": sources,
    }


def test_validate_accepts_score_ranked_nonchronological_top3() -> None:
    assert [item["step"] for item in validate_config(config())] == [3507, 2004, 3006]


def test_validate_rejects_chronological_reordering() -> None:
    value = config()
    value["sources"] = sorted(value["sources"], key=lambda item: item["step"])
    with pytest.raises(ExplicitAverageError, match="descending validation Q"):
        validate_config(value)


def test_lexical_repo_path_rejects_escape(tmp_path) -> None:
    with pytest.raises(ExplicitAverageError, match="escapes repository"):
        lexical_repo_path("repo://../../escape", tmp_path.resolve())


def test_build_average_verifies_explicit_sources_and_reload(tmp_path: Path) -> None:
    core_source = (
        Path(os.environ["WAXAL3_REPO_ROOT"]).resolve()
        / "experiments/omniasr_ctc_3b_v2"
        / "mono/lin/supervised_ft/control_max12_target_es"
        / "X0019_ctc3b_v2_cv002_fsdp2_control/packet/src/model_family"
        / "checkpoint_average.py"
    )
    core = tmp_path / "frozen/checkpoint_average.py"
    core.parent.mkdir(parents=True)
    shutil.copy2(core_source, core)

    packet_digest = "a" * 64
    source_records = []
    for rank, (step, score, value) in enumerate(
        (
            (3507, 0.7599, 1.0),
            (2004, 0.7567, 2.0),
            (3006, 0.7548, 6.0),
        ),
        start=1,
    ):
        source = tmp_path / f"runs/RUN{rank:04d}/exported_model"
        source.mkdir(parents=True)
        model = source / "model.pt"
        torch.save(
            {
                "model": {
                    "weight": torch.tensor([value, value + 1], dtype=torch.float32),
                    "count": torch.tensor([7], dtype=torch.int64),
                },
                "fs2": True,
            },
            model,
        )
        model_sha = sha256_file(model)
        identity = {
            "status": "PASS",
            "experiment_id": "X0019",
            "packet_digest": packet_digest,
            "source_checkpoint_step": step,
            "model_sha256": model_sha,
        }
        export = source / "EXPORT.json"
        export.write_text(json.dumps(identity), encoding="utf-8")
        verify = source / "VERIFY.json"
        verify.write_text(
            json.dumps(
                {
                    **identity,
                    "strict_load": True,
                    "maximum_absolute_logit_delta": 0.0,
                }
            ),
            encoding="utf-8",
        )
        source_records.append(
            {
                "rank": rank,
                "step": step,
                "run_id": f"RUN{rank:04d}",
                "target_weighted_q": score,
                "model": f"repo://{model.relative_to(tmp_path).as_posix()}",
                "model_bytes": model.stat().st_size,
                "model_sha256": model_sha,
                "export_record": f"repo://{export.relative_to(tmp_path).as_posix()}",
                "export_record_sha256": sha256_file(export),
                "verify_record": f"repo://{verify.relative_to(tmp_path).as_posix()}",
                "verify_record_sha256": sha256_file(verify),
            }
        )
    frozen = {
        "schema_version": 1,
        "experiment_id": "X0019_TOP3_PARAMETER_AVERAGE_V1",
        "algorithm": ALGORITHM,
        "source_experiment_id": "X0019",
        "source_packet_digest": packet_digest,
        "averaging_core": {
            "path": f"repo://{core.relative_to(tmp_path).as_posix()}",
            "sha256": sha256_file(core),
        },
        "sources": source_records,
        "output": "repo://derived/top3",
    }
    config_path = tmp_path / "packet/config.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(frozen), encoding="utf-8")

    manifest = build_average(config_path, tmp_path)

    assert manifest["status"] == "PASS"
    assert manifest["source_steps_score_ranked"] == [3507, 2004, 3006]
    state = torch.load(
        tmp_path / "derived/top3/model.pt", map_location="cpu", weights_only=True
    )["model"]
    assert torch.equal(state["weight"], torch.tensor([3.0, 4.0]))
    assert torch.equal(state["count"], torch.tensor([7]))
