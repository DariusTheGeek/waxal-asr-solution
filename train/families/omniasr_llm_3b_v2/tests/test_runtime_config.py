from __future__ import annotations

from pathlib import Path

import pytest

from runtime_config import (
    runtime_geometry_from_environment,
    runtime_geometry_from_experiment,
)


def _minimal_manifests(path: Path) -> None:
    path.mkdir(parents=True)
    for name in (
        "train.rows.parquet",
        "train.tsv",
        "train.wrd",
        "dev.rows.parquet",
        "dev.tsv",
        "dev.wrd",
        "dev.rank_map.world8.csv",
    ):
        (path / name).write_bytes(b"fixture")


def test_explicit_lingala_runtime_geometry(monkeypatch, tmp_path: Path) -> None:
    manifests = tmp_path / "data/portable/lin/manifests"
    _minimal_manifests(manifests)
    monkeypatch.setenv("WAXAL3_LANGUAGE", "lin")
    monkeypatch.setenv("WAXAL3_LANGUAGE_CODE", "lin_Latn")
    monkeypatch.setenv(
        "WAXAL3_MANIFEST_DIR", "data/portable/lin/manifests"
    )
    monkeypatch.setenv("WAXAL3_EXPECTED_TRAIN_ROWS", "16035")
    monkeypatch.setenv("WAXAL3_UPDATES_PER_EPOCH", "501")
    geometry = runtime_geometry_from_environment(tmp_path)
    assert geometry.language == "lin"
    assert geometry.language_code == "lin_Latn"
    assert geometry.manifest_dir == manifests.resolve()
    assert geometry.expected_train_rows == 16_035
    assert geometry.updates_per_epoch == 501
    assert geometry.expected_model_language_id is None


def test_explicit_shona_runtime_geometry(monkeypatch, tmp_path: Path) -> None:
    manifests = tmp_path / "data/portable/sna/manifests"
    _minimal_manifests(manifests)
    monkeypatch.setenv("WAXAL3_LANGUAGE", "sna")
    monkeypatch.setenv("WAXAL3_LANGUAGE_CODE", "sna_Latn")
    monkeypatch.setenv(
        "WAXAL3_MANIFEST_DIR", "data/portable/sna/manifests"
    )
    monkeypatch.setenv("WAXAL3_EXPECTED_TRAIN_ROWS", "16293")
    monkeypatch.setenv("WAXAL3_UPDATES_PER_EPOCH", "509")
    monkeypatch.setenv("WAXAL3_EXPECTED_MODEL_LANGUAGE_ID", "1343")
    geometry = runtime_geometry_from_environment(tmp_path)
    assert geometry.language == "sna"
    assert geometry.language_code == "sna_Latn"
    assert geometry.manifest_dir == manifests.resolve()
    assert geometry.expected_train_rows == 16_293
    assert geometry.updates_per_epoch == 509
    assert geometry.expected_model_language_id == 1_343


def test_runtime_manifest_cannot_escape_repo(monkeypatch, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-manifests"
    _minimal_manifests(outside)
    monkeypatch.setenv("WAXAL3_MANIFEST_DIR", str(outside))
    with pytest.raises(RuntimeError, match="escapes the WAXAL3 root"):
        runtime_geometry_from_environment(tmp_path)


def test_materialized_packet_resolves_frozen_specification(tmp_path: Path) -> None:
    manifests = tmp_path / "data/portable/lin/manifests"
    _minimal_manifests(manifests)
    packet = tmp_path / "experiment/packet"
    packet.mkdir(parents=True)
    (packet / "PACKET.json").write_text("{}\n", encoding="utf-8")
    (packet / "resolved_experiment.yaml").write_text(
        "language: lin\n"
        "runtime_contract:\n"
        "  language: lin\n"
        "  language_code: lin_Latn\n"
        "  manifest_dir: data/portable/lin/manifests\n"
        "  expected_train_rows: 16035\n"
        "  updates_per_epoch: 501\n",
        encoding="utf-8",
    )
    geometry = runtime_geometry_from_experiment(packet, tmp_path)
    assert geometry.language == "lin"
    assert geometry.language_code == "lin_Latn"
    assert geometry.manifest_dir == manifests.resolve()
    assert geometry.expected_train_rows == 16_035
    assert geometry.updates_per_epoch == 501
    assert geometry.expected_model_language_id is None


def test_shona_contract_resolves_expected_model_language_id(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "data/portable/sna/manifests"
    _minimal_manifests(manifests)
    packet = tmp_path / "experiment/packet"
    packet.mkdir(parents=True)
    (packet / "PACKET.json").write_text("{}\n", encoding="utf-8")
    (packet / "resolved_experiment.yaml").write_text(
        "language: sna\n"
        "runtime_contract:\n"
        "  language: sna\n"
        "  language_code: sna_Latn\n"
        "  expected_model_language_id: 1343\n"
        "  manifest_dir: data/portable/sna/manifests\n"
        "  expected_train_rows: 16293\n"
        "  updates_per_epoch: 509\n",
        encoding="utf-8",
    )
    geometry = runtime_geometry_from_experiment(packet, tmp_path)
    assert geometry.language == "sna"
    assert geometry.language_code == "sna_Latn"
    assert geometry.expected_train_rows == 16_293
    assert geometry.updates_per_epoch == 509
    assert geometry.expected_model_language_id == 1_343


def test_resolved_specification_without_packet_marker_is_rejected(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "experiment/packet"
    packet.mkdir(parents=True)
    (packet / "resolved_experiment.yaml").write_text(
        "language: lin\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unsafe or missing"):
        runtime_geometry_from_experiment(packet, tmp_path)
