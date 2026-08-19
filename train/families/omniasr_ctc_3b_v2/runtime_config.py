"""Frozen per-experiment runtime geometry for the OmniASR CTC-3B-v2 family."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


MANIFEST_DIR_ENV = "WAXAL3_MANIFEST_DIR"
TRAIN_ROWS_ENV = "WAXAL3_EXPECTED_TRAIN_ROWS"
UPDATES_PER_EPOCH_ENV = "WAXAL3_UPDATES_PER_EPOCH"
LANGUAGE_ENV = "WAXAL3_LANGUAGE"

LEGACY_LANGUAGE = "lin"
LEGACY_MANIFEST_DIR = (
    "data/derived/portable/omniasr1b_lin_cv002_v1/manifests"
)
LEGACY_TRAIN_ROWS = 16_035
LEGACY_UPDATES_PER_EPOCH = 501


@dataclass(frozen=True)
class RuntimeGeometry:
    language: str
    manifest_dir: Path
    expected_train_rows: int
    updates_per_epoch: int


def _positive_int(value: object, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return result


def _safe_repo_path(root: Path, value: object, *, name: str) -> Path:
    raw = Path(str(value))
    path = (raw if raw.is_absolute() else root / raw).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{name} escapes the WAXAL3 root: {path}") from error
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{name} is not a safe directory: {path}")
    for filename in (
        "train.rows.parquet",
        "train.tsv",
        "train.wrd",
        "dev.rows.parquet",
        "dev.tsv",
        "dev.wrd",
        "dev.rank_map.world8.csv",
    ):
        candidate = path / filename
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"{name} is incomplete: {candidate}")
    return path


def runtime_geometry_from_environment(root: Path) -> RuntimeGeometry:
    """Resolve the launcher's frozen runtime variables, retaining X0024 defaults."""

    root = root.resolve()
    language = os.environ.get(LANGUAGE_ENV, LEGACY_LANGUAGE).strip()
    if not language or "/" in language or "\\" in language:
        raise RuntimeError(f"invalid {LANGUAGE_ENV}: {language!r}")
    manifest_dir = _safe_repo_path(
        root,
        os.environ.get(MANIFEST_DIR_ENV, LEGACY_MANIFEST_DIR),
        name=MANIFEST_DIR_ENV,
    )
    return RuntimeGeometry(
        language=language,
        manifest_dir=manifest_dir,
        expected_train_rows=_positive_int(
            os.environ.get(TRAIN_ROWS_ENV, LEGACY_TRAIN_ROWS), name=TRAIN_ROWS_ENV
        ),
        updates_per_epoch=_positive_int(
            os.environ.get(UPDATES_PER_EPOCH_ENV, LEGACY_UPDATES_PER_EPOCH),
            name=UPDATES_PER_EPOCH_ENV,
        ),
    )


def runtime_geometry_from_experiment(
    experiment_dir: Path, root: Path
) -> RuntimeGeometry:
    """Read a live experiment or its materialized packet runtime contract."""

    experiment_dir = experiment_dir.resolve()
    live_specification = experiment_dir / "experiment.yaml"
    packet_specification = experiment_dir / "resolved_experiment.yaml"
    if live_specification.is_file() and not live_specification.is_symlink():
        specification_path = live_specification
    elif (
        (experiment_dir / "PACKET.json").is_file()
        and not (experiment_dir / "PACKET.json").is_symlink()
        and packet_specification.is_file()
        and not packet_specification.is_symlink()
    ):
        specification_path = packet_specification
    else:
        specification_path = live_specification
    if specification_path.is_symlink() or not specification_path.is_file():
        raise RuntimeError(f"unsafe or missing experiment specification: {specification_path}")
    specification = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
    if not isinstance(specification, dict):
        raise RuntimeError("experiment specification is not an object")
    contract = specification.get("runtime_contract")
    if contract is None:
        contract = {
            "language": specification.get("language", LEGACY_LANGUAGE),
            "manifest_dir": LEGACY_MANIFEST_DIR,
            "expected_train_rows": LEGACY_TRAIN_ROWS,
            "updates_per_epoch": LEGACY_UPDATES_PER_EPOCH,
        }
    if not isinstance(contract, dict):
        raise RuntimeError("runtime_contract is not an object")
    language = str(contract.get("language", "")).strip()
    if language != str(specification.get("language", "")).strip():
        raise RuntimeError("runtime language does not match experiment language")
    if not language or "/" in language or "\\" in language:
        raise RuntimeError(f"invalid runtime language: {language!r}")
    return RuntimeGeometry(
        language=language,
        manifest_dir=_safe_repo_path(
            root.resolve(), contract.get("manifest_dir"), name="runtime manifest_dir"
        ),
        expected_train_rows=_positive_int(
            contract.get("expected_train_rows"), name="runtime expected_train_rows"
        ),
        updates_per_epoch=_positive_int(
            contract.get("updates_per_epoch"), name="runtime updates_per_epoch"
        ),
    )
