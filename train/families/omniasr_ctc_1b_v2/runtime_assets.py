"""Render relocatable fairseq2 asset cards from a verified WAXAL3 root."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


ROOT_ENV = "WAXAL3_REPO_ROOT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_root() -> Path:
    raw = os.environ.get(ROOT_ENV)
    if not raw:
        raise RuntimeError(f"{ROOT_ENV} must name the extracted WAXAL3 bundle root")
    root = Path(raw).expanduser().resolve()
    required = (root / "README.md", root / "models/MODELS.json", root / "data/provenance/SOURCES.json")
    if not root.is_dir() or not all(path.is_file() for path in required):
        raise RuntimeError(f"invalid WAXAL3 repository root: {root}")
    return root


def render_asset_cards(template: Path, output_dir: Path) -> Path:
    root = resolve_repo_root()
    template = template.resolve()
    output_dir = output_dir.resolve()
    if not template.is_file():
        raise FileNotFoundError(template)
    rendered = template.read_text(encoding="utf-8").replace(
        "@WAXAL3_REPO_ROOT@", root.as_posix()
    )
    if "@WAXAL3_REPO_ROOT@" in rendered:
        raise RuntimeError("asset-card root placeholder remained after rendering")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "waxal3.yaml"
    payload = rendered.encode("utf-8")

    # torchrun starts every rank in the same runtime directory. Publish a fully
    # written card with an atomic no-replace link so no rank can observe another
    # rank's partially written file. Losing publishers accept only identical
    # bytes; a foreign file, directory, or symlink still fails closed.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".waxal3.yaml.", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
        raise RuntimeError(f"rendered asset-card collision: {path}")
    os.environ["FAIRSEQ2_ASSET_DIR"] = str(output_dir)
    return path


def verify_anchor(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"byte-count drift: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 drift: {path}: {observed}")
