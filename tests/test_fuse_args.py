"""Argument contract of the fusion CLI: ttia requires its two inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.fuse.fuse import main  # noqa: E402


def test_ttia_without_keys_and_train_texts_refuses(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "fuse.py", "--method", "ttia", "--output", "out.csv", "a.csv", "b.csv"])
    with pytest.raises(SystemExit, match="--keys and --train-texts"):
        main()


def test_duplicate_surface_refuses(monkeypatch):
    """The same surface twice silently re-weights a vote; refuse it instead."""
    monkeypatch.setattr(sys, "argv", [
        "fuse.py", "--method", "medoid", "--output", "out.csv", "a.csv", "a.csv"])
    with pytest.raises(SystemExit, match="same surface"):
        main()
