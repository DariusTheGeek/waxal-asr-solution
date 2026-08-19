"""Contracts the family code and the pipeline both rely on."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_repository_root_contract_is_satisfied():
    """train/families/*/runtime_assets.py requires these three files."""
    for relpath in ("README.md", "models/MODELS.json", "data/provenance/SOURCES.json"):
        assert (ROOT / relpath).is_file(), f"missing root marker: {relpath}"


def test_model_manifest_is_complete():
    spec = json.loads((ROOT / "models/MODELS.json").read_text())
    assert len(spec["repos"]) == 10
    weights = [f for r in spec["repos"] for f in r["files"] if f["bytes"] > 1_000_000_000]
    assert len(weights) == 12, "expected 12 weight files across 10 repositories"
    digests = [f["sha256"] for f in weights]
    assert len(set(digests)) == len(digests), "two released weights share a digest"
    assert all(len(f["sha256"]) == 64 for f in weights)


def test_every_model_config_has_a_recipe():
    for config_path in sorted(ROOT.glob("configs/*/*.yaml")):
        lane = config_path.parent.name
        if lane not in {"lin", "sna", "joint"}:
            continue
        config = yaml.safe_load(config_path.read_text())
        stem = config["model"]
        candidates = [ROOT / "train/recipes" / lane / f"{stem}.yaml"]
        assert any(c.is_file() for c in candidates), \
            f"{config_path} has no recipe under train/recipes/{lane}/"


def test_lane_members_match_available_configs():
    inference = yaml.safe_load((ROOT / "configs/inference.yaml").read_text())
    for lane, spec in inference["lanes"].items():
        for member in spec["members"]:
            assert (ROOT / "configs" / lane / f"{member}.yaml").is_file(), \
                f"lane {lane} names member {member} with no config"


def test_lane_member_order_is_pinned():
    """Member order is load-bearing: the first member is the ROVER pivot and
    order breaks selection ties, so reordering changes the submission."""
    inference = yaml.safe_load((ROOT / "configs/inference.yaml").read_text())
    assert inference["lanes"]["lin"]["members"] == ["ctc3b", "llm3b", "llm1b", "ctc1b"]
    assert inference["lanes"]["sna"]["members"] == ["llm3b", "llm1b", "ctc7b", "mms1b"]


def test_rover_lanes_have_at_least_three_members():
    """Conservative word ROVER degenerates to a no-op below three sources."""
    inference = yaml.safe_load((ROOT / "configs/inference.yaml").read_text())
    for lane, spec in inference["lanes"].items():
        if spec["method"] == "rover":
            assert len(spec["members"]) >= 3, f"lane {lane} cannot ROVER"


def test_llm_configs_carry_a_language_conditioning_code():
    """The LLM decoder takes a language token; the CTC models ignore one.

    Omitting it does not error -- it silently decodes unconditioned, which
    changed roughly a quarter of the Shona hypotheses against the reference
    surface. The code is produced by this pipeline's own stage-0 routing
    decision.
    """
    expected = {"lin": "lin_Latn", "sna": "sna_Latn"}
    for lane in ("lin", "sna"):
        for model in ("llm1b", "llm3b"):
            config = yaml.safe_load((ROOT / "configs" / lane / f"{model}.yaml").read_text())
            code = config["decode"].get("language_code")
            assert code == expected[lane], \
                f"configs/{lane}/{model}.yaml must set language_code: {expected[lane]}"


def test_ctc_configs_do_not_set_a_language_code():
    """CTC models ignore the token; setting one would imply it has an effect."""
    for path in [ROOT / "configs/lin/ctc3b.yaml", ROOT / "configs/lin/ctc1b.yaml",
                 ROOT / "configs/sna/ctc7b.yaml", ROOT / "configs/joint/joint.yaml"]:
        config = yaml.safe_load(path.read_text())
        assert "language_code" not in config["decode"], f"{path} sets an ignored language_code"


def test_every_decode_uses_batch_size_one():
    """Batching makes output depend on GPU count, so the pipeline does not batch.

    Clips in a batch are padded to the longest member, which changes the encoder
    output and therefore the decoded text. The reference decodes were produced on
    four ranks with uneven shards, so the same code on two GPUs would not
    reproduce them. One clip at a time removes padding and the dependence.
    """
    for path in sorted(ROOT.glob("configs/*/*.yaml")):
        config = yaml.safe_load(path.read_text())
        if "decode" not in config:
            continue
        size = config["decode"].get("batch_size")
        assert size == 1, f"{path.relative_to(ROOT)} decodes with batch_size {size}"
