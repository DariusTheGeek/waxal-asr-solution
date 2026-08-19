"""Guard the third-party entry points each environment depends on.

These imports have moved between omnilingual-asr releases; a rename here is
otherwise only discovered part-way through a multi-hour decode.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBES = {
    "omni": [
        "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline",
        "from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig",
    ],
    "hf": [
        "from transformers import Wav2Vec2ForCTC, AutoProcessor",
    ],
    "fuse": [
        "import sklearn, joblib; assert sklearn.__version__ == '1.5.2'",
        "import joblib; assert joblib.__version__ == '1.4.2'",
    ],
}


@pytest.mark.parametrize("profile,statement",
                         [(p, s) for p, xs in PROBES.items() for s in xs])
def test_entry_point_imports(profile: str, statement: str) -> None:
    python = ROOT / ".venvs" / profile / "bin" / "python"
    if not python.exists():
        pytest.skip(f"{profile} environment not built")
    result = subprocess.run([str(python), "-c", statement],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr.strip()[-400:]
