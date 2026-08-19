#!/usr/bin/env python3
"""Generate a model card and, for OmniASR repos, a fairseq2 asset card per HF repo."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "repos.json").read_text())
NS = SPEC["namespace"]
SOLUTION = "https://github.com/DariusTheGeek/waxal-asr-solution"

LANE_NAME = {"lin": "Lingala", "sna": "Shona", "routing": "language routing"}

ROLE = {
    "waxal-joint-ctc-1b-lid":
        "Decodes all 892 clips once with no language conditioning; its transcripts feed "
        "the text language classifier that routes every clip to a language lane.",
    "waxal-lin-omniasr-ctc-3b":
        "First member of the Lingala TTIA fusion and the pivot of its ROVER candidate.",
    "waxal-lin-omniasr-llm-3b":
        "Member of the Lingala TTIA fusion.",
    "waxal-lin-omniasr-llm-1b":
        "Member of the Lingala TTIA fusion.",
    "waxal-lin-omniasr-ctc-1b":
        "Member of the Lingala TTIA fusion; the conservative-ROVER candidate over all four "
        "members is scored alongside the members themselves.",
    "waxal-lin-mms-1b":
        "The TTIA voice embedder: the enrolment gallery and every test clip pass through "
        "its encoder, and mean+std-pooled hidden states (layers 4 and 8) drive the "
        "idiolect-profile match. It decodes no text in the shipped pipeline.",
    "waxal-sna-omniasr-llm-3b":
        "Member of the 4-family word-medoid MBR on the Shona lane.",
    "waxal-sna-omniasr-llm-1b":
        "Member of the 4-family word-medoid MBR on the Shona lane.",
    "waxal-sna-omniasr-ctc-7b":
        "Its three checkpoints collapse to one hypothesis by conservative word ROVER, which "
        "then enters the 4-family word-medoid MBR.",
    "waxal-sna-mms-1b":
        "Member of the 4-family word-medoid MBR on the Shona lane; the medoid is chosen by "
        "distance to peers, so a member can shape the selection geometry without being "
        "selected itself.",
}

CARD_YAML = """\
# fairseq2 asset card for {name}.
# Substitute @WAXAL_MODEL_DIR@ with the directory holding this file, e.g.
#   python -c "from huggingface_hub import snapshot_download; print(snapshot_download('{ns}/{name}'))"

name: waxal_omni_tokenizer_written_v2
tokenizer_family: char_tokenizer
tokenizer: @WAXAL_MODEL_DIR@/omniASR_tokenizer_written_v2.model

---

name: {asset}
model_family: {family}
model_arch: {arch}
checkpoint: @WAXAL_MODEL_DIR@/{ckpt}
tokenizer_ref: waxal_omni_tokenizer_written_v2
"""

# fairseq2 model family and architecture per run. The family strings are the
# ones fairseq2 registers: CTC models are `wav2vec2_asr`, the LLM-decoder models
# are `wav2vec2_llama`. Getting this wrong fails at load with an AssetCardError,
# not at card-generation time, so tests/test_asset_cards.py checks them.
ARCH = {"waxal-lin-omniasr-ctc-1b": ("wav2vec2_asr", "1b_v2"),
        "waxal-lin-omniasr-ctc-3b": ("wav2vec2_asr", "3b_v2"),
        "waxal-sna-omniasr-ctc-7b": ("wav2vec2_asr", "7b_v2"),
        "waxal-lin-omniasr-llm-1b": ("wav2vec2_llama", "1b_v2"),
        "waxal-lin-omniasr-llm-3b": ("wav2vec2_llama", "3b_v2"),
        "waxal-sna-omniasr-llm-1b": ("wav2vec2_llama", "1b_v2"),
        "waxal-sna-omniasr-llm-3b": ("wav2vec2_llama", "3b_v2")}


# Decode config each repo's usage snippet points at.
CONFIG = {"waxal-joint-ctc-1b-lid":   "configs/joint/joint.yaml",
          "waxal-lin-omniasr-ctc-3b": "configs/lin/ctc3b.yaml",
          "waxal-lin-omniasr-ctc-1b": "configs/lin/ctc1b.yaml",
          "waxal-lin-omniasr-llm-3b": "configs/lin/llm3b.yaml",
          "waxal-lin-omniasr-llm-1b": "configs/lin/llm1b.yaml",
          "waxal-sna-omniasr-llm-3b": "configs/sna/llm3b.yaml",
          "waxal-sna-omniasr-llm-1b": "configs/sna/llm1b.yaml",
          "waxal-sna-omniasr-ctc-7b": "configs/sna/ctc7b.yaml",
          "waxal-lin-mms-1b":         "configs/lin/mms1b.yaml",
          "waxal-sna-mms-1b":         "configs/sna/mms1b.yaml"}

# Inference-time requirements per family, subset of the env/ locks.
REQ_OMNI = """\
# Inference pins for this model (fairseq2 stack). Full lock: env/requirements-omni.txt
# in the solution repository -- torch here is the cu126 build.
torch==2.8.0
torchaudio==2.8.0
omnilingual-asr==0.2.0
fairseq2==0.6
fairseq2n==0.6
numpy==1.26.4
PyYAML==6.0.3
huggingface_hub==0.36.2
"""

REQ_MMS = """\
# Inference pins for this model (transformers stack). Full lock:
# env/requirements-hf.txt in the solution repository.
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1
torchaudio==2.5.1
transformers==4.46.3
tokenizers==0.20.3
safetensors==0.7.0
librosa==0.10.2
soundfile==0.12.1
numpy==1.26.4
huggingface_hub==0.36.2
"""

REQ_JOINT_EXTRA = """\
# The routing classifier bundled in this repo additionally needs:
scikit-learn==1.5.2
joblib==1.4.2
"""


def usage(r) -> str:
    if not is_omni(r):
        return f"""```python
# pip install -r requirements.txt   (pinned; see Files)
import librosa, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

repo = "{NS}/{r['name']}"
processor = AutoProcessor.from_pretrained(repo)
model = Wav2Vec2ForCTC.from_pretrained(repo).eval().cuda()

audio, _ = librosa.load("clip.wav", sr=16_000, mono=True)
inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
with torch.no_grad():
    logits = model(inputs.input_values.cuda()).logits
print(processor.batch_decode(logits.argmax(-1))[0])
```"""
    return f"""This is a fairseq2 / `omnilingual-asr` checkpoint, loaded through the asset
card shipped alongside it (`card.yaml`). The solution repository wraps the whole
contract -- pinned environment, batch-size-1 decode, tokenizer wiring -- in one
CLI:

```bash
git clone {SOLUTION}
cd waxal-asr-solution && bash install.sh        # pinned environments, ~15 min
python models/download_models.py --repo {r['name']}

.venvs/omni/bin/python inference/decode/omniasr.py \\
    --config {CONFIG[r['name']]} \\
    --audio path/to/wav_dir --output transcripts.csv
```

`--weights` accepts any directory holding this repo's files, e.g. the path
returned by `huggingface_hub.snapshot_download("{NS}/{r['name']}")`.
`requirements.txt` in this repo pins the runtime alone; the environment lock
the release was verified under is `env/requirements-omni.txt` in the solution
repository."""


def is_omni(r): return r["name"] in ARCH or r["name"] == "waxal-joint-ctc-1b-lid"


# Byte sizes come from the release manifest, which pins every published file.
MODELS = json.loads((ROOT.parent / "MODELS.json").read_text())
BYTES = {m["name"]: {f["path"]: f["bytes"] for f in m["files"]} for m in MODELS["repos"]}


def card(r) -> str:
    files = [(f["dst"], BYTES[r["name"]][f["dst"]]) for f in r["files"]]
    total = sum(b for _, b in files)
    lang_yaml = "\n".join(f"- {c}" for c in r["lang"])
    lib = "fairseq2" if is_omni(r) else "transformers"
    tags = ["automatic-speech-recognition",
            "omnilingual-asr" if is_omni(r) else "mms",
            *( ["lingala"] if "ln" in r["lang"] else [] ),
            *( ["shona"] if "sn" in r["lang"] else [] ),
            "waxal", "zindi"]
    tag_yaml = "\n".join(f"- {t}" for t in tags)
    rows = "\n".join(f"| `{d}` | {b:,} |" for d, b in files)
    extra = ""
    if is_omni(r):
        extra = ("| `omniASR_tokenizer_written_v2.model` | 91,481 |\n"
                 "| `card.yaml` | — |\n")
    extra += "| `requirements.txt` | — |\n"
    decoded = (f"Decoded with **{r['decode']}**." if not r["decode"].startswith("none")
               else "Not decoded in the shipped pipeline — this model embeds voices for TTIA.")
    nc = ""
    if r["license"] == "cc-by-nc-4.0":
        nc = ("\n> **Non-commercial.** This licence is inherited from `facebook/mms-1b-all` and is "
              "binding on anyone who downloads these weights. The OmniASR models in this solution "
              "are `apache-2.0`; only the two MMS models carry the NC restriction.\n")
    return f"""---
license: {r['license']}
base_model: {r['base']}
datasets:
- google/WaxalNLP
language:
{lang_yaml}
library_name: {lib}
pipeline_tag: automatic-speech-recognition
tags:
{tag_yaml}
---

# {r['name']}

{r['family']}, fine-tuned for **{LANE_NAME[r['lane']]}** as part of the
[WAXAL ASR solution]({SOLUTION}).

Released artifact: **{r['artifact']}**. {decoded}

## Files

| File | Bytes |
|---|---:|
{rows}
{extra}
Total weights: {total:,} bytes ({total/1e9:.2f} GB).

## Role in the pipeline

{ROLE[r['name']]}

This model is **one component of an ensemble solution** and is not intended to be used
alone. The routing, decoding, fusion and post-processing pipeline are in the
[solution repository]({SOLUTION}).

## How to use

{usage(r)}

## Provenance

| | |
|---|---|
| Parent | `{r['base']}` |
| Language | {LANE_NAME[r['lane']]} |
| Fine-tuning data | Waxal Lingala/Shona supervised split ([`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP)) |
| Seed | 42 |
{nc}
## Usage

See [`{SOLUTION}`]({SOLUTION}) for the environment locks, decode configuration and the exact
command that reproduces the submission end to end from audio.

## Licence

`{r['license']}`, inherited from the parent model.
"""


def main() -> None:
    for r in SPEC["repos"]:
        d = ROOT.parent / "cards" / r["name"]
        d.mkdir(exist_ok=True)
        if r["name"] != "waxal-joint-ctc-1b-lid":      # repo 1 has a hand-written card
            (d / "README.md").write_text(card(r))
        req = REQ_OMNI if is_omni(r) else REQ_MMS
        if r["name"] == "waxal-joint-ctc-1b-lid":
            req += REQ_JOINT_EXTRA
        (d / "requirements.txt").write_text(req)
        if r["name"] in ARCH:
            fam, arch = ARCH[r["name"]]
            ck = r["files"][0]["dst"] if len(r["files"]) == 1 else "step_3563/model.pt"
            (d / "card.yaml").write_text(CARD_YAML.format(
                name=r["name"], ns=NS, asset=r["name"].replace("-", "_"),
                family=fam, arch=arch, ckpt=ck))
        print(f"  {r['name']:<28} card{' + asset card' if r['name'] in ARCH else ''}")


if __name__ == "__main__":
    main()
