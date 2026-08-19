---
license: apache-2.0
base_model: facebook/omniASR-CTC-7B-v2
datasets:
- google/WaxalNLP
language:
- sn
library_name: fairseq2
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- omnilingual-asr
- shona
- waxal
- zindi
---

# waxal-sna-omniasr-ctc-7b

OmniASR CTC-7B v2, fine-tuned for **Shona** as part of the
[WAXAL ASR solution](https://github.com/DariusTheGeek/waxal-asr-solution).

Released artifact: **three single checkpoints: steps 3563 / 4581 / 5090**. Decoded with **greedy, then conservative word ROVER across the three checkpoints**.

## Files

| File | Bytes |
|---|---:|
| `step_3563/model.pt` | 26,023,706,842 |
| `step_4581/model.pt` | 26,023,706,842 |
| `step_5090/model.pt` | 26,023,706,842 |
| `omniASR_tokenizer_written_v2.model` | 91,481 |
| `card.yaml` | — |
| `requirements.txt` | — |

Total weights: 78,071,120,526 bytes (78.07 GB).

## Role in the pipeline

Its three checkpoints collapse to one hypothesis by conservative word ROVER, which then enters the 4-family word-medoid MBR.

This model is **one component of an ensemble solution** and is not intended to be used
alone. The routing, decoding, fusion and post-processing pipeline are in the
[solution repository](https://github.com/DariusTheGeek/waxal-asr-solution).

## How to use

This is a fairseq2 / `omnilingual-asr` checkpoint, loaded through the asset
card shipped alongside it (`card.yaml`). The solution repository wraps the whole
contract -- pinned environment, batch-size-1 decode, tokenizer wiring -- in one
CLI:

```bash
git clone https://github.com/DariusTheGeek/waxal-asr-solution
cd waxal-asr-solution && bash install.sh        # pinned environments, ~15 min
python models/download_models.py --repo waxal-sna-omniasr-ctc-7b

.venvs/omni/bin/python inference/decode/omniasr.py \
    --config configs/sna/ctc7b.yaml \
    --audio path/to/wav_dir --output transcripts.csv
```

`--weights` accepts any directory holding this repo's files, e.g. the path
returned by `huggingface_hub.snapshot_download("DariusTheGeek/waxal-sna-omniasr-ctc-7b")`.
`requirements.txt` in this repo pins the runtime alone; the environment lock
the release was verified under is `env/requirements-omni.txt` in the solution
repository.

## Provenance

| | |
|---|---|
| Parent | `facebook/omniASR-CTC-7B-v2` |
| Language | Shona |
| Fine-tuning data | Waxal Lingala/Shona supervised split ([`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP)) |
| Seed | 42 |

## Usage

See [`https://github.com/DariusTheGeek/waxal-asr-solution`](https://github.com/DariusTheGeek/waxal-asr-solution) for the environment locks, decode configuration and the exact
command that reproduces the submission end to end from audio.

## Licence

`apache-2.0`, inherited from the parent model.
