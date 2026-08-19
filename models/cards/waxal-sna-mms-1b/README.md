---
license: cc-by-nc-4.0
base_model: facebook/mms-1b-all
datasets:
- google/WaxalNLP
language:
- sn
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- mms
- shona
- waxal
- zindi
---

# waxal-sna-mms-1b

MMS-1B native Shona adapter, continued pre-training then supervised fine-tune, fine-tuned for **Shona** as part of the
[WAXAL ASR solution](https://github.com/DariusTheGeek/waxal-asr-solution).

Released artifact: **checkpoint-2040**. Decoded with **greedy**.

## Files

| File | Bytes |
|---|---:|
| `model.safetensors` | 3,858,931,916 |
| `config.json` | 2,072 |
| `preprocessor_config.json` | 254 |
| `vocab.json` | 430 |
| `requirements.txt` | — |

Total weights: 3,858,934,672 bytes (3.86 GB).

## Role in the pipeline

Member of the 4-family word-medoid MBR on the Shona lane; the medoid is chosen by distance to peers, so a member can shape the selection geometry without being selected itself.

This model is **one component of an ensemble solution** and is not intended to be used
alone. The routing, decoding, fusion and post-processing pipeline are in the
[solution repository](https://github.com/DariusTheGeek/waxal-asr-solution).

## How to use

```python
# pip install -r requirements.txt   (pinned; see Files)
import librosa, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

repo = "DariusTheGeek/waxal-sna-mms-1b"
processor = AutoProcessor.from_pretrained(repo)
model = Wav2Vec2ForCTC.from_pretrained(repo).eval().cuda()

audio, _ = librosa.load("clip.wav", sr=16_000, mono=True)
inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
with torch.no_grad():
    logits = model(inputs.input_values.cuda()).logits
print(processor.batch_decode(logits.argmax(-1))[0])
```

## Provenance

| | |
|---|---|
| Parent | `facebook/mms-1b-all` |
| Language | Shona |
| Fine-tuning data | Waxal Lingala/Shona supervised split ([`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP)) |
| Seed | 42 |

> **Non-commercial.** This licence is inherited from `facebook/mms-1b-all` and is binding on anyone who downloads these weights. The OmniASR models in this solution are `apache-2.0`; only the two MMS models carry the NC restriction.

## Usage

See [`https://github.com/DariusTheGeek/waxal-asr-solution`](https://github.com/DariusTheGeek/waxal-asr-solution) for the environment locks, decode configuration and the exact
command that reproduces the submission end to end from audio.

## Licence

`cc-by-nc-4.0`, inherited from the parent model.
