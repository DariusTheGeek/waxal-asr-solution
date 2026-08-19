---
license: cc-by-nc-4.0
base_model: facebook/mms-1b-all
datasets:
- google/WaxalNLP
language:
- ln
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- mms
- lingala
- waxal
- zindi
---

# waxal-lin-mms-1b

MMS-1B native Lingala adapter, fine-tuned for **Lingala** as part of the
[WAXAL ASR solution](https://github.com/DariusTheGeek/waxal-asr-solution).

Released artifact: **top-3 uniform average of steps 1004 / 1506 / 1757**. Not decoded in the shipped pipeline — this model embeds voices for TTIA.

## Files

| File | Bytes |
|---|---:|
| `model.safetensors` | 3,858,957,656 |
| `config.json` | 2,072 |
| `preprocessor_config.json` | 254 |
| `vocab.json` | 505 |
| `requirements.txt` | — |

Total weights: 3,858,960,487 bytes (3.86 GB).

## Role in the pipeline

The TTIA voice embedder: the enrolment gallery and every test clip pass through its encoder, and mean+std-pooled hidden states (layers 4 and 8) drive the idiolect-profile match. It decodes no text in the shipped pipeline.

This model is **one component of an ensemble solution** and is not intended to be used
alone. The routing, decoding, fusion and post-processing pipeline are in the
[solution repository](https://github.com/DariusTheGeek/waxal-asr-solution).

## How to use

```python
# pip install -r requirements.txt   (pinned; see Files)
import librosa, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

repo = "DariusTheGeek/waxal-lin-mms-1b"
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
| Language | Lingala |
| Fine-tuning data | Waxal Lingala/Shona supervised split ([`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP)) |
| Seed | 42 |

> **Non-commercial.** This licence is inherited from `facebook/mms-1b-all` and is binding on anyone who downloads these weights. The OmniASR models in this solution are `apache-2.0`; only the two MMS models carry the NC restriction.

## Usage

See [`https://github.com/DariusTheGeek/waxal-asr-solution`](https://github.com/DariusTheGeek/waxal-asr-solution) for the environment locks, decode configuration and the exact
command that reproduces the submission end to end from audio.

## Licence

`cc-by-nc-4.0`, inherited from the parent model.
