---
license: apache-2.0
base_model: facebook/omniASR-CTC-1B
datasets:
- google/WaxalNLP
language:
- ln
- sn
library_name: fairseq2
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- omnilingual-asr
- lingala
- shona
- waxal
- zindi
model-index:
- name: waxal-joint-ctc-1b-lid
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: WaxalNLP held-out validation split (pooled Lingala + Shona)
      type: google/WaxalNLP
    metrics:
    - name: CER
      type: cer
      value: 0.0885
    - name: WER
      type: wer
      value: 0.328
---

# waxal-joint-ctc-1b-lid

Joint Lingala + Shona OmniASR CTC-1B, fine-tuned **without any language tag**, paired with
the text language-identification classifier used to route its output.

This is the entry point of the [WAXAL ASR solution](https://github.com/DariusTheGeek/waxal-asr-solution).
It exists so the pipeline can determine each clip's language **from the audio**: a tag-free
decode of every clip, then a character n-gram classification of the decoded text.

## What is in this repository

| File | Bytes | Role |
|---|---:|---|
| `model.pt` | 3,903,024,817 | OmniASR CTC-1B v2, joint LIN+SNA, step 3033 |
| `omniASR_tokenizer_written_v2.model` | 91,481 | shared char tokenizer (identical across all OmniASR v2 families) |
| `text_lid_train_only.joblib` | 1,910,530 | TF-IDF char 3–5 gram + logistic regression language classifier |
| `card.yaml` | — | fairseq2 asset card |
| `requirements.txt` | — | pinned inference runtime |

## How routing works

1. `model.pt` decodes every clip once, with **no language conditioning** — the model has one
   recognizer, one shared encoder, one shared CTC head, and no language tag, prompt, ID, or
   embedding input.
2. `text_lid_train_only.joblib` classifies the resulting transcript as Lingala or Shona.
3. Each clip is routed to its language-specific decoder stack.

The classifier is fitted **only on gold training transcripts** (32,328 rows), never on model
output and never on test data. At inference it is applied to hypothesis strings.

On the 892-clip evaluation set it assigns 447 clips to Lingala and 445 to Shona.

## Model contract

```
recognizers: 1          shared_encoder: true     shared_tokenizer: true
shared_ctc_head: true   language_tag: false      language_prompt: false
language_id: false      language_embedding: false
metadata_input: false   model_level_routing: false
```

## Training

| | |
|---|---|
| Parent | `facebook/omniASR-CTC-1B` (v2, untouched official init) |
| Seed | 42 |
| Parallelism | DDP, world size 4 |
| Batch | 2 per GPU × 4 accumulation × 4 GPUs = 32 |
| Updates | 4,044 total; released checkpoint at **step 3033** |
| Peak LR | 1.0e-5, tri-stage 0.1/0.4/0.5 |
| Precision | bfloat16, activation checkpointing every layer |

Character and word error on the held-out validation split:

| Split | CER | WER |
|---|---|---|
| pooled | 0.0885 | 0.3280 |
| Lingala | 0.1345 | 0.3579 |
| Shona | 0.0492 | 0.2894 |

Standalone it is weaker than the language-specific models it routes to;
its value is that one object serves both languages, so a single decode of the full test set
yields the routing decision.

## Usage

```python
from huggingface_hub import snapshot_download
import joblib, sklearn

d = snapshot_download("DariusTheGeek/waxal-joint-ctc-1b-lid")
assert sklearn.__version__ == "1.5.2"      # the classifier was fitted under this version
lid = joblib.load(f"{d}/text_lid_train_only.joblib")
lid.predict(["mbote na yo", "mhoro sei"])  # -> ['lin', 'sna']
```

For the ASR half, point fairseq2 at `card.yaml` after substituting
`@WAXAL_MODEL_DIR@`, or use the solution repository's CLI, which wraps the whole
contract (pinned environment, batch-size-1 decode, tokenizer wiring):

```bash
git clone https://github.com/DariusTheGeek/waxal-asr-solution
cd waxal-asr-solution && bash install.sh
python models/download_models.py --repo waxal-joint-ctc-1b-lid

.venvs/omni/bin/python inference/decode/omniasr.py \
    --config configs/joint/joint.yaml \
    --audio path/to/wav_dir --output hypotheses.csv
.venvs/fuse/bin/python inference/route/route.py \
    --hypotheses hypotheses.csv \
    --model artifacts/text_lid_train_only.joblib --output route.csv
```

`requirements.txt` in this repo pins the inference runtime.

## Licence

`apache-2.0`, inherited from the OmniASR parent.
