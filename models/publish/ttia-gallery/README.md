---
license: cc-by-sa-4.0
language:
  - ln
tags:
  - speech
  - speaker-embeddings
  - asr
  - lingala
pretty_name: Waxal TTIA enrolment gallery
---

# Waxal TTIA enrolment gallery

Data assets for the Test-Time Idiolect Adaptation (TTIA) Lingala lane of the
[waxal-asr-solution](https://github.com/DariusTheGeek/waxal-asr-solution)
pipeline (Google Waxal ASR Challenge on Zindi). Fetched into place by
`models/download_assets.py` in that repository; every file is verified there
against a recorded SHA-256.

| File | Size | Contents |
|---|---|---|
| `enrollment.npz` | 998 MB | MMS-1B hidden-state voice vectors for 21,566 enrolment clips (layers 4, 6, 8, 12, 16; the pipeline reads 4 and 8), plus their clip ids |
| `enrollment.parquet` | 661 KB | enrolment manifest: clip id, idiolect profile key, derived audio path, language |
| `train.rows.parquet` | 4.5 MB | per-profile Lingala training text the TTIA fusion scores candidates against |

## Provenance and licence

Derived from [google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP)
(Lingala labelled training audio and transcripts, and clips from the
unlabelled release), © the WaxalNLP authors, licensed CC-BY-SA-4.0 /
CC-BY-4.0. These derivatives are published under **CC-BY-SA-4.0** with
attribution to google/WaxalNLP, as ShareAlike requires. No evaluation/test
audio or transcripts are included.

The files are reproducible from google/WaxalNLP with
`inference/ttia/build_enrollment.py`, `embed.py` and `merge.py` in the
solution repository; this upload exists so a reviewer can run the pipeline
without rebuilding them (a ~21,500-clip GPU embedding pass).

| File | SHA-256 |
|---|---|
| `enrollment.npz` | `e112a828ef220fc38eb6ff6e24c49a3e2ec38834f2e806ade0fe3bbdf1fe7086` |
| `enrollment.parquet` | `4fb6005ce024683ea4d5a111615438ba204a37ac7a4a0a57f935046d81f75ce7` |
| `train.rows.parquet` | `b44df5ca8e370df28408bb71ad6628aaa6d316ccf3a4e6b97d11d17d663e53bc` |
