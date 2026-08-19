#!/usr/bin/env python3
"""Decode one MMS-1B model over the clips assigned to a language lane.

Runs under .venvs/hf (transformers). MMS is a plain CTC model, so decoding is
greedy argmax over the frame posteriors -- there is no beam search and no
language model.

The Lingala model emits lowercase; ``decode.sentence_case`` capitalises the
first letter so its surface matches the casing convention of the other lane
members. The shipped pipeline decodes only the Shona MMS surface — the Lingala
weights embed voices for TTIA.

Usage
-----
python inference/decode/mms.py --config configs/sna/mms1b.yaml \
    --audio data/test_audio --route outputs/route/route.csv --lane sna \
    --output outputs/decodes/sna_mms1b.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.text import normalize_text  # noqa: E402
from inference.decode.omniasr import clip_ids_for_lane, resolve_audio  # noqa: E402

TARGET_SAMPLE_RATE = 16_000


def require_hf_environment() -> None:
    expected = ROOT / ".venvs/hf/bin/python"
    if expected.exists() and Path(sys.executable).resolve() != expected.resolve():
        raise SystemExit(
            f"decode/mms.py must run under the hf environment.\n"
            f"  expected: {expected}\n  running:  {sys.executable}\n"
            f"  fix:      bash run_inference.sh")


def sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def main() -> int:
    require_hf_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--route", type=Path)
    parser.add_argument("--lane")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    # See inference/decode/omniasr.py: cuDNN's TF32 is on by default and
    # perturbs convolutional models. MMS is a CTC model, so it is affected.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    config = yaml.safe_load(args.config.read_text())
    weights_dir = args.weights or (ROOT / "weights" / config["repo"].split("/")[-1])
    if not weights_dir.is_dir():
        raise SystemExit(f"weights not found at {weights_dir}; "
                         f"run: python models/download_models.py")

    audio = resolve_audio(args.audio, clip_ids_for_lane(args.route, args.lane))

    processor = AutoProcessor.from_pretrained(weights_dir)
    model = Wav2Vec2ForCTC.from_pretrained(weights_dir).to(args.device).eval()

    # Batch size comes from the config: batching pads shorter clips, which
    # makes the output depend on how work was grouped.
    batch_size = int(config["decode"].get("batch_size", args.batch_size))
    apply_case = bool(config["decode"].get("sentence_case", False))
    rows: list[dict[str, str]] = []
    with torch.inference_mode():
        for start in range(0, len(audio), batch_size):
            chunk = audio[start:start + batch_size]
            waveforms = [librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)[0]
                         for _, path in chunk]
            inputs = processor(waveforms, sampling_rate=TARGET_SAMPLE_RATE,
                               return_tensors="pt", padding=True)
            logits = model(inputs.input_values.to(args.device),
                           attention_mask=getattr(inputs, "attention_mask",
                                                  torch.ones_like(inputs.input_values,
                                                                  dtype=torch.long)
                                                  ).to(args.device)).logits
            predicted = torch.argmax(logits, dim=-1)
            for (identifier, _), text in zip(chunk, processor.batch_decode(predicted)):
                target = normalize_text(text)
                rows.append({"ID": identifier,
                             "Target": sentence_case(target) if apply_case else target})
            print(f"  {min(start + batch_size, len(audio))}/{len(audio)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"decoded {len(rows)} clips with {config['model']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
