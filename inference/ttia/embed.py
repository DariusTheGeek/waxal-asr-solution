#!/usr/bin/env python3
"""Embed audio clips into fixed-size voice vectors, at several encoder depths.

One forward pass of the MMS-1B model (`waxal-lin-mms-1b`, the same weights the
repository already ships) per clip; the hidden states of the requested encoder
layers are mean+std pooled over time, giving one vector per clip per layer.
Early-to-middle layers carry the character of a voice rather than its words,
which is what enrolment matching needs.

Runs under the ``hf`` environment. Shardable across GPUs: run one process per
``--shard`` and merge the outputs with ``merge.py``.

Usage
-----
python inference/ttia/embed.py --audio data/test_audio --output outputs/ttia/test.npz
python inference/ttia/embed.py --manifest outputs/ttia/enrollment.parquet \\
    --audio outputs/ttia/audio --shard 0 --num-shards 4 \\
    --output outputs/ttia/_shard0.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
TARGET_SR = 16000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True,
                    help="directory of clips, or the root the manifest's "
                         "relative paths resolve against")
    parser.add_argument("--manifest", type=Path,
                    help="parquet with id and derived_audio_relpath; omit to "
                         "embed every audio file under --audio")
    parser.add_argument("--model", type=Path,
                    default=ROOT / "weights/waxal-lin-mms-1b")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8, 12, 16])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.manifest:
        import pandas as pd
        frame = pd.read_parquet(args.manifest,
                                columns=["id", "derived_audio_relpath"])
        clips = [(str(r.id), args.audio / str(r.derived_audio_relpath))
                 for r in frame.itertuples()]
    else:
        clips = [(p.stem, p) for p in sorted(args.audio.rglob("*"))
                 if p.suffix.lower() in AUDIO_SUFFIXES]
    missing = [str(p) for _, p in clips if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, first: {missing[:2]}")

    # Interleaved sharding, so every shard sees a similar mix of clip lengths
    # and parallel processes finish at about the same time.
    if args.num_shards > 1:
        clips = clips[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(clips)} clips",
              flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    layers = [layer for layer in args.layers if layer <= n_layers]
    print(f"{n_layers} layers available; extracting {layers}", flush=True)

    ids, per_layer = [], {layer: [] for layer in layers}
    with torch.inference_mode():
        for index, (clip_id, path) in enumerate(clips, start=1):
            wave, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
            inputs = processor(wave, sampling_rate=TARGET_SR,
                               return_tensors="pt")
            out = model.wav2vec2(inputs.input_values.to(args.device),
                                 output_hidden_states=True)
            for layer in layers:
                h = out.hidden_states[layer][0]
                vec = torch.cat([h.mean(0), h.std(0)]).float().cpu().numpy()
                per_layer[layer].append(vec)
            ids.append(clip_id)
            if index % 200 == 0:
                print(f"  {index}/{len(clips)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, ids=np.array(ids),
        **{f"layer_{layer}": np.stack(per_layer[layer]) for layer in layers})
    args.output.with_suffix(".json").write_text(json.dumps(
        {"clips": len(ids), "layers": layers, "model": str(args.model),
         "pooling": "mean+std"}, indent=2) + "\n")
    print(f"wrote {len(ids)} clips x {len(layers)} layers -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
