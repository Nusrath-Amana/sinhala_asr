"""Resolve every unknown about EmbeddingGemma 2's audio tower, then write arch.json.

RUN THIS FIRST. Nothing else in this directory hardcodes a guess about the processor's
argument name, the feature tensor's name, the encoder's output width or its frame rate --
they are all measured here and read from arch.json by data.py / train.py.

Cheap enough for CPU (one forward pass). On a 7.7 GB box prefer --dtype float32 and expect
~3 GB resident; on Kaggle just run it as is.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

MODEL = "google/embeddinggemma-2"
SR = 16000


def _try(label, fn):
    try:
        out = fn()
        if out is None:
            return None, f"{label}: not available"
        shapes = {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")}
        return (label, out, shapes), f"{label}: OK {shapes}"
    except Exception as e:  # noqa: BLE001 - we are probing on purpose
        return None, f"{label}: ERR {type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default="arch.json")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoModel, AutoProcessor

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    ac = cfg.audio_config
    keys = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "output_proj_dims", "conv_kernel_size", "subsampling_conv_channels",
            "attention_chunk_size", "attention_context_left", "attention_context_right")
    audio_cfg = {k: getattr(ac, k, None) for k in keys}
    print("== audio_config ==")
    for k, v in audio_cfg.items():
        print(f"  {k:30} {v}")

    proc = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    fe = getattr(proc, "feature_extractor", None)
    print(f"\n== processor == {type(proc).__name__}  feature_extractor={type(fe).__name__ if fe else None}")

    wav = (0.01 * np.random.randn(int(SR * a.seconds))).astype(np.float32)
    cands = [
        ("processor(audio=)", lambda: proc(audio=[wav], sampling_rate=SR, return_tensors="pt")),
        ("processor(audios=)", lambda: proc(audios=[wav], sampling_rate=SR, return_tensors="pt")),
        ("processor(text+audio)", lambda: proc(text=["<|audio|>"], audio=[wav], sampling_rate=SR, return_tensors="pt")),
        ("feature_extractor", lambda: fe(wav, sampling_rate=SR, return_tensors="pt") if fe else None),
    ]
    winner = None
    print("\n== how audio goes in ==")
    for label, fn in cands:
        got, msg = _try(label, fn)
        print("  " + msg)
        if got and winner is None:
            winner = got
    if winner is None:
        raise SystemExit("No way in. Paste the block above and we pick the feature path by hand.")

    how, feats, shapes = winner
    feat_key = max((k for k in shapes), key=lambda k: len(shapes[k]))
    tdim = shapes[feat_key][1]
    # The mask must index the FEATURE time axis. A multimodal processor also returns the TEXT
    # attention_mask, a different length entirely -- picking by name alone (as this first did)
    # makes the tower fail with a 399-vs-104 size mismatch.
    mask_key = next((k for k in shapes
                     if "mask" in k.lower() and len(shapes[k]) == 2 and shapes[k][1] == tdim), None)
    others = [k for k in shapes if "mask" in k.lower() and k != mask_key]
    print(f"\n  using {how}: features={feat_key}{shapes[feat_key]} mask={mask_key}"
          + (f"  (ignoring {others}: wrong axis)" if others else ""))

    dtype = getattr(torch, a.dtype)
    print(f"\n== loading audio_tower ({a.dtype}) ==")
    model = AutoModel.from_pretrained(a.model, revision=a.revision, dtype=dtype)
    tower = model.audio_tower
    n_tower = sum(p.numel() for p in tower.parameters())
    print(f"  audio_tower params: {n_tower/1e6:.1f}M   layers: {len(tower.layers)}")
    print(f"  full model params:  {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    x = feats[feat_key].to(dtype)
    m = feats[mask_key] if mask_key else None
    # transformers itself calls it positionally: audio_tower(input_features, input_features_mask)
    sigs = [("tower(x, mask)", lambda: tower(x, m)), ("tower(x)", lambda: tower(x))]
    if mask_key:
        sigs.insert(1, (f"tower({feat_key}=, {mask_key}=)",
                        lambda: tower(**{feat_key: x, mask_key: m})))
    out = ret = None
    print("\n== forward ==")
    with torch.no_grad():
        for label, fn in sigs:
            try:
                r = fn()
                out = r[0] if isinstance(r, tuple) else getattr(r, "last_hidden_state", r)
                ret = r
                print(f"  {label}: OK -> {tuple(out.shape)}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {label}: ERR {type(e).__name__}: {e}")
    if out is None:
        raise SystemExit("audio_tower would not run. Paste the block above.")

    # The tower returns its OWN downsampled frame mask (True = valid). Using that beats dividing
    # the feature length by the stride, which drifts a frame on padded batches.
    enc_mask = getattr(ret, "attention_mask", None)
    returns_mask = enc_mask is not None and getattr(enc_mask, "ndim", 0) == 2
    print(f"  tower returns a frame mask: {returns_mask}"
          + (f" {tuple(enc_mask.shape)} valid={int(enc_mask.sum())}" if returns_mask else ""))

    n_feat_frames = shapes[feat_key][1]
    t_out = out.shape[1]
    enc_dim = out.shape[-1]
    arch = {
        "model": a.model, "revision": a.revision, "audio_config": audio_cfg,
        "processor_class": type(proc).__name__, "how": how,
        "feat_key": feat_key, "mask_key": mask_key,
        "feat_shape_for_seconds": {str(a.seconds): list(shapes[feat_key])},
        "encoder_dim": int(enc_dim),
        "encoder_params_m": round(n_tower / 1e6, 1),
        "feat_frames": int(n_feat_frames),
        "encoder_frames": int(t_out),
        "feat_to_enc_downsample": round(n_feat_frames / t_out, 4),
        "returns_mask": bool(returns_mask),
        "encoder_frames_per_second": round(t_out / a.seconds, 3),
        "seconds_per_encoder_frame": round(a.seconds / t_out, 5),
        "sample_rate": SR,
    }
    print("\n== measured ==")
    print(f"  encoder_dim                {enc_dim}   <- CTC head input width")
    print(f"  encoder frames / second    {arch['encoder_frames_per_second']}")
    print(f"  feature -> encoder stride  {arch['feat_to_enc_downsample']}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)
    print(f"\nwrote {a.out}")
    fps = arch["encoder_frames_per_second"]
    if fps < 8:
        print(f"  WARNING: {fps} frames/s is coarse for CTC. Sinhala averages well over {fps} "
              "graphemes/s in fast speech, and CTC cannot emit more labels than it has frames. "
              "Check vocab.py's --report for the worst-case ratio before training.")


if __name__ == "__main__":
    main()
