"""stratified_v7 -> CTC batches. Everything model-shaped is read from arch.json, nothing guessed.

The `audio` column is raw 16 kHz WAV bytes (not a datasets Audio feature), so rows are decoded
with soundfile straight from memory. Streaming by default: the repo is 17.2 GB and a Kaggle
session's disk is not worth spending on it.
"""
from __future__ import annotations

import io
import json
import math
import unicodedata

import numpy as np
import torch


def norm_text(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


class Arch:
    def __init__(self, path: str = "arch.json"):
        d = json.load(open(path, encoding="utf-8"))
        self.model = d["model"]
        self.revision = d.get("revision")
        self.how = d["how"]
        self.feat_key = d["feat_key"]
        self.mask_key = d.get("mask_key")
        self.encoder_dim = d["encoder_dim"]
        self.downsample = d["feat_to_enc_downsample"]
        self.fps = d["encoder_frames_per_second"]
        self.sr = d.get("sample_rate", 16000)

    def enc_len(self, feat_len: int) -> int:
        return max(1, int(math.floor(feat_len / self.downsample)))


class Vocab:
    def __init__(self, path: str = "vocab.json"):
        d = json.load(open(path, encoding="utf-8"))
        self.stoi = d["vocab"]
        self.itos = {i: s for s, i in self.stoi.items()}
        self.blank = d["blank_id"]
        self.unk = d["unk_id"]
        self.delim = d["delim_id"]

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        out = []
        for ch in norm_text(text):
            out.append(self.delim if ch == " " else self.stoi.get(ch, self.unk))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(" " if i == self.delim else self.itos.get(i, "") for i in ids)

    def ctc_collapse(self, ids: list[int]) -> str:
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.blank:
                out.append(i)
            prev = i
        return self.decode(out).strip()


class Features:
    """One call shape, chosen by probe.py. Returns (feats[T,F], real_frames)."""

    def __init__(self, arch: Arch):
        from transformers import AutoProcessor
        self.arch = arch
        self.proc = AutoProcessor.from_pretrained(arch.model, revision=arch.revision)
        self.fe = getattr(self.proc, "feature_extractor", None)

    def _call(self, wavs: list[np.ndarray]):
        a, sr = self.arch, self.arch.sr
        if a.how == "processor(audio=)":
            return self.proc(audio=wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        if a.how == "processor(audios=)":
            return self.proc(audios=wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        if a.how == "processor(text+audio)":
            return self.proc(text=["<|audio|>"] * len(wavs), audio=wavs,
                             sampling_rate=sr, return_tensors="pt", padding=True)
        return self.fe(wavs, sampling_rate=sr, return_tensors="pt", padding=True)

    def batch(self, wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
        out = self._call(wavs)
        feats = out[self.arch.feat_key]
        mask = out.get(self.arch.mask_key) if self.arch.mask_key else None
        if mask is not None:
            lens = mask.sum(-1).long().tolist() if mask.dtype != torch.bool else mask.sum(-1).tolist()
        else:
            per = feats.shape[1] / max(len(w) for w in wavs)
            lens = [min(feats.shape[1], max(1, int(round(len(w) * per)))) for w in wavs]
        return feats, mask, [int(x) for x in lens]


def decode_wav(b: bytes, sr: int = 16000) -> np.ndarray | None:
    import soundfile as sf
    try:
        x, got = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - a corrupt row is skipped, never fatal
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if got != sr:
        n = int(round(len(x) * sr / got))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def rows(repo: str, split: str, *, streaming: bool = True, lang: str | None = None,
         shuffle_buffer: int = 2000, seed: int = 42):
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=streaming)
    if streaming and shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for r in ds:
        if lang and r.get("language") != lang:
            continue
        yield r


def batches(repo: str, split: str, arch: Arch, vocab: Vocab, *, batch_size: int = 8,
            max_seconds: float = 16.0, min_seconds: float = 0.4, bucket: int = 64,
            lang: str | None = None, streaming: bool = True, seed: int = 42,
            features: Features | None = None, epochs: int = 1):
    """Length-bucketed batches. `bucket` rows are buffered and sorted by duration before
    slicing into batches, which cuts padding waste sharply without a full-dataset sort."""
    feat = features or Features(arch)
    for _ in range(epochs):
        buf: list[tuple[np.ndarray, list[int]]] = []

        def flush(buf):
            buf.sort(key=lambda t: len(t[0]))
            for i in range(0, len(buf), batch_size):
                chunk = buf[i:i + batch_size]
                wavs = [w for w, _ in chunk]
                labs = [l for _, l in chunk]
                feats, mask, flens = feat.batch(wavs)
                elens = [min(arch.enc_len(fl), feats.shape[1]) for fl in flens]
                keep = [j for j, (e, l) in enumerate(zip(elens, labs)) if l and e >= len(l)]
                if not keep:
                    continue
                targets = torch.cat([torch.tensor(labs[j], dtype=torch.long) for j in keep])
                yield {
                    "features": feats[keep],
                    "mask": None if mask is None else mask[keep],
                    "enc_lens": torch.tensor([elens[j] for j in keep], dtype=torch.long),
                    "targets": targets,
                    "target_lens": torch.tensor([len(labs[j]) for j in keep], dtype=torch.long),
                    "texts": [vocab.decode(labs[j]) for j in keep],
                    "dropped": len(chunk) - len(keep),
                }

        for r in rows(repo, split, streaming=streaming, lang=lang, seed=seed):
            x = decode_wav(r["audio"], arch.sr)
            if x is None:
                continue
            secs = len(x) / arch.sr
            if not (min_seconds <= secs <= max_seconds):
                continue
            lab = vocab.encode(r["text"])
            if not lab:
                continue
            buf.append((x, lab))
            if len(buf) >= bucket:
                yield from flush(buf)
                buf = []
        if buf:
            yield from flush(buf)
