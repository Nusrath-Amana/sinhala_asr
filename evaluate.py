"""Score a checkpoint: WER, CER, and latency decomposed into the three parts that move
independently.

Latency is reported per stage because the middle one is where this plan usually dies -- a wide
beam over a large 4-gram can cost more than the 300M encoder forward, and an overall number
hides that. Measure on the device you intend to serve from; a Kaggle T4 figure compared against
Gemini's network round-trip proves nothing.

    # the dataset's own test split, greedy
    python evaluate.py --ckpt ckpt/last.pt --split test --limit 2000

    # your own holdout: a folder of 16 kHz wavs, each with a matching .txt
    python evaluate.py --ckpt ckpt/last.pt --audio-dir holdout/

    # with the 4-gram (pip install pyctcdecode kenlm)
    python evaluate.py --ckpt ckpt/last.pt --split test --kenlm lm/si_4gram.bin --beam 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import time

import numpy as np
import torch

import data as D
from train import CTCHead, encode, patch_fp16_attn, resolve_ckpt


def load_ckpt(path, arch, n_vocab, device, dtype):
    from transformers import AutoConfig, AutoModel
    ck = torch.load(path, map_location="cpu")
    cfg = AutoConfig.from_pretrained(arch.model, revision=arch.revision)
    if dtype == torch.float16:      # -1e9 mask constant overflows fp16; see train.build()
        lim = float(torch.finfo(dtype).min / 2)
        for c in (cfg, getattr(cfg, "audio_config", None)):
            if c is not None and getattr(c, "attention_invalid_logits_value", 0) < lim:
                c.attention_invalid_logits_value = lim
    full = AutoModel.from_pretrained(arch.model, revision=arch.revision, config=cfg, dtype=dtype)
    tower = full.audio_tower
    del full
    patch_fp16_attn(tower, dtype)
    if ck.get("tower"):
        missing = tower.load_state_dict(ck["tower"], strict=False)
        print(f"  encoder: loaded finetuned weights ({missing})")
    else:
        print("  encoder: base weights (head-only run)")
    hidden = ck["args"].get("head_hidden", 0)
    head = CTCHead(arch.encoder_dim, ck.get("vocab_size", n_vocab), hidden)
    head.load_state_dict(ck["head"])
    return tower.to(device).eval(), head.to(device).float().eval(), ck


def make_decoder(kenlm, vocab, beam, alpha, beta, hotwords, hotword_weight):
    """pyctcdecode wants the label list in index order, with the blank as "".
    hotwords is where the estate's own vocabulary goes -- division names, field codes, "target".
    This is the lexical control that a prompted cloud model would not give you."""
    from pyctcdecode import build_ctcdecoder
    labels = []
    for i in range(len(vocab)):
        tok = vocab.itos.get(i, "")
        labels.append("" if i == vocab.blank else (" " if i == vocab.delim else tok))
    dec = build_ctcdecoder(labels, kenlm_model_path=kenlm, alpha=alpha, beta=beta)
    return lambda logits: dec.decode(logits, beam_width=beam, hotwords=hotwords or None,
                                     hotword_weight=hotword_weight)


def local_rows(audio_dir, sr):
    for w in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        t = os.path.splitext(w)[0] + ".txt"
        if not os.path.exists(t):
            continue
        with open(w, "rb") as f:
            x = D.decode_wav(f.read(), sr)
        if x is None:
            continue
        yield x, open(t, encoding="utf-8").read().strip(), os.path.basename(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="a local file, or hf://<repo>/<file>. A bare name is also looked up in "
                         "--hub-repo, so a wiped /kaggle/working costs nothing.")
    ap.add_argument("--hub-repo", default=None,
                    help="where to fetch --ckpt from when it is not on disk")
    ap.add_argument("--arch", default="arch.json")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7")
    ap.add_argument("--split", default="test")
    ap.add_argument("--audio-dir", default=None, help="score your own holdout instead of the split")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--max-seconds", type=float, default=16.0)
    ap.add_argument("--kenlm", default=None,
                    help="a KenLM .bin, decoded by pyctcdecode. Needs the `kenlm` python module, "
                         "which has no cp313 wheel -- use --arpa on Kaggle instead.")
    ap.add_argument("--arpa", default=None,
                    help="an lmplz .arpa, scored by arpa_lm.py (pure python, no compiler). "
                         "Beam search runs without an LM and the n-best list is rescored.")
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--hotwords", default=None, help="file, one estate term per line")
    ap.add_argument("--hotword-weight", type=float, default=10.0)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "off"])
    ap.add_argument("--out", default="eval.json")
    a = ap.parse_args()

    import jiwer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch, vocab = D.Arch(a.arch), D.Vocab(a.vocab)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[a.amp]
    if device.type == "cpu":
        amp_dtype = None
    load_dtype = torch.float32 if device.type == "cpu" else torch.float16
    print(f"device={device} amp={amp_dtype}")
    ckpt_path = resolve_ckpt(a.ckpt, a.hub_repo)
    if ckpt_path is None:
        raise SystemExit(
            f"no checkpoint at {a.ckpt!r}. A Kaggle session wipes /kaggle/working, but training "
            f"pushed it to the Hub -- pass --hub-repo <user>/sinhala-gemma-ctc, or "
            f"--ckpt hf://<user>/sinhala-gemma-ctc/last.pt")
    tower, head, ck = load_ckpt(ckpt_path, arch, len(vocab), device, load_dtype)
    print(f"  checkpoint step {ck['step']}")

    hot = []
    if a.hotwords and os.path.exists(a.hotwords):
        hot = [l.strip() for l in open(a.hotwords, encoding="utf-8") if l.strip()]
        print(f"  {len(hot)} hotwords")
    lm_decode = None
    if a.kenlm:
        lm_decode = make_decoder(a.kenlm, vocab, a.beam, a.alpha, a.beta, hot, a.hotword_weight)
        mode = f"KenLM beam {a.beam}"
    elif a.arpa:
        # Beam search with NO LM compiled in, then rescore the n-best with the ARPA ourselves.
        from pyctcdecode import build_ctcdecoder

        from arpa_lm import ArpaLM, rescore
        labels = ["" if i == vocab.blank else (" " if i == vocab.delim else vocab.itos.get(i, ""))
                  for i in range(len(vocab))]
        dec = build_ctcdecoder(labels)
        lm = ArpaLM(a.arpa)
        if hot:
            print(f"  note: --hotwords needs the kenlm path; ignored when rescoring an ARPA")

        def lm_decode(logits, _dec=dec, _lm=lm):
            beams = _dec.decode_beams(logits, beam_width=a.beam)
            return rescore(beams, _lm, a.alpha, a.beta)

        mode = f"ARPA rescore, beam {a.beam}"
    else:
        mode = "greedy"
    print(f"  decode: {mode}")

    feats = D.Features(arch)
    t_feat, t_enc, t_dec, secs = [], [], [], []
    refs, hyps, rows = [], [], []

    def one(wav, ref, name=""):
        t0 = time.perf_counter()
        f, m, flens = feats.batch([wav])
        t1 = time.perf_counter()
        with torch.no_grad():
            enc, enc_mask = encode(tower, {"features": f, "mask": m}, arch, device, amp_dtype)
            logits = head(enc.float())
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if enc_mask is not None and enc_mask.ndim == 2:
            L = int(enc_mask[0].sum())
        else:
            L = min(arch.enc_len(flens[0]), logits.shape[1])
        L = min(L, logits.shape[1])
        lg = logits[0, :L]
        if lm_decode:
            hyp = lm_decode(torch.log_softmax(lg, -1).cpu().numpy())
        else:
            hyp = vocab.ctc_collapse(lg.argmax(-1).cpu().tolist())
        t3 = time.perf_counter()
        t_feat.append((t1 - t0) * 1e3)
        t_enc.append((t2 - t1) * 1e3)
        t_dec.append((t3 - t2) * 1e3)
        secs.append(len(wav) / arch.sr)
        refs.append(D.norm_text(ref))
        hyps.append(hyp)
        rows.append({"name": name, "ref": D.norm_text(ref), "hyp": hyp,
                     "audio_s": round(len(wav) / arch.sr, 2)})

    if a.audio_dir:
        for wav, ref, name in local_rows(a.audio_dir, arch.sr):
            one(wav, ref, name)
    else:
        n = 0
        for r in D.rows(a.repo, a.split, streaming=True, lang=a.lang, shuffle_buffer=0):
            wav = D.decode_wav(r["audio"], arch.sr)
            if wav is None or not (0.4 <= len(wav) / arch.sr <= a.max_seconds):
                continue
            one(wav, r["text"], r.get("source_dataset", ""))
            n += 1
            if n % 200 == 0:
                print(f"  {n} rows...")
            if a.limit and n >= a.limit:
                break

    if not refs:
        raise SystemExit("nothing scored -- check --audio-dir or the split name")

    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    R, H = [p[0] for p in pairs], [p[1] for p in pairs]
    wer, cer = jiwer.wer(R, H), jiwer.cer(R, H)

    def p(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))]

    total = [f + e + d for f, e, d in zip(t_feat, t_enc, t_dec)]
    res = {
        "ckpt": a.ckpt, "step": ck["step"], "n": len(R),
        "decode": mode,
        "wer": round(wer, 4), "cer": round(cer, 4),
        "audio_s_median": round(st.median(secs), 2),
        "ms": {k: {"p50": round(p(v, 0.5), 1), "p95": round(p(v, 0.95), 1)}
               for k, v in (("features", t_feat), ("encoder", t_enc),
                            ("decode", t_dec), ("total", total))},
        "rtf_p50": round(p(total, 0.5) / 1e3 / st.median(secs), 4),
    }
    print(f"\n  n={res['n']}  WER {wer:.4f}  CER {cer:.4f}   decode={res['decode']}")
    print("  latency (ms)    p50     p95")
    for k in ("features", "encoder", "decode", "total"):
        print(f"    {k:10} {res['ms'][k]['p50']:7.1f} {res['ms'][k]['p95']:7.1f}")
    print(f"  real-time factor (p50): {res['rtf_p50']}  "
          f"-- Gemini measured 1766-2666 ms end to end in docs/voice/voice_timing.json")
    json.dump({"summary": res, "rows": rows[:500]},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
