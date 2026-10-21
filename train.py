"""CTC head on EmbeddingGemma 2's audio tower -- Sinhala ASR, stratified_v7.

Staged on purpose, cheapest first, because a free Kaggle week is ~30 GPU-hours and one full
epoch over 108k rows is most of it:

  stage 1  --freeze-encoder --max-steps 200 --subset 500    pipeline smoke test, ~20 min
  stage 2  --freeze-encoder --max-steps 4000                head-only baseline to beat
  stage 3  --unfreeze-top 8 --lr 2e-5                       where the accuracy actually comes from

T4 and P100 HAVE NO bf16. --amp fp16 casts the encoder only; the CTC loss always runs in fp32,
because fp16 CTC gives NaN or a silent inf and that failure looks exactly like a bad LR.

Checkpoints every --save-every steps and on SIGTERM, so a 12-hour session cap costs nothing.
Resume with --resume ckpt/last.pt.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D


class CTCHead(nn.Module):
    """Deliberately small: LayerNorm, an optional bottleneck, then the projection to units.
    With the encoder frozen this is the only thing training, and anything larger just
    overfits 174 h."""

    def __init__(self, in_dim: int, vocab: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.mid = nn.Linear(in_dim, hidden) if hidden else None
        self.out = nn.Linear(hidden or in_dim, vocab)

    def forward(self, x):
        x = self.drop(self.norm(x))
        if self.mid is not None:
            x = F.gelu(self.mid(x))
        return self.out(x)


def resolve_ckpt(path: str, hub_repo: str | None) -> str | None:
    """Find a checkpoint locally, or pull it from the Hub.

    This is what lets one run move between Kaggle and Colab: both push `last.pt` to the same
    HF repo with --hub-repo, and either can pick the run up with the same --resume. Forms:
      ckpt/last.pt                 a local file
      hf://org/name/last.pt        an explicit Hub location
      ckpt/last.pt + --hub-repo    local if present, else that basename from the Hub
    """
    if path.startswith("hf://"):
        rest = path[5:].split("/")
        repo_id, fname = "/".join(rest[:2]), ("/".join(rest[2:]) or "last.pt")
    elif os.path.exists(path):
        return path
    elif hub_repo:
        repo_id, fname = hub_repo, os.path.basename(path)
    else:
        print(f"  no checkpoint at {path} and no --hub-repo -- starting fresh")
        return None
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=repo_id, filename=fname)
        print(f"  pulled {fname} from {repo_id}")
        return got
    except Exception as e:  # noqa: BLE001 - a missing remote checkpoint means "start fresh"
        print(f"  could not fetch {fname} from {repo_id} ({e}) -- starting fresh")
        return None


def build(arch: D.Arch, n_vocab: int, head_hidden: int, device, dtype):
    from transformers import AutoModel
    full = AutoModel.from_pretrained(arch.model, revision=arch.revision, dtype=dtype)
    tower = full.audio_tower
    del full  # the language and vision towers play no part in this task
    head = CTCHead(arch.encoder_dim, n_vocab, head_hidden)
    return tower.to(device), head.to(device).float()


def set_trainable(tower, unfreeze_top: int) -> int:
    for p in tower.parameters():
        p.requires_grad = False
    if unfreeze_top > 0:
        for layer in list(tower.layers)[-unfreeze_top:]:
            for p in layer.parameters():
                p.requires_grad = True
    return sum(p.numel() for p in tower.parameters() if p.requires_grad)


def encode(tower, batch, arch, device, amp_dtype):
    x = batch["features"].to(device)
    m = batch["mask"].to(device) if batch["mask"] is not None else None
    ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    with ctx:
        if arch.mask_key and m is not None:
            r = tower(**{arch.feat_key: x, arch.mask_key: m})
        else:
            r = tower(x)
    hidden = r[0] if isinstance(r, tuple) else getattr(r, "last_hidden_state", r)
    # The tower hands back its own downsampled frame mask (True = valid). Prefer it over
    # feature_len / stride, which drifts by a frame once a batch is padded.
    enc_mask = None if isinstance(r, tuple) else getattr(r, "attention_mask", None)
    return hidden, enc_mask


def fwd(tower, head, batch, arch, device, amp_dtype):
    enc, enc_mask = encode(tower, batch, arch, device, amp_dtype)
    logits = head(enc.float())  # head and loss stay in fp32
    if enc_mask is not None and enc_mask.ndim == 2:
        lens = enc_mask.sum(-1).long()
    else:
        lens = batch["enc_lens"].to(logits.device)
    return logits, lens.clamp(max=logits.shape[1])


def ctc_loss(logits, enc_lens, batch, blank: int):
    logp = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V), fp32
    return F.ctc_loss(logp, batch["targets"], enc_lens.clamp(max=logp.shape[0]),
                      batch["target_lens"], blank=blank, reduction="mean", zero_infinity=True)


@torch.no_grad()
def quick_wer(tower, head, arch, vocab, args, device, amp_dtype, feats, n: int = 200):
    import jiwer
    tower.eval()
    head.eval()
    refs, hyps = [], []
    for b in D.batches(args.repo, args.val_split, arch, vocab, batch_size=args.batch_size,
                       max_seconds=args.max_seconds, bucket=args.bucket, lang=args.lang,
                       features=feats, streaming=True, seed=0):
        logits, enc_lens = fwd(tower, head, b, arch, device, amp_dtype)
        ids = logits.argmax(-1).cpu().tolist()
        for seq, ref, L in zip(ids, b["texts"], enc_lens.cpu().tolist()):
            hyps.append(vocab.ctc_collapse(seq[:L]))
            refs.append(ref)
        if len(refs) >= n:
            break
    tower.train()
    head.train()
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 1.0, 1.0, 0
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return jiwer.wer(r, h), jiwer.cer(r, h), len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7",
                    help="a Hub id, or a directory of reshard.py output")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation",
                    help="set both to 'validation' for a first end-to-end run: it is 1.46 GB, so "
                         "it reshards in minutes, where train is 14.1 GB")
    ap.add_argument("--arch", default="arch.json")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--lang", default=None,
                    help="si or en; omit to keep v7's own mix (the 10k English rows are the anti-forgetting set)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--bucket", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="1e-3 head-only; 1e-5 to 3e-5 once the encoder is unfrozen")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--subset", type=int, default=0, help="stop after N rows (smoke tests)")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--unfreeze-top", type=int, default=0)
    ap.add_argument("--head-hidden", type=int, default=0)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "off"])
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--hub-repo", default=None, help="push checkpoints here (survives the 12 h cap)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch, vocab = D.Arch(args.arch), D.Vocab(args.vocab)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[args.amp]
    if amp_dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("!! this GPU has no bf16 (T4/P100) -- falling back to fp16")
        amp_dtype = torch.float16
    if device.type == "cpu":
        amp_dtype = None
    load_dtype = torch.float32 if device.type == "cpu" else torch.float16

    print(f"device={device} amp={amp_dtype} vocab={len(vocab)} "
          f"enc_dim={arch.encoder_dim} enc_fps={arch.fps}")
    tower, head = build(arch, len(vocab), args.head_hidden, device, load_dtype)
    n_unfrozen = set_trainable(tower, 0 if args.freeze_encoder else args.unfreeze_top)
    print(f"trainable: head {sum(p.numel() for p in head.parameters())/1e6:.2f}M "
          f"+ encoder {n_unfrozen/1e6:.2f}M")

    groups = [{"params": list(head.parameters()), "lr": args.lr}]
    if n_unfrozen:
        groups.append({"params": [p for p in tower.parameters() if p.requires_grad], "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype is torch.float16))

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, seen, t0 = 0, 0, time.time()

    resume_path = resolve_ckpt(args.resume, args.hub_repo) if args.resume else None
    if resume_path:
        ck = torch.load(resume_path, map_location=device)
        head.load_state_dict(ck["head"])
        if ck.get("tower"):
            tower.load_state_dict(ck["tower"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}")

    def save(tag="last"):
        p = os.path.join(args.out, f"{tag}.pt")
        torch.save({"head": head.state_dict(),
                    "tower": tower.state_dict() if n_unfrozen else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "args": vars(args), "vocab_size": len(vocab)}, p)
        print(f"  saved {p}")
        if args.hub_repo:
            try:
                from huggingface_hub import upload_file
                upload_file(path_or_fileobj=p, path_in_repo=f"{tag}.pt", repo_id=args.hub_repo)
            except Exception as e:  # noqa: BLE001 - a failed upload must not kill the run
                print(f"  hub upload failed: {e}")

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(now=True))

    feats = D.Features(arch)
    tower.train()
    head.train()
    run, nrun, micro, dropped = 0.0, 0, 0, 0
    print("step      loss      lr        rows/s   dropped")
    for b in D.batches(args.repo, args.train_split, arch, vocab, batch_size=args.batch_size,
                       max_seconds=args.max_seconds, bucket=args.bucket, lang=args.lang,
                       features=feats, streaming=True, epochs=100):
        on_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        logits, enc_lens = fwd(tower, head, b, arch, device, amp_dtype)
        loss = ctc_loss(logits, enc_lens, on_dev, vocab.blank)
        dropped += b["dropped"]
        if not torch.isfinite(loss):
            dropped += 1
            opt.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / args.accum).backward()
        run += loss.item()
        nrun += 1
        micro += 1
        seen += len(b["texts"])

        if micro % args.accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], args.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % 50 == 0:
                print(f"{step:6d}  {run/max(nrun,1):8.4f}  {sched.get_last_lr()[0]:.2e}  "
                      f"{seen/(time.time()-t0):7.1f}  {dropped}")
                run, nrun = 0.0, 0
            if step % args.save_every == 0:
                save("last")
            if args.eval_every and step % args.eval_every == 0:
                w, c, n = quick_wer(tower, head, arch, vocab, args, device, amp_dtype, feats)
                print(f"  [val n={n}] WER {w:.4f}  CER {c:.4f}")
            if step >= args.max_steps or stop["now"]:
                break
        if args.subset and seen >= args.subset:
            break

    save("last")
    w, c, n = quick_wer(tower, head, arch, vocab, args, device, amp_dtype, feats)
    print(f"FINAL [val n={n}] WER {w:.4f}  CER {c:.4f}  steps={step}  rows={seen}")
    json.dump({"wer": w, "cer": c, "n": n, "steps": step, "rows": seen},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
