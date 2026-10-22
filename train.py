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
import sys
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


def patch_fp16_attn(module, dtype) -> int:
    """Gemma4's audio attention masks invalid logits with `attention_invalid_logits_value`, which
    the config sets to -1e9. fp16 tops out at 65504, so the masked_fill raises

        RuntimeError: value cannot be converted to type c10::Half without overflow

    the first time a padded batch reaches attention. Clamp the constant to the dtype's own floor;
    anything that negative is zero after softmax, so the mask behaves identically.

    Not needed for bf16 (same exponent range as fp32) or fp32, and T4/P100 have no bf16 -- so on
    Kaggle this is the path that runs.
    """
    if dtype not in (torch.float16,):
        return 0
    lim = torch.finfo(dtype).min / 2          # -32752 for fp16
    seen, n = set(), 0
    for m in module.modules():
        c = getattr(m, "config", None)
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        v = getattr(c, "attention_invalid_logits_value", None)
        if v is not None and v < lim:
            c.attention_invalid_logits_value = lim
            n += 1
    return n


def build(arch: D.Arch, n_vocab: int, head_hidden: int, device, dtype, clamp_dtype=None):
    """`dtype` is how the weights are stored; `clamp_dtype` is the dtype the ACTIVATIONS will be.
    They differ once the encoder is trained: the weights must be fp32 for GradScaler, while
    autocast still produces fp16 attention logits that need the mask constant clamped."""
    from transformers import AutoConfig, AutoModel
    clamp_dtype = clamp_dtype or dtype
    # Patch the CONFIG BEFORE constructing the model. Attention modules capture the config object
    # at __init__, so clamping it afterwards by walking .modules() does not reliably reach the one
    # they actually read -- the -1e9 overflow came straight back.
    cfg = AutoConfig.from_pretrained(arch.model, revision=arch.revision)
    lim = float(torch.finfo(clamp_dtype).min / 2) if clamp_dtype == torch.float16 else None
    if lim is not None:
        for c in (cfg, getattr(cfg, "audio_config", None)):
            if c is not None and getattr(c, "attention_invalid_logits_value", 0) < lim:
                c.attention_invalid_logits_value = lim
        print(f"  attention_invalid_logits_value -> {lim:.0f} (fp16 cannot hold -1e9)")
    full = AutoModel.from_pretrained(arch.model, revision=arch.revision, config=cfg, dtype=dtype)
    tower = full.audio_tower
    del full  # the language and vision towers play no part in this task
    if lim is not None:
        patch_fp16_attn(tower, clamp_dtype)   # belt and braces for any config built separately
        seen = {getattr(getattr(m, "config", None), "attention_invalid_logits_value", None)
                for m in tower.modules()}
        seen.discard(None)
        print(f"  verified mask constants in the tower: {sorted(seen)}")
        bad = [v for v in seen if v < torch.finfo(clamp_dtype).min]
        if bad:
            raise SystemExit(f"mask constant {bad} still unrepresentable in {clamp_dtype}; "
                             f"re-run with --amp off to use fp32")
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
    # WEIGHT dtype. GradScaler refuses to unscale gradients of fp16 parameters ("Attempting to
    # unscale FP16 gradients"), so anything we actually train must be stored fp32 and reach fp16
    # only through autocast. With the encoder frozen it has no gradients, so fp16 weights are fine
    # there and halve its memory.
    train_encoder = (not args.freeze_encoder) and args.unfreeze_top > 0
    if device.type == "cpu" or train_encoder:
        load_dtype = torch.float32
    else:
        load_dtype = torch.float16
    print(f"weights={str(load_dtype).replace('torch.','')} "
          f"(encoder {'trained -> fp32 master weights' if train_encoder else 'frozen -> fp16'})")

    print(f"device={device} amp={amp_dtype} vocab={len(vocab)} "
          f"enc_dim={arch.encoder_dim} enc_fps={arch.fps}")
    tower, head = build(arch, len(vocab), args.head_hidden, device, load_dtype,
                        clamp_dtype=amp_dtype or load_dtype)
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
        # Stage 2 trains one param group (the head); stage 3 adds a second (the unfrozen encoder
        # layers). Loading stage 2's optimizer state into stage 3 raises "loaded state dict has a
        # different number of parameter groups" -- seconds into a multi-hour run. When the
        # trainable set has changed, keep the weights and restart the optimizer and schedule.
        prev, now = ck["opt"]["param_groups"], opt.param_groups
        same = len(prev) == len(now) and all(len(a["params"]) == len(b["params"])
                                             for a, b in zip(prev, now))
        if same:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            step = ck["step"]
            print(f"resumed at step {step}")
        else:
            print(f"loaded weights from step {ck['step']}; optimizer state dropped because the "
                  f"trainable set changed ({len(prev)} group(s) -> {len(now)}). "
                  f"Warmup and the cosine schedule restart from 0 -- which is what you want when "
                  f"unfreezing.")

    if args.hub_repo:
        # Create it up front: upload_file does NOT create a missing repo, and finding that out
        # from the first --save-every push would mean an hour of training pushed nowhere.
        try:
            from huggingface_hub import create_repo
            create_repo(args.hub_repo, exist_ok=True)
            print(f"checkpoints -> https://huggingface.co/{args.hub_repo}")
        except Exception as e:  # noqa: BLE001
            print(f"!! cannot use --hub-repo {args.hub_repo}: {e}\n"
                  f"!! training will run but NOTHING will be pushed; a session cap loses it all")

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
    # Time training only: the 1.5 GB model download and the processor fetch would otherwise
    # dominate a short run and make rows/s meaningless.
    t_train = time.time()
    print("step      loss      lr        rows/s   dropped", flush=True)
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

            if step % 10 == 0:
                print(f"{step:6d}  {run/max(nrun,1):8.4f}  {sched.get_last_lr()[0]:.2e}  "
                      f"{seen/max(time.time()-t_train,1e-9):7.1f}  {dropped}", flush=True)
                run, nrun = 0.0, 0
            # Save early once, then on the regular cadence: the first push is the only proof that
            # --hub-repo works, and waiting 500 steps for it makes a healthy run indistinguishable
            # from a hung one.
            if step == 25 or step % args.save_every == 0:
                save("last")
            if args.eval_every and step % args.eval_every == 0:
                w, c, n = quick_wer(tower, head, arch, vocab, args, device, amp_dtype, feats)
                print(f"  [val n={n}] WER {w:.4f}  CER {c:.4f}")
            if step >= args.max_steps or stop["now"]:
                break
        if args.subset and seen >= args.subset:
            break

    save("last")
    rate = seen / max(time.time() - t_train, 1e-9)
    w, c, n = quick_wer(tower, head, arch, vocab, args, device, amp_dtype, feats)
    print(f"FINAL [val n={n}] WER {w:.4f}  CER {c:.4f}  steps={step}  rows={seen}")
    print(f"THROUGHPUT {rate:.1f} rows/s  ->  1 epoch of train (108,278 rows) = "
          f"{108278/rate/3600:.1f} h", flush=True)
    json.dump({"wer": w, "cer": c, "n": n, "steps": step, "rows": seen,
               "rows_per_sec": round(rate, 2), "epoch_hours": round(108278 / rate / 3600, 2)},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)
    # `datasets` streaming leaves non-daemon worker threads alive, so the process can hang after
    # the last line is printed. Everything is saved by here; leave immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
