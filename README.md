# Sinhala ASR — EmbeddingGemma 2 audio tower + CTC

Step one of replacing `gemini-3.8-flash` STT in Yaha's voice path: a CTC reader on
EmbeddingGemma 2's audio tower, trained on SinhaSpeech `stratified_v7`, scored against the
incumbent. Sinhala only at this stage, by decision.

## What this reproduces

[`ovindu-a/sinhala-gemma-asr`](https://huggingface.co/ovindu-a/sinhala-gemma-asr) — a CTC head on
the audio tower of [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2),
decoded with a Sinhala 4-gram KenLM. Published: **WER 14.7 / CER 3.7** on the v6 test set, from
174 h (SinhaSpeech v6 + OpenSLR-52).

Targets to beat, in order:

| system | WER | CER |
|---|---|---|
| gemma-CTC + KenLM (published) | 14.7 | 3.7 |
| `whisper-small-sinhala-run12-best` | 15.52 | 4.36 |
| `whisper-small-sinhala-run13` | 17.08 | 4.55 |

Note the published gap over a plain Whisper-small finetune is only ~0.8 WER, and part of it is
the KenLM rather than the encoder. `evaluate.py` scores with and without LM decoding so you can
tell which. If no-LM lands near 18, the encoder isn't the win.

## Architecture, read from the weights (not guessed)

From the `model.safetensors` header of `google/embeddinggemma-2`:

- `audio_tower.layers.*` — 12 Conformer blocks, hidden 1024, 8 heads (752 tensors)
- `audio_tower.subsample_conv_projection` — conv frontend, kernel 5, channels [128, 32]
- `audio_tower.output_proj` — → 1536 (`output_proj_dims`)
- `attention_context_right: 0` — causal, so streaming stays possible later
- also in the checkpoint but unused here: `language_model.*`, `vision_tower.*`, `embed_audio.*`

740M total = 270M text + 170M vision + **300M audio**. The audio tower alone is *larger* than all
of whisper-small (244M), so the speed case is not parameter count. It rests on CTC being
non-autoregressive — one forward pass plus a beam search, versus Whisper's token-by-token
decode — and on dropping the network hop.

**Licence:** the base model is `apache-2.0` per its own card. The derivative above is labelled
`license: gemma` and claims Gemma Terms apply. Settle which governs before this ships in a
commercial product; they are not the same obligation. `stratified_v7`'s own licence is listed
**"unknown"** and its `source_dataset` includes `youtube`.

Running it on Kaggle or Colab -- what to upload, what survives a 12-hour cap, and the notebook
cell by cell -- is in [KAGGLE.md](KAGGLE.md).

## Order of operations

```bash
pip install -r requirements.txt

python probe.py                       # writes arch.json — RUN FIRST
python vocab.py --report --limit 20000 # writes vocab.json + the frame-rate check
python train.py --freeze-encoder --max-steps 200 --subset 500   # smoke test
python train.py --freeze-encoder --max-steps 4000               # head-only baseline
python train.py --unfreeze-top 8 --lr 2e-5 --max-steps 12000    # the real run
python evaluate.py --ckpt ckpt/last.pt --split test --limit 2000
```

Nothing hardcodes a model assumption. `probe.py` measures the processor's argument name, the
feature tensor's name, the encoder's output width and its **frame rate**, and writes `arch.json`;
everything else reads that file. Run it first or the rest will fail on a missing `arch.json`,
deliberately.

### Why `vocab.py --report` matters

CTC cannot emit more labels than it has frames. If the tower's frame rate is low and Sinhala
graphemes-per-second is high, those rows are unlearnable — the loss goes to inf, `zero_infinity`
swallows it, and WER plateaus looking exactly like a bad learning rate. The report prints
headroom at p95 and names the fix (subword units, usually) before you spend a Kaggle hour.

## Free-GPU notes

Kaggle: ~30 GPU-h/week, 2×T4 16 GB or 1×P100, **12-hour session cap**. Colab free T4 as overflow.

- **T4 and P100 have no bf16**, though the model config says bf16. `train.py` detects this and
  falls back to fp16, keeping the CTC loss in fp32 — fp16 CTC NaNs silently.
- Streaming is the default; 17.2 GB is not worth a session's disk.
- `--hub-repo` pushes `last.pt` on every save, so the 12-hour kill costs nothing. Resume with
  `--resume ckpt/last.pt`.
- Pick the **single P100** over 2×T4: this script is single-GPU, and one P100 beats one T4. Using
  both T4s would need DDP, which is not worth it at this scale.

### How long it takes — measure, don't guess

Don't trust an a-priori estimate here: the cost per row is set by the encoder's frame rate, which
`probe.py` measures and which the conv frontend makes coarse. With a frozen encoder this run is
likely **data-bound, not GPU-bound** — streaming 17.2 GB of parquet, decoding WAV and computing
features on Kaggle's 4 vCPU, not the 300M forward pass.

So `train.py` prints `rows/s`. Run stage 1 (~20 min), read it off, and multiply:

    epoch_hours = 108278 / rows_per_sec / 3600

| rows/s observed | 1 epoch | stage 2 (~1.2 ep) | stage 3 (~3.5 ep) |
|---|---|---|---|
| 20 | 1.5 h | ~2 h | ~5 h |
| 10 | 3 h | ~4 h | ~11 h |
| 5 | 6 h | ~7 h | ~21 h |
| 2 | 15 h | ~18 h | — exceeds a week |

At the default `--batch-size 8 --accum 4` one optimizer step is 32 rows, so stage 2's 4000 steps
is ~1.2 epochs and stage 3's 12000 steps is ~3.5 epochs. Budget **one free week (30 h) for a
first comparable number, two if rows/s lands below ~8**. Stage 3 will span 2–3 sessions under the
12-hour cap; that is what `--hub-repo` and `--resume` are for.

If rows/s is the bottleneck rather than the GPU, raise `--batch-size` before anything else, and
consider caching features once to disk instead of recomputing them every epoch.

### Where each step runs

| step | needs GPU? | run it on |
|---|---|---|
| `probe.py` | no | Kaggle **CPU** session (downloads ~1.5 GB of weights) |
| `vocab.py --report` | no | Kaggle **CPU** session (streams ~3 GB at 20k rows) |
| `train.py` | yes | Kaggle GPU / Colab |
| `evaluate.py` accuracy | yes | Kaggle GPU / Colab |
| `evaluate.py` latency | — | **the target device only** — a phone or the box you'd serve from |

Do not spend GPU quota on the first two: they are pure CPU, and a **Kaggle CPU session does not
draw on your 30 GPU-hours**. `arch.json` and `vocab.json` are a few hundred bytes each — generate
them in a CPU session, download them, commit them, and every later GPU session starts ready.

Your own machine (2 GB VRAM, 7.7 GB RAM, non-admin) can run `probe.py` on CPU if you'd rather,
but it costs you a ~1.5 GB download and the disk. It cannot train this at any batch size.

### Kaggle + Colab together

Yes — separate accounts, separate quotas, and both are worth using. But correct the budget first:
Kaggle gives **one pooled ~30 GPU-hours/week**, not 30 h of T4 *plus* 20 h of P100. Choosing P100
over T4 picks a machine, not a second allowance. The separate 20 h/week is **TPU v3-8** (9 h per
session), a different pool that this script would need an XLA rewrite to touch.

A workable split:

- **Colab free T4** — stages 1 and 2, and evaluation runs. Disconnects are unpredictable and idle
  timeouts are short, which is fine for runs measured in hours.
- **Kaggle P100** — stage 3, the long one, using the 12-hour sessions.

Checkpoints move between them through the Hub, so the platform is interchangeable:

```bash
# on either platform
python train.py --unfreeze-top 8 --lr 2e-5 --hub-repo <you>/sinhala-gemma-ctc \
                --resume ckpt/last.pt
```

`--resume` falls back to the Hub when the local file is absent, so the same command continues the
run wherever it lands; `hf://<you>/sinhala-gemma-ctc/last.pt` names a checkpoint explicitly.
Stage 3 under a 12-hour cap is 2–3 sessions either way.

Your own machine (2 GB VRAM, 7.7 GB RAM) can run `probe.py` on CPU; it cannot train this.

## Known limits of this step

- **Sinhala only.** `stratified_v7` has zero Tamil. Yaha's voice layer is si/ta/en, so this
  cannot replace Gemini wholesale — at best it takes the Sinhala route.
- **Code-switching is not covered.** The 10k LibriSpeech rows are *anti-forgetting* (keeping
  English alive when a whole utterance is English), not mixed-utterance data. Every v7 row is
  pure one language. `vocab.py` prints how many rows actually contain both scripts.
- **A Sinhala-only KenLM fights code-switching**, assigning near-zero probability to a Latin word
  mid-Sinhala-sentence and pushing the beam away from the right answer. The counter is in
  `evaluate.py`: `--hotwords`, a file of division names, field codes and task terms, boosted at
  decode time. That lexical control is the one respect in which this beats a prompted cloud model.
- **Production can't serve it.** The Lightsail container is `micro` — 0.25 vCPU, 1 GB RAM, already
  at 461 MB. Serving means a separate box, a serverless GPU, or on-device in the Flutter app.
- **No in-domain eval set exists yet.** `cloud_stt.py` stores nothing, by design, and the 34 wavs
  in `docs/voice/spike_audio/` are all synthetic or staged. `evaluate.py --audio-dir` takes a
  folder of real 16 kHz wavs with matching `.txt` once you have consented recordings.

## Baseline to beat

From `meddecombra-ontology/docs/voice/voice_timing.json`, Gemini `hear` on real estate clips,
3 repeats: **1766–2666 ms**, flat across 1.1 s–5.1 s of audio. That time is round-trip, not
inference, which is why a local model can win on latency even while being a bigger model.

Accuracy baseline must be re-measured, not taken from `spike_cascade.json` — that file is the
*rejected* `gemini-3.5-transcribe` (si CER 0.321, si+en 0.618) on machine-spoken audio, and its
own note calls it "an upper bound".
