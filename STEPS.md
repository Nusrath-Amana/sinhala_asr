# Every step, in order

Tick these off. Detail for each sits in [README.md](README.md) (why, architecture, limits) and
[KAGGLE.md](KAGGLE.md) (notebook cells, persistence, uploads).

Two gates below can stop the project early and are marked **GATE** — read those before spending
GPU hours. One item, step 3, has a long lead time and must start *now*, in parallel, not when you
reach phase 6.

---

## Phase 0 — one-off setup (~30 min, no GPU)

- [ ] **1. Hugging Face account** + a token with **write** access
      (Settings → Access Tokens). Create an empty model repo for checkpoints, e.g.
      `YOURNAME/sinhala-gemma-ctc`.
- [ ] **2. Kaggle account, phone-verified** — verification is what unlocks accelerators.
      Add the HF token under *Add-ons → Secrets* as `HF_TOKEN`.
- [ ] **3. START THE DATA QUESTION NOW.** Decide whether you may record real manager speech, get
      whoever owns that call to agree, and begin collecting. You need ~200–300 consented clips,
      hand-transcribed, as code-switched as people really speak. Nothing in phase 6 can happen
      without it, `cloud_stt.py` stores nothing by design, and the 34 wavs in
      `docs/voice/spike_audio/` are all synthetic or staged. **Lead time, not compute, is the
      critical path.**
- [ ] **4. Code onto Kaggle** — private GitHub repo (easier to iterate) or a zipped Kaggle Dataset.
      Do not upload `stratified_v7`; it streams.
- [ ] **5. Settle the licence.** `google/embeddinggemma-2` is `apache-2.0` on its own card, while
      the reference derivative claims Gemma Terms, and `stratified_v7`'s licence is listed
      **"unknown"** with `youtube` among its sources. These are different obligations and this is
      going into a commercial estate product. Answer it before training, not after.

## Phase 1 — CPU only, spends no GPU quota (~1 h)

- [ ] **6. `python probe.py`** → writes `arch.json`. Confirms the processor's argument name, the
      feature tensor name, encoder width (expect 1536) and the **frame rate**.
- [ ] **7. GATE: `python vocab.py --report --limit 20000`** → writes `vocab.json`, prints script
      mix and the grapheme-rate check.
      - Headroom at p95 ≥ ~1.4× → proceed on character units.
      - Below that → **switch to subword/BPE units before training.** CTC cannot emit more labels
        than it has frames; ignore this and the loss silently infs, `zero_infinity` hides it, and
        WER plateaus looking exactly like a bad learning rate.
      - Note the "Sinhala+Latin in the SAME row" percentage — that is your entire code-switching
        supply, and it will be small.
- [ ] **8. Commit `arch.json` and `vocab.json`** to the repo. A few hundred bytes each; every later
      session then starts straight at training.

## Phase 2 — smoke test (~30 min GPU)

- [ ] **9.** `python train.py --freeze-encoder --max-steps 200 --subset 500 --out /kaggle/working/ckpt`
      - Proves the pipeline end to end.
      - **Read `rows/s`** and compute `epoch_hours = 108278 / rows_per_sec / 3600`.
      - Decide the budget from that number before anything longer. Under ~8 rows/s, plan on two
        free weeks rather than one.

## Phase 3 — baseline to beat (~2–7 h GPU)

- [ ] **10.** `python train.py --freeze-encoder --max-steps 4000 --out /kaggle/working/ckpt --hub-repo YOURNAME/sinhala-gemma-ctc`
      (~1.2 epochs at the defaults)
- [ ] **11.** `python evaluate.py --ckpt ckpt/last.pt --split test --limit 2000` — record WER/CER.
      This is the head-only floor; everything later must beat it.

## Phase 4 — the real run (~5–21 h GPU, 2–3 sessions)

- [ ] **12.** `python train.py --unfreeze-top 8 --lr 2e-5 --max-steps 12000 --out /kaggle/working/ckpt --hub-repo YOURNAME/sinhala-gemma-ctc --resume ckpt/last.pt`
      - Use **Save & Run All (Commit)**, not an interactive session.
      - Re-run the identical command after each 12-hour cap; `--resume` pulls from the Hub.
- [ ] **13.** `python evaluate.py --ckpt ckpt/last.pt --split test --limit 2000` — greedy, no LM.
      Compare against **14.7 WER / 3.7 CER** (published, with LM) and **15.52 WER** (whisper-small
      finetune).

## Phase 5 — the 4-gram (CPU, ~1 day)

- [ ] **14. Build the KenLM.** Order 4, on Sinhala text (Wikipedia, FineWeb-2) **plus sentences
      generated from your own ontology** — divisions, field codes, task vocabulary, in the mixed
      templates people actually say. Remove dev/test sentences.
- [ ] **15.** `pip install pyctcdecode kenlm`, then
      `python evaluate.py --ckpt ckpt/last.pt --split test --kenlm lm/si_4gram.bin --beam 100`
      Tune `--alpha` / `--beta` / `--beam`.
- [ ] **16. GATE: isolate the LM's contribution.** Compare step 13 (no LM) against step 15 (LM).
      If no-LM lands near 18 WER and the LM carries most of the gain, the encoder was never the
      win — a Whisper-small finetune plus an LM is the cheaper path, and you should say so and
      stop here.

## Phase 6 — the decision this was all for

- [ ] **17.** Hand-transcribe the clips from step 3 into a folder of 16 kHz `.wav` files each with a
      matching `.txt`.
- [ ] **18. Re-measure `gemini-3.8-flash` yourself** on that same holdout. Do **not** reuse
      `spike_cascade.json` — that is the rejected `gemini-3.5-transcribe` (si CER 0.321, si+en
      0.618) on machine-spoken audio, and its own note calls it "an upper bound".
- [ ] **19.** `python evaluate.py --ckpt ckpt/last.pt --audio-dir holdout/ --kenlm lm/si_4gram.bin --hotwords estate_terms.txt`
      `estate_terms.txt` is one division name, field code or task term per line. This is the
      lexical control a prompted cloud model would not give you, and the main reason to prefer
      this stack at all.
- [ ] **20. Latency on the target device** — the phone, or whatever box would serve it. Never a
      Kaggle T4 figure against Gemini's network round-trip. Baseline to beat:
      **1766–2666 ms**, from `docs/voice/voice_timing.json`. Read the per-stage split from
      `evaluate.py`; a wide beam over a big 4-gram can cost more than the encoder.
- [ ] **21. Decide, and write it down:**
      - accuracy on *your* audio, Sinhala and code-switched separately
      - latency on the real device
      - where it would run — **not** the `micro` Lightsail container (0.25 vCPU, 1 GB, already at
        461 MB RSS)
      - Tamil stays on Gemini regardless; `stratified_v7` has none
      - Likely outcome worth planning for: ship as the **Sinhala route only**, on-device or on a
        separate box, with Gemini kept for Tamil, English and mixed speech.

---

## Budget

| phase | GPU hours |
|---|---|
| 1 (CPU only) | 0 |
| 2 smoke | ~0.5 |
| 3 baseline | 2–7 |
| 4 real run | 5–21 |
| 5 LM (CPU) | 0 |
| **total** | **~8–29 h** |

One free Kaggle week is ~30 pooled GPU-hours (TPU's 20 h is a separate pool this script can't use).
So one week if `rows/s` is kind, two if not. Phases 1 and 5 cost nothing — run them in CPU sessions.
