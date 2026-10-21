"""Build the CTC unit vocabulary from stratified_v7's transcripts, and check the one thing that
silently ruins a CTC run: whether the encoder emits enough frames per second to hold the labels.

CTC cannot emit more labels than it has frames. At the measured 25 frames/s, a row needing more
graphemes per second than that can never be learned -- the loss goes to inf, zero_infinity hides
it, and WER plateaus looking exactly like a bad learning rate. --report prints that headroom
against arch.json before you spend a Kaggle hour on it.

MEMORY -- WHY THIS DOES NOT USE `datasets`. Every split of this dataset is ONE parquet row group:
train is 108,278 rows in a single 14.6 GB group, of which `audio` is 14.1 GB and `text` just 5.3 MB.
A row group is the unit a parquet reader materialises, so `load_dataset(streaming=True)` tries to
pull all 14 GB and Kaggle kills the session before the first row arrives. It is also why the HF
dataset viewer reports "Scan size limit exceeded" on this repo.

pyarrow can prune to the columns it needs, which is the whole fix:

  pass 1  `text` only, read straight from the Hub with range requests -- 108,278 rows in ~4 s and
          ~12 MB, because the audio pages are never fetched. --limit defaults to every row.
  pass 2  only with --report. Durations need the audio, and no reader can take a slice of a single
          14 GB group, so this uses the `validation` split instead (1.46 GB, same sources, so the
          same grapheme-rate distribution) and caps rows at --rate-sample.

    python vocab.py --report
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import unicodedata

SPECIAL = ["<pad>", "<unk>", "|"]  # <pad> is index 0 = the CTC blank; "|" is the word delimiter
TEXT_COLS = ["text", "language", "source_dataset"]


def norm(s: str) -> str:
    """NFC, collapse whitespace. Deliberately NOT lowercasing or stripping Latin: the estate's
    English words are part of the target, and case carries meaning in field codes (13B)."""
    return " ".join(unicodedata.normalize("NFC", s).split())


def script_of(ch: str) -> str:
    o = ord(ch)
    if 0x0D80 <= o <= 0x0DFF:
        return "sinhala"
    if 0x0B80 <= o <= 0x0BFF:
        return "tamil"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"


def open_split(repo: str, split: str):
    """A ParquetFile over the Hub, read by range request. Nothing is downloaded until a column
    is actually asked for."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    return pq.ParquetFile(fs.open(f"datasets/{repo}/{split}.parquet", "rb"))


def text_pass(repo: str, split: str, limit: int):
    """Vocabulary and script statistics. Reads the text column and nothing else."""
    import time
    t0 = time.time()
    pf = open_split(repo, split)
    print(f"  {pf.metadata.num_rows:,} rows in the file; fetching the text column only", flush=True)
    table = pf.read(columns=TEXT_COLS)
    print(f"  text column in {time.time()-t0:.1f}s ({table.nbytes/1e6:.1f} MB)", flush=True)
    ds = ({"text": t, "language": l, "source_dataset": sd} for t, l, sd in zip(
        table.column("text").to_pylist(),
        table.column("language").to_pylist(),
        table.column("source_dataset").to_pylist()))
    counts: collections.Counter = collections.Counter()
    scripts: collections.Counter = collections.Counter()
    langs: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    mixed = n = 0
    for row in ds:
        t = norm(row["text"])
        counts.update(t)
        langs[row.get("language", "?")] += 1
        sources[row.get("source_dataset", "?")] += 1
        s = {script_of(c) for c in t}
        for k in s:
            scripts[k] += 1
        if "sinhala" in s and "latin" in s:
            mixed += 1
        n += 1
        if limit and n >= limit:
            break
        # An early heartbeat, then regular ones: the first parquet row group takes a while to
        # open and a silent terminal is indistinguishable from a hung stream.
        if n % 25000 == 0:
            print(f"  counted {n:,}", flush=True)
    print(f"  done: {n:,} rows in {time.time()-t0:.1f}s total", flush=True)
    return counts, scripts, langs, sources, mixed, n


def rate_pass(repo: str, split: str, sample: int):
    """Grapheme rate. Reads the audio column of ONE split into memory -- so this runs on
    `validation` (1.46 GB), never `train` (14.1 GB). Only WAV headers are parsed, not samples."""
    import time
    import soundfile as sf
    t0 = time.time()
    pf = open_split(repo, split)
    print(f"  reading {split} audio+text ({pf.metadata.num_rows:,} rows, ~1.5 GB)...", flush=True)
    table = pf.read(columns=["audio", "text"])
    print(f"  in memory after {time.time()-t0:.0f}s", flush=True)
    audio, texts = table.column("audio"), table.column("text")
    rates: list[float] = []
    worst: list[tuple[float, str]] = []
    for i in range(min(len(texts), sample)):
        t = norm(texts[i].as_py())
        b = audio[i].as_py()
        if not t or not b:
            continue
        try:
            info = sf.info(io.BytesIO(b))
            secs = info.frames / info.samplerate
        except Exception:  # noqa: BLE001 - 16 kHz PCM16 fallback if the header will not parse
            secs = max(0.0, (len(b) - 44) / 32000)
        if secs > 0.2:
            r = len(t) / secs
            rates.append(r)
            worst.append((r, t[:40]))
            worst.sort(key=lambda x: -x[0])
            del worst[3:]
    return rates, worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 = every row (text pass is cheap)")
    ap.add_argument("--rate-sample", type=int, default=4000, help="rows scored for --report")
    ap.add_argument("--rate-split", default="validation",
                    help="split used for durations: validation (1.46 GB) or test (1.63 GB). "
                         "NEVER train -- its single row group is 14.1 GB of audio.")
    ap.add_argument("--min-count", type=int, default=5, help="chars rarer than this become <unk>")
    ap.add_argument("--arch", default="arch.json")
    ap.add_argument("--out", default="vocab.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    print(f"pass 1: text only ({'all rows' if not a.limit else f'{a.limit:,} rows'})")
    counts, scripts, langs, sources, mixed, n = text_pass(a.repo, a.split, a.limit)

    kept = [c for c, k in counts.most_common() if k >= a.min_count and not c.isspace()]
    vocab = {tok: i for i, tok in enumerate(SPECIAL + sorted(kept))}
    dropped = sum(k for c, k in counts.items() if k < a.min_count and not c.isspace())

    print(f"\nrows scanned      {n:,}")
    print(f"language          {dict(langs)}")
    print(f"source_dataset    {dict(sources.most_common())}")
    print(f"vocab size        {len(vocab)} (blank=0 <unk>=1 word-delim='|'=2)")
    print(f"chars below --min-count {a.min_count}: {dropped:,} occurrences -> <unk>")
    print("\nrows containing each script:")
    for k, v in scripts.most_common():
        print(f"  {k:8} {v:7,}  {100*v/max(n,1):5.1f}%")
    print(f"\nSinhala+Latin in the SAME row: {mixed:,} ({100*mixed/max(n,1):.2f}%)"
          "   <- your whole code-switching supply")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "blank_id": 0, "unk_id": 1, "delim_id": 2,
                   "normalise": "NFC + whitespace collapse, case preserved"}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nwrote {a.out}")

    if not a.report:
        return

    if a.rate_split == "train":
        raise SystemExit("--rate-split train would load a 14.1 GB audio column in one row group "
                         "and be OOM-killed. Use validation (1.46 GB) or test (1.63 GB).")
    print(f"\npass 2: grapheme rate, {a.rate_sample:,} rows from the {a.rate_split} split")
    rates, worst = rate_pass(a.repo, a.rate_split, a.rate_sample)
    if not rates:
        print("  no durations read -- skipping the check")
        return
    import statistics as st
    rates.sort(reverse=True)
    p95 = rates[int(0.05 * len(rates))]
    print(f"  chars/s over {len(rates):,} rows: p50 {st.median(rates):.1f}  "
          f"p95 {p95:.1f}  max {rates[0]:.1f}")
    try:
        fps = json.load(open(a.arch, encoding="utf-8"))["encoder_frames_per_second"]
    except Exception:  # noqa: BLE001
        print("  (no arch.json -- run probe.py to compare against the frame rate)")
        return
    over = sum(1 for r in rates if r > fps)
    print(f"  encoder frames/s: {fps}")
    print(f"  GATE: headroom at p95 {fps/p95:.2f}x   rows needing more frames than exist: "
          f"{over:,} ({100*over/len(rates):.2f}%)")
    if fps / p95 >= 1.4:
        print("  -> OK on character units. Proceed.")
    else:
        print("  -> TOO TIGHT. Switch to subword/BPE units (shortens targets ~2-3x) before "
              "training, or drop the encoder's final subsampling. Do NOT just rely on "
              "zero_infinity and train anyway.")
    print("  worst rows: " + " | ".join(f"{r:.0f}c/s {t!r}" for r, t in worst))


if __name__ == "__main__":
    main()
