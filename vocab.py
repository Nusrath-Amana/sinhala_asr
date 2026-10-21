"""Build the CTC unit vocabulary from stratified_v7's transcripts, and check the one thing that
silently ruins a CTC run: whether the encoder emits enough frames per second to hold the labels.

CTC cannot emit more labels than it has frames. At the measured 25 frames/s, a row needing more
graphemes per second than that can never be learned -- the loss goes to inf, zero_infinity hides
it, and WER plateaus looking exactly like a bad learning rate. --report prints that headroom
against arch.json before you spend a Kaggle hour on it.

MEMORY. Two passes on purpose, because the `audio` column is megabytes per row and pulling it to
count characters is what OOMs a Kaggle session:

  pass 1  text only, via select_columns -- the audio bytes are never read at all. Cheap over
          every row, so --limit defaults to the whole split.
  pass 2  only with --report, and only over --rate-sample rows (default 1500), because durations
          need the audio. A sample is plenty to find the p95 grapheme rate.

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


def text_pass(repo: str, split: str, limit: int):
    """Vocabulary and script statistics, without ever touching the audio column."""
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    try:
        ds = ds.select_columns(TEXT_COLS)
    except Exception as e:  # noqa: BLE001 - older datasets, or a column renamed upstream
        print(f"  (select_columns unavailable: {e} -- falling back, this will use more memory)")
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
        if n % 20000 == 0:
            print(f"  {n:,} rows...")
    return counts, scripts, langs, sources, mixed, n


def rate_pass(repo: str, split: str, sample: int):
    """Grapheme rate, over a bounded sample. Only the header of each clip is parsed -- sf.info
    reads the WAV header, not the samples -- and nothing is retained but the rates."""
    import soundfile as sf
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    rates: list[float] = []
    worst: list[tuple[float, str]] = []
    n = 0
    for row in ds:
        t = norm(row["text"])
        try:
            info = sf.info(io.BytesIO(row["audio"]))
            secs = info.frames / info.samplerate
        except Exception:  # noqa: BLE001 - a bad row must not stop the scan
            continue
        finally:
            row = None
        if secs > 0.2 and t:
            r = len(t) / secs
            rates.append(r)
            worst.append((r, t[:40]))
            worst.sort(key=lambda x: -x[0])
            del worst[3:]          # keep three, not twenty thousand
        n += 1
        if n >= sample:
            break
    return rates, worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 = every row (text pass is cheap)")
    ap.add_argument("--rate-sample", type=int, default=1500, help="rows read WITH audio for --report")
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

    print(f"\npass 2: grapheme rate over {a.rate_sample:,} rows (reads audio headers)")
    rates, worst = rate_pass(a.repo, a.split, a.rate_sample)
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
