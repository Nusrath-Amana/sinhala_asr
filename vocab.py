"""Build the CTC unit vocabulary from stratified_v7's transcripts, and check the one thing
that silently ruins a CTC run: whether the encoder emits enough frames per second to hold
the labels.

CTC cannot emit more labels than it has frames. If the encoder gives (say) 6.25 frames/s and
a fast Sinhala utterance needs 14 graphemes/s, those rows can never be learned -- the loss
goes to inf (zero_infinity hides it) and WER plateaus for reasons that look like a bad LR.
--report prints that ratio against arch.json before you spend a Kaggle hour on it.

    python vocab.py --report --limit 20000
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import unicodedata

SPECIAL = ["<pad>", "<unk>", "|"]  # <pad> is index 0 = the CTC blank; "|" is the word delimiter


def norm(s: str) -> str:
    """NFC, collapse whitespace. Deliberately NOT lowercasing or stripping Latin: the estate's
    English words are part of the target, and casing carries meaning in field codes (13B)."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--min-count", type=int, default=5, help="chars rarer than this become <unk>")
    ap.add_argument("--arch", default="arch.json")
    ap.add_argument("--out", default="vocab.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(a.repo, split=a.split, streaming=True)
    counts: collections.Counter = collections.Counter()
    scripts: collections.Counter = collections.Counter()
    langs: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    mixed = 0
    rates: list[tuple[float, str]] = []
    n = 0

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
        if a.report and row.get("audio"):
            try:
                import soundfile as sf
                info = sf.info(io.BytesIO(row["audio"]))
                secs = info.frames / info.samplerate
                if secs > 0.2:
                    rates.append((len(t) / secs, t[:60]))
            except Exception:  # noqa: BLE001 - a bad row must not stop vocab building
                pass
        n += 1
        if a.limit and n >= a.limit:
            break

    kept = [c for c, k in counts.most_common() if k >= a.min_count and not c.isspace()]
    vocab = {tok: i for i, tok in enumerate(SPECIAL + sorted(kept))}
    dropped = sum(k for c, k in counts.items() if k < a.min_count and not c.isspace())

    print(f"rows scanned      {n:,}")
    print(f"language          {dict(langs)}")
    print(f"source_dataset    {dict(sources.most_common())}")
    print(f"vocab size        {len(vocab)} (blank=0 <unk>=1 word-delim='|'=2)")
    print(f"chars below --min-count {a.min_count}: {dropped:,} occurrences -> <unk>")
    print("\nrows containing each script:")
    for k, v in scripts.most_common():
        print(f"  {k:8} {v:7,}  {100*v/max(n,1):5.1f}%")
    print(f"\nSinhala+Latin in the SAME row: {mixed:,} ({100*mixed/max(n,1):.2f}%)"
          "   <- this is your code-switching supply")

    if a.report and rates:
        rates.sort(key=lambda x: -x[0])
        import statistics as st
        rr = [r for r, _ in rates]
        print(f"\ngrapheme rate (chars/s over {len(rr):,} rows): "
              f"p50 {st.median(rr):.1f}  p95 {rr[int(0.05*len(rr))]:.1f}  max {rr[0]:.1f}")
        try:
            fps = json.load(open(a.arch, encoding="utf-8"))["encoder_frames_per_second"]
        except Exception:  # noqa: BLE001
            print("  (no arch.json yet -- run probe.py to compare against the frame rate)")
        else:
            p95 = rr[int(0.05 * len(rr))]
            print(f"encoder frames/s from {a.arch}: {fps}")
            over = sum(1 for r in rr if r > fps)
            verdict = "OK" if p95 < 0.7 * fps else "TOO TIGHT -- see note below"
            print(f"  headroom at p95: {fps/p95:.2f}x   rows needing more frames than exist: "
                  f"{over:,} ({100*over/len(rr):.2f}%)   -> {verdict}")
            if p95 >= 0.7 * fps:
                print("  Fix options, cheapest first: (1) a subword/BPE unit set instead of characters, "
                      "which shortens targets ~2-3x; (2) drop the encoder's final subsampling; "
                      "(3) drop the fastest rows. Do NOT just set zero_infinity and train on.")
            print(f"  worst rows: " + " | ".join(f"{r:.0f}c/s {t[:28]!r}" for r, t in rates[:3]))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "blank_id": 0, "unk_id": 1, "delim_id": 2,
                   "normalise": "NFC + whitespace collapse, case preserved"}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
