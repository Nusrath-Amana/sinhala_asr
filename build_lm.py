"""Build the Sinhala 4-gram KenLM that CTC decoding needs. CPU only -- no GPU quota.

Greedy CTC picks the best character per frame independently. A 4-gram scores whole hypotheses
during beam search, so it fixes exactly what greedy gets wrong: plausible-sounding character runs
that are not Sinhala words. On CTC systems this is typically worth 20-40% relative WER, and it is
the difference between the published 14.7 and whatever that model scores greedy.

    python build_lm.py --out lm/si_4gram.bin --push-to <you>/sinhala-gemma-ctc

TEST CONTAMINATION. The LM is built from TRAIN transcripts only, and any sentence appearing in
validation or test is removed -- v7 is text-disjoint across splits, but a shared sentence would let
the LM memorise an eval answer and quietly inflate the score.

THE ESTATE'S OWN WORDS. --extra-text takes a file of lines to fold in (division names, field codes,
task vocabulary, in the mixed phrasings people actually speak), repeated --extra-weight times so the
4-gram treats them as common rather than freak events. This is the lexical control that made the
local stack worth trying: a prompted cloud model ignored the estate's vocabulary, and here you own
the language model. Use the same terms as `evaluate.py --hotwords`.

KenLM is a C++ tool; `pip install kenlm` ships only the scorer. This builds lmplz from source if it
is missing (~3 min on Kaggle, which runs as root so apt-get works).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import unicodedata

TEXT_COLS = ["text"]


def sh(cmd: str, **kw) -> None:
    print(f"  $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def ensure_kenlm(prefix: str) -> str:
    """Return a path to lmplz, building KenLM if needed."""
    found = shutil.which("lmplz")
    if found:
        return found
    local = os.path.join(prefix, "kenlm", "build", "bin", "lmplz")
    if os.path.exists(local):
        return local
    print("lmplz not found -- building KenLM from source (~3 min)", flush=True)
    os.makedirs(prefix, exist_ok=True)
    sh("apt-get -qq update && apt-get -qq install -y build-essential cmake "
       "libboost-all-dev libbz2-dev liblzma-dev zlib1g-dev > /dev/null")
    sh(f"cd {prefix} && rm -rf kenlm && git clone -q https://github.com/kpu/kenlm.git")
    sh(f"cd {prefix}/kenlm && mkdir -p build && cd build && "
       f"cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j$(nproc) lmplz build_binary > /dev/null")
    if not os.path.exists(local):
        raise SystemExit("KenLM build finished but lmplz is missing -- check the output above")
    return local


def norm(s: str) -> str:
    """Must match vocab.py exactly, or the LM scores strings the decoder can never emit."""
    return " ".join(unicodedata.normalize("NFC", s).split())


def texts(repo: str, split: str) -> list[str]:
    """The text column, read straight from the Hub. Seconds, and no audio is fetched."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    t0 = time.time()
    pf = pq.ParquetFile(fs.open(f"datasets/{repo}/{split}.parquet", "rb"))
    col = pf.read(columns=TEXT_COLS).column("text").to_pylist()
    print(f"  {split}: {len(col):,} lines in {time.time()-t0:.1f}s", flush=True)
    return col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data-v7")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", default="lm/si_4gram.bin")
    ap.add_argument("--extra-text", default=None, help="estate vocabulary, one line per phrase")
    ap.add_argument("--extra-weight", type=int, default=200, help="times to repeat --extra-text")
    ap.add_argument("--prune", default="0 0 1", help="lmplz --prune; '' disables")
    ap.add_argument("--keep-arpa", action="store_true")
    ap.add_argument("--work", default="/kaggle/temp/lm", help="scratch for the corpus and KenLM")
    ap.add_argument("--push-to", default=None, help="HF repo to upload the .bin to")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    lmplz = ensure_kenlm(a.work)
    build_binary = os.path.join(os.path.dirname(lmplz), "build_binary")

    print("\ncollecting transcripts")
    held = {norm(t) for s in ("validation", "test") for t in texts(a.repo, s)}
    train = [norm(t) for t in texts(a.repo, "train")]
    lines = [t for t in train if t and t not in held]
    print(f"  train {len(train):,} -> {len(lines):,} after removing {len(train)-len(lines):,} "
          f"sentences shared with validation/test")

    if a.extra_text:
        extra = [norm(l) for l in open(a.extra_text, encoding="utf-8") if l.strip()]
        lines.extend(extra * a.extra_weight)
        print(f"  + {len(extra):,} estate phrases x{a.extra_weight} = {len(extra)*a.extra_weight:,} lines")

    corpus = os.path.join(a.work, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  corpus: {len(lines):,} lines, {os.path.getsize(corpus)/1e6:.1f} MB -> {corpus}")

    arpa = os.path.join(a.work, f"si_{a.order}gram.arpa")
    prune = f"--prune {a.prune}" if a.prune.strip() else ""
    print(f"\nlmplz -o {a.order} {prune}")
    # discount_fallback: the corpus is small enough that some n-gram orders have too few
    # singletons for Kneser-Ney to estimate discounts, and lmplz aborts without it.
    sh(f'"{lmplz}" -o {a.order} {prune} --discount_fallback -S 40% -T {a.work} '
       f'< "{corpus}" > "{arpa}"')
    print(f"  arpa {os.path.getsize(arpa)/1e6:.1f} MB")

    print(f"\nbuild_binary -> {a.out}")
    sh(f'"{build_binary}" "{arpa}" "{a.out}"')
    print(f"  {a.out}: {os.path.getsize(a.out)/1e6:.1f} MB")
    if not a.keep_arpa:
        os.remove(arpa)

    if a.push_to:
        from huggingface_hub import create_repo, upload_file
        create_repo(a.push_to, exist_ok=True)
        upload_file(path_or_fileobj=a.out, path_in_repo=os.path.basename(a.out),
                    repo_id=a.push_to)
        print(f"  pushed to https://huggingface.co/{a.push_to}")

    print("\nNext -- score greedy against LM-decoded on the SAME checkpoint:")
    print("  python evaluate.py --ckpt ckpt/last.pt --split test --limit 2000")
    print(f"  python evaluate.py --ckpt ckpt/last.pt --split test --limit 2000 --kenlm {a.out}")
    print("The gap between those two is how much of the published 14.7 was ever the encoder.")


if __name__ == "__main__":
    main()
