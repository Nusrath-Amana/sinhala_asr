"""A pure-Python ARPA n-gram with Katz backoff. No kenlm, no compiler.

WHY THIS EXISTS. `pyctcdecode` scores an LM inside its beam search through the `kenlm` Python
module, which has no cp313 wheel and whose source build fails on Kaggle's Python 3.13 (the C++
`lmplz` compiles fine; it is only the binding that breaks). Rather than depend on that, we read
the `.arpa` that `lmplz` already produced and rescore pyctcdecode's n-best list ourselves.

Rescoring is weaker than true shallow fusion -- the LM cannot steer the beam, only reorder what
the beam found -- so treat it as a lower bound on the LM's value. With a wide enough beam the two
usually land close, and it answers the question that matters: how much does the LM buy at all.

ARPA is log10. Scores here stay log10 so alpha/beta keep the meaning pyctcdecode gives them.
"""
from __future__ import annotations

import math

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class ArpaLM:
    def __init__(self, path: str, verbose: bool = True):
        self.logp: dict[tuple[str, ...], float] = {}
        self.bo: dict[tuple[str, ...], float] = {}
        self.order = 0
        n = 0
        order = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("\\") and line.endswith(":"):
                    # "\3-grams:"
                    try:
                        order = int(line[1:].split("-")[0])
                        self.order = max(self.order, order)
                    except ValueError:
                        order = None
                    continue
                if line.startswith("\\"):
                    order = None
                    continue
                if order is None:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                gram = tuple(parts[1].split())
                if len(gram) != order:
                    continue
                self.logp[gram] = float(parts[0])
                if len(parts) > 2:
                    self.bo[gram] = float(parts[2])
                n += 1
        if not self.logp:
            raise ValueError(f"no n-grams parsed from {path} -- is it really an ARPA file?")
        if verbose:
            print(f"  ARPA: order {self.order}, {n:,} n-grams from {path}")

    def word_logp(self, hist: tuple[str, ...], w: str) -> float:
        """log10 P(w | hist) with Katz backoff: longest matching n-gram wins, and each step down
        adds the backoff weight of the context we gave up."""
        hist = hist[-(self.order - 1):] if self.order > 1 else ()
        backoff = 0.0
        for i in range(len(hist) + 1):
            ctx = hist[i:]
            p = self.logp.get(ctx + (w,))
            if p is not None:
                return p + backoff
            backoff += self.bo.get(ctx, 0.0)
        return self.logp.get((UNK,), -7.0) + backoff

    def score(self, text: str) -> float:
        """log10 P(sentence), bracketed with <s> and </s> as lmplz trained it."""
        words = text.split()
        if not words:
            return -99.0
        hist: tuple[str, ...] = (BOS,)
        total = 0.0
        for w in words:
            total += self.word_logp(hist, w)
            hist = hist + (w,)
        total += self.word_logp(hist, EOS)
        return total


def rescore(beams, lm: ArpaLM, alpha: float, beta: float):
    """Pick the best of pyctcdecode's n-best under acoustic + alpha*LM + beta*|words|.

    `beams` are (text, last_state, text_frames, logit_score, lm_score) tuples; with no LM compiled
    in, lm_score is the acoustic score, so we use logit_score and add our own LM term.
    """
    best, best_score = None, -math.inf
    for b in beams:
        text = b[0]
        acoustic = b[3]
        s = acoustic + alpha * lm.score(text) + beta * len(text.split())
        if s > best_score:
            best, best_score = text, s
    return best if best is not None else ""
