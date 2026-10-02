"""
Bayesian Networks and Autoregressive Language Models.

A first-order and a second-order autoregressive language model, built from
transition counts over a tiny corpus, viewed as Bayesian networks:

    first order:   X1 -> X2 -> X3 -> ... -> XT          (Markov chain)
    second order:  (X_{t-2}, X_{t-1}) -> X_t

No ML library, no pretrained model: ordinary Python dicts and random sampling.

Run:  python3 language_model.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

START, END = "<START>", "<END>"

# Part III corpus.
CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(corpus: List[str]) -> List[List[str]]:
    """Lower-case, split on whitespace, wrap with <START>/<END>."""
    return [[START] + line.lower().split() + [END] for line in corpus]


# ---------------------------------------------------------------------------
# First-order model:  P(X_t | X_{t-1})
# ---------------------------------------------------------------------------
class FirstOrderLM:
    def __init__(self) -> None:
        # counts[current][next] = C(current, next)
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.probs: Dict[str, Dict[str, float]] = {}

    def fit(self, sentences: List[List[str]]) -> None:
        for s in sentences:
            for cur, nxt in zip(s, s[1:]):
                self.counts[cur][nxt] += 1
        self._normalise()

    def _normalise(self) -> None:
        self.probs = {}
        for cur, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[cur] = {w: c / total for w, c in nexts.items()}

    def distribution(self, current: str) -> Dict[str, float]:
        return self.probs.get(current, {})

    def most_probable(self, current: str) -> str:
        dist = self.distribution(current)
        return max(dist, key=dist.get) if dist else END

    def sample_next(self, current: str, rng: random.Random) -> str:
        dist = self.distribution(current)
        if not dist:                         # unseen context -> stop gracefully
            return END
        words, weights = zip(*dist.items())
        return rng.choices(words, weights=weights, k=1)[0]

    def generate(self, rng: random.Random, greedy: bool = False,
                 max_len: int = 20) -> List[str]:
        out, cur = [], START
        for _ in range(max_len):
            nxt = self.most_probable(cur) if greedy else self.sample_next(cur, rng)
            if nxt == END:
                break
            out.append(nxt)
            cur = nxt
        return out


# ---------------------------------------------------------------------------
# Second-order model:  P(X_t | X_{t-2}, X_{t-1})
# ---------------------------------------------------------------------------
class SecondOrderLM:
    def __init__(self) -> None:
        self.counts: Dict[Tuple[str, str], Dict[str, int]] = \
            defaultdict(lambda: defaultdict(int))
        self.probs: Dict[Tuple[str, str], Dict[str, float]] = {}

    def fit(self, sentences: List[List[str]]) -> None:
        for s in sentences:
            # Pad with an extra START so the first real word has a 2-word context.
            padded = [START] + s
            for a, b, c in zip(padded, padded[1:], padded[2:]):
                self.counts[(a, b)][c] += 1
        self._normalise()

    def _normalise(self) -> None:
        self.probs = {}
        for ctx, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[ctx] = {w: n / total for w, n in nexts.items()}

    def distribution(self, ctx: Tuple[str, str]) -> Dict[str, float]:
        return self.probs.get(ctx, {})

    def sample_next(self, ctx: Tuple[str, str], rng: random.Random) -> str:
        dist = self.distribution(ctx)
        if not dist:
            return END
        words, weights = zip(*dist.items())
        return rng.choices(words, weights=weights, k=1)[0]

    def generate(self, rng: random.Random, max_len: int = 20) -> List[str]:
        out = []
        prev2, prev1 = START, START
        for _ in range(max_len):
            nxt = self.sample_next((prev2, prev1), rng)
            if nxt == END:
                break
            out.append(nxt)
            prev2, prev1 = prev1, nxt
        return out


# ---------------------------------------------------------------------------
# Tests (Part VII) — probabilistic invariants that must hold.
# ---------------------------------------------------------------------------
def test_distributions_sum_to_one(model) -> bool:
    ok = True
    print("Normalisation check — sum_v P(v | w) should be 1.0 for every context:")
    for ctx, dist in model.probs.items():
        total = sum(dist.values())
        flag = "" if abs(total - 1.0) < 1e-9 else "   <-- NOT NORMALISED"
        print(f"  {str(ctx):<28} total = {total:.6f}{flag}")
        ok = ok and abs(total - 1.0) < 1e-9
    return ok


def main() -> None:
    sentences = tokenise(CORPUS)

    print("=" * 70)
    print("PART IV — FIRST-ORDER CONDITIONAL PROBABILITY TABLE")
    print("=" * 70)
    fo = FirstOrderLM()
    fo.fit(sentences)
    for word in ["the", "cat", "dog", "sat", "ran", "on", "to"]:
        dist = fo.distribution(word)
        pretty = {w: round(p, 3) for w, p in dist.items()}
        print(f"P(next | {word!r}) = {pretty}")

    print("\n(Question 3) zero-probability transitions: any (current, next) pair")
    print("not listed above has probability 0 — e.g. P('dog' | 'cat') = 0,")
    print("P('mat' | 'the') = 0, P('cat' | 'sat') = 0.")

    print("\n" + "=" * 70)
    print("PART VII — NORMALISATION TEST (first-order)")
    print("=" * 70)
    ok = test_distributions_sum_to_one(fo)
    print(f"All first-order distributions normalised? {ok}")

    print("\n" + "=" * 70)
    print("PART VIII — NEXT-WORD PREDICTION")
    print("=" * 70)
    for w in ["the", "cat", "dog", "sat", "on", "ran", "to"]:
        dist = fo.distribution(w)
        if dist:
            pretty = {k: round(v, 3) for k, v in dist.items()}
            print(f"P(next | {w!r}) = {pretty}   argmax = {fo.most_probable(w)!r}")

    print("\n" + "=" * 70)
    print("PART IX / X — GENERATION (sampling vs greedy)")
    print("=" * 70)
    rng = random.Random(42)
    print("\n20 sampled sentences (first-order):")
    for i in range(20):
        print(f"  {i+1:2d}. {' '.join(fo.generate(rng, greedy=False))}")

    print("\n5 greedy sentences (first-order) — note the determinism:")
    for i in range(5):
        print(f"  {i+1}. {' '.join(fo.generate(rng, greedy=True))}")

    print("\n5 sampled sentences (first-order) for the variation comparison:")
    for i in range(5):
        print(f"  {i+1}. {' '.join(fo.generate(rng, greedy=False))}")

    print("\n" + "=" * 70)
    print("PART XI / XII — SECOND-ORDER MODEL")
    print("=" * 70)
    so = SecondOrderLM()
    so.fit(sentences)
    print("Example second-order contexts:")
    for ctx in [(START, "the"), ("the", "cat"), ("the", "dog"),
                ("cat", "sat"), ("sat", "on"), ("on", "the")]:
        dist = so.distribution(ctx)
        pretty = {w: round(p, 3) for w, p in dist.items()}
        print(f"P(next | {ctx}) = {pretty}")
    print("\nNormalisation test (second-order):")
    ok2 = test_distributions_sum_to_one(so)
    print(f"All second-order distributions normalised? {ok2}")

    print("\n20 sampled sentences (second-order):")
    rng2 = random.Random(42)
    for i in range(20):
        print(f"  {i+1:2d}. {' '.join(so.generate(rng2))}")

    print("\n" + "=" * 70)
    print("PART XIII — COMPARISON (first-order vs second-order)")
    print("=" * 70)
    fo_params = sum(len(d) for d in fo.probs.values())
    so_params = sum(len(d) for d in so.probs.values())
    print(f"distinct first-order parameters (nonzero entries):  {fo_params}")
    print(f"distinct second-order parameters (nonzero entries): {so_params}")
    print(f"first-order contexts:  {len(fo.probs)}")
    print(f"second-order contexts: {len(so.probs)}")

    # Diversity: unique generated sentences out of 50.
    rng3 = random.Random(7)
    fo_sent = {tuple(fo.generate(rng3)) for _ in range(50)}
    rng4 = random.Random(7)
    so_sent = {tuple(so.generate(rng4)) for _ in range(50)}
    print(f"unique sentences / 50 (first-order):  {len(fo_sent)}")
    print(f"unique sentences / 50 (second-order): {len(so_sent)}")
    print("Second-order contexts are more specific, so each one has fewer")
    print("continuations -> more coherent but less diverse text, and more")
    print("zero-probability contexts for unseen 2-word histories.")


if __name__ == "__main__":
    main()
