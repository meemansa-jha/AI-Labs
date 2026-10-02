"""
Validation tests for the autoregressive language models.
Run with:  python3 test_lm.py
"""

import random
from language_model import (FirstOrderLM, SecondOrderLM, tokenise, CORPUS,
                            START, END)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


sentences = tokenise(CORPUS)


def test_first_order_normalised():
    m = FirstOrderLM(); m.fit(sentences)
    ok = all(abs(sum(d.values()) - 1.0) < 1e-9 for d in m.probs.values())
    check("every first-order distribution sums to 1", ok)


def test_second_order_normalised():
    m = SecondOrderLM(); m.fit(sentences)
    ok = all(abs(sum(d.values()) - 1.0) < 1e-9 for d in m.probs.values())
    check("every second-order distribution sums to 1", ok)


def test_known_probability():
    m = FirstOrderLM(); m.fit(sentences)
    # 'the' is followed by cat (x2), dog (x2), mat, rug, park across the corpus
    # starting-word counts: 6 sentences start 'the', 3 lead to cat? Let's just
    # assert the known conditional P(sat|cat) = 2/3.
    check("P(sat | cat) == 2/3", abs(m.distribution("cat")["sat"] - 2/3) < 1e-9)
    check("P(on | sat) == 1.0", abs(m.distribution("sat")["on"] - 1.0) < 1e-9)


def test_zero_probability_transition():
    m = FirstOrderLM(); m.fit(sentences)
    check("P(dog | cat) is zero (not in table)",
          "dog" not in m.distribution("cat"))


def test_generation_terminates():
    m = SecondOrderLM(); m.fit(sentences)
    rng = random.Random(0)
    s = m.generate(rng, max_len=50)
    check("second-order generation terminates within max_len", len(s) <= 50)
    check("generated tokens contain no START/END markers",
          START not in s and END not in s)


def test_second_order_more_coherent():
    # On this corpus, every second-order sentence should be a corpus sentence
    # (the data is deterministic enough), so diversity <= first-order diversity.
    fo = FirstOrderLM(); fo.fit(sentences)
    so = SecondOrderLM(); so.fit(sentences)
    rng = random.Random(7)
    fo_u = {tuple(fo.generate(rng)) for _ in range(50)}
    rng = random.Random(7)
    so_u = {tuple(so.generate(rng)) for _ in range(50)}
    check("second-order produces fewer distinct sentences (more coherent)",
          len(so_u) <= len(fo_u))


if __name__ == "__main__":
    test_first_order_normalised()
    test_second_order_normalised()
    test_known_probability()
    test_zero_probability_transition()
    test_generation_terminates()
    test_second_order_more_coherent()
    print("\nAll tests passed.")
