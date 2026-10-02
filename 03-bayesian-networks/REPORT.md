# Laboratory Report — Bayesian Networks and Autoregressive Language Models

**Artificial Intelligence**

| File | Purpose |
|------|---------|
| `language_model.py` | first- and second-order models, CPTs, normalisation tests, generation, comparison |
| `test_lm.py` | property tests (normalisation, known probabilities, termination, coherence) |
| `experiment_transcript.txt` | full captured output |

```bash
python3 test_lm.py           # validation tests
python3 language_model.py    # all parts
```

Corpus (Part III), lower-cased and wrapped in `<START>` / `<END>`:

```
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
```

---

## Question 1 — Why is the chain-rule decomposition useful for generating text?

The chain rule factorises the joint probability of a sequence into a product of next-token conditionals,
`P(X₁,…,X_T) = P(X₁) · Π_{t≥2} P(X_t | X₁,…,X_{t−1})`.
This turns the hard problem of modelling a whole sentence at once into a sequence of *local* decisions: predict one token, append it, condition on the longer prefix, predict the next. Generation becomes a simple loop — sample `X₁`, then sample each `X_t` from its conditional — and we only ever need distributions over single next tokens, not over entire sentences.

## Question 2 — Independence assumption of the first-order network

The chain `X₁ → X₂ → ⋯ → X_T` encodes the **first-order Markov assumption**: each token depends only on the token immediately before it,
`P(X_t | X₁,…,X_{t−1}) = P(X_t | X_{t−1})`.
Equivalently, given `X_{t−1}`, token `X_t` is conditionally independent of all earlier tokens.

---

## Part IV & Question 3 — First-order conditional probability table

Using `P(wⱼ | wᵢ) = C(wᵢ, wⱼ) / Σ_k C(wᵢ, w_k)`:

| current word | P(next \| current) |
|---|---|
| `the` | cat 0.25, dog 0.25, mat 0.167, rug 0.167, park 0.167 |
| `cat` | sat 0.667, ran 0.333 |
| `dog` | sat 0.667, ran 0.333 |
| `sat` | on 1.0 |
| `ran` | to 1.0 |
| `on`  | the 1.0 |
| `to`  | the 1.0 |

**Zero-probability transitions.** Every pair not listed has probability 0. Examples: `P(dog | cat) = 0`, `P(mat | the) = 0` only in the sense that `the`→`mat` *is* observed (0.167) — but e.g. `P(cat | sat) = 0`, `P(sat | on) = 0`, `P(park | cat) = 0`. Any current word with a single observed successor (`sat`, `ran`, `on`, `to`) assigns probability 0 to every other word. All distributions are verified to sum to 1 (Part VII).

---

## Part V & VI — Inspecting the generated code

**Q4 — Where are transition counts stored?** In `self.counts`, a nested dict `counts[current][next] = integer count` (`counts[(prev2, prev1)][next]` for the second-order model).

**Q5 — Where is `P(X_t | X_{t−1})` computed?** In `_normalise()`, which divides each count by the row total: `probs[cur][w] = counts[cur][w] / total`.

**Q6 — How does the program choose the next word?** It supports **both**. Greedy (`most_probable`) always returns `argmax_w P(w | current)`; sampling (`sample_next`) draws from the full distribution via `random.choices(words, weights=probs)`. The difference: greedy is deterministic and repeats the single most likely continuation every time; sampling is stochastic and reproduces the *variety* present in the data.

**Q7 — What happens on a word with no observed transition?** `distribution()` returns an empty dict, and both `sample_next`/`most_probable` return `<END>`, so generation stops gracefully instead of crashing. (A real model would smooth or back off; here we terminate.)

---

## Part VII — Normalisation test

For every context the program checks `Σ_v P(v | w)`:

```
first-order:  all 11 contexts  -> total = 1.000000   (normalised? True)
second-order: all 15 contexts  -> total = 1.000000   (normalised? True)
```

**Q8 — If a total were 0.87, what would it tell you?** That the implementation is wrong — a conditional distribution must sum to 1. A total of 0.87 means probability mass is missing: likely a counting bug (missed transitions), dividing by the wrong total, or dropping entries. It is a direct, cheap invariant that catches such bugs without any reference output.

---

## Part VIII & Question 9 — Next-word prediction

```
P(next | 'the') = {cat 0.25, dog 0.25, mat 0.167, rug 0.167, park 0.167}  argmax = cat
P(next | 'cat') = {sat 0.667, ran 0.333}                                   argmax = sat
P(next | 'sat') = {on 1.0}                                                 argmax = on
P(next | 'on')  = {the 1.0}                                                argmax = the
P(next | 'to')  = {the 1.0}                                                argmax = the
```

**Q9 — Are the most-probable predictions always what you'd personally expect?** Not always. After `the`, a human might expect `cat`, but the model's argmax is `cat` only because of a count tie broken arbitrarily — `dog` is equally likely (0.25). The model reflects *corpus frequencies*, not linguistic intuition: it has no notion of meaning, only of what followed what in the data. A probability model answers "what is statistically likely here," which can diverge from "what a competent speaker expects."

---

## Part IX / X & Question 10 — Generation: sampling vs greedy

Greedy generation collapses into a loop (it always picks the single most probable next word):

```
greedy: the cat sat on the cat sat on the cat sat on ...   (identical every run)
```

Sampling produces varied sentences, some sensible, some rambling:

```
sampled: the dog sat on the mat
         the cat ran to the rug
         the dog ran to the dog sat on the park
```

**Q10 — Which mode produces more variation, and why?** Sampling. Greedy is deterministic — same seed or not, it always follows the argmax chain and so produces one fixed (here, looping) sentence. Sampling draws from the whole conditional distribution, so wherever a word has several possible successors it can branch, reproducing the diversity of the training data. The cost is coherence: a first-order model forgets everything but the last word, so sampling can wander (`the dog sat on the dog sat on …`).

---

## Part XI / XII & Question 11 — Second-order model

Model: `P(X_t | X_{t−2}, X_{t−1})`, graph `X_{t−2} → X_t ← X_{t−1}`.

```
P(next | (<START>, the)) = {cat 0.5, dog 0.5}
P(next | (the, cat))     = {sat 0.667, ran 0.333}
P(next | (on, the))      = {mat 0.5, rug 0.5}
P(next | (cat, sat))     = {on 1.0}
```

**Q11 — How does the second-order model differ?**
1. **Graph structure:** each node now has two parents (`X_{t−2}` and `X_{t−1}`) instead of one.
2. **Conditional probability table:** keyed by *pairs* of words, so it is much larger (one distribution per observed 2-word context).
3. **Context for prediction:** two words of history instead of one — more information about what comes next.
4. **Data needed:** far more. The number of possible contexts grows from `V` to `V²`, so with a fixed corpus each context is seen fewer times and many never occur (sparsity).

Second-order sampled sentences are dramatically more coherent — **all 20 are grammatical, corpus-like sentences**:

```
the cat sat on the rug
the dog ran to the park
the dog sat on the mat
...
```

---

## Part XIII & Question 12 — Comparing the two models

| Measure | First-order | Second-order |
|---|:--:|:--:|
| Distinct nonzero parameters | 17 | 19 |
| Number of contexts | 11 | 15 |
| Unique sentences per 50 generated | **20** | **6** |

The first-order model is more *diverse* (20 unique) but frequently incoherent (loops, run-ons); the second-order model is more *coherent* (every sentence is corpus-like) but less diverse (6 unique), because its contexts are specific enough that most have only one or two continuations.

**Q12 — Why does more context help prediction yet hurt estimation?** More context (a longer history) lets the model condition on a more specific situation, so its predictions can be sharper and more coherent. But the number of distinct contexts grows combinatorially (`V` → `V²` → `Vⁿ`), so with limited data each context is observed only a handful of times — estimates become noisy and many contexts are never seen at all (zero-probability contexts). This is the bias–variance / sparsity trade-off: the CPT grows so large that there is not enough data to estimate it reliably.

---

## Part XIV / Important Distinction

A modern autoregressive LLM models the same object — `P(X_t | X₁,…,X_{t−1})` — and generates by the same "predict next token conditional on previous tokens" loop. The difference is purely in *how the conditional is represented and learned*:

| | Simple BN model | Modern autoregressive LM |
|---|---|---|
| Representation | explicit CPTs | neural network |
| Context | fixed window (1–2 words) | large learned context |
| Parameters | explicit counts/probabilities | learned weights |
| Learning | counting | gradient-based training |
| Generation | sampling | sampling |

The probabilistic question `P(next token | previous tokens)` is identical; only the estimation machinery changes.

---

## Part XV & Question 13 — Approach A vs Approach B

**Q13 — Why is Approach B ("implement P(X_t | X_{t−1}) from transition counts, with sampling generation") preferable to Approach A ("write a Python language model for me")?**
Because Approach B states the *behaviour* the system must have before any code is written. It fixes (i) the intended behaviour (estimate a specific conditional, generate by sampling), (ii) the representation (counts → probabilities), (iii) the criteria for validating the output (distributions sum to 1, known conditionals like `P(sat|cat)=2/3`), (iv) testable probabilistic invariants, and (v) a clear line between the *model* (the probabilistic specification) and the *implementation* (the particular Python). Approach A leaves all of this to the LLM's guesswork, so you cannot tell whether the result implements *your* model or merely *a* program that runs.

---

## Question 14 — What did thinking of the LM as a Bayesian network add?

Viewing the language model as a Bayesian network gave: (i) **a representation of dependencies** — the arrows make explicit which previous tokens each token depends on; (ii) **a factorisation of the joint distribution** — the chain rule turns `P(sentence)` into a product of local conditionals we can actually estimate; (iii) **a way to interpret conditional probabilities** — each CPT row is a genuine distribution over next words; (iv) **a principled generation method** — ancestral sampling down the chain; (v) **a way to reason about independence assumptions** — first- vs second-order is literally a change of graph structure; (vi) **a way to understand the effect of context** — adding a parent edge is adding history; and (vii) **a way to test an implementation against its specification** — the normalisation invariant `Σ_v P(v|w)=1` is a property the correct model *must* satisfy, which is exactly how the tests catch bugs.

---

## Reflection on the LLM and validation

The LLM produced the counting/normalising/sampling scaffold quickly. Human verification was essential in two places that *ran fine but were wrong in spirit*: an early draft normalised over the wrong denominator (per-word global frequency instead of per-context), which still produced numbers but **failed the `Σ_v P(v|w)=1` test**; and the first sampler used `max()` (greedy) while claiming to "generate text," which was caught by noticing zero variation across runs. Both bugs were invisible from "the program runs" and only surfaced through the probabilistic invariants and the diversity check — the lab's central lesson that a running probabilistic program must be validated against properties the model is required to satisfy.
