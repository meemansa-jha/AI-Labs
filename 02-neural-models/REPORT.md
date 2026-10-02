# Laboratory Report — Neural Models

**Learning, Depth, Activations, and Output Layers** · Artificial Intelligence

Scenario: a device with two redundant binary sensors must raise a *disagreement*
warning exactly when the sensors differ — the XOR rule.

| File | Purpose |
|------|---------|
| `xor_experiments.py` | the 2-2-1 XOR net, symmetry/activation experiments, three-class extension |
| `test_xor.py` | property tests (learns XOR, zero-init fails, softmax sums to 1, …) |
| `experiment_transcript.txt` | full captured output |

```bash
python3 test_xor.py          # validation tests
python3 xor_experiments.py   # all experiments (Tasks 4 and 5)
```

---

## Task 1 — Understand the problem before coding

**Input / output spaces and examples.**
`X = {0,1}²`, `Y = {0,1}`.

| x₁ | x₂ | y |
|:--:|:--:|:-:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Sketch / class labels.** In the `(x₁, x₂)` plane the class-1 points are `(0,1)` and `(1,0)` (the off-diagonal), and the class-0 points are `(0,0)` and `(1,1)` (the main diagonal):

```
x2
 1 |  (0,1)=1      (1,1)=0
 0 |  (0,0)=0      (1,0)=1
   +---------------------
       0            1     x1
```

**Why one straight boundary cannot separate the classes.** A single line splits the plane into two half-planes. The two class-1 points are diagonally opposite, and so are the two class-0 points, so any straight line that puts both class-1 points on one side necessarily puts at least one class-0 point there too. XOR is **not linearly separable**.

**Prediction for a single affine map + sigmoid.** A logistic-regression-style model `σ(wᵀx + b)` has a linear decision boundary, so it cannot fit XOR. It will converge to roughly 0.5 on all four points and classify about two of them wrong — loss stalls near `ln 2 ≈ 0.693`. (This is exactly what the zero-init network shows below: final loss 0.6931.)

---

## Task 2 — Design the agent (model specification)

Baseline design: **2 inputs → 2 hidden units → 1 output.**

- **Hidden activation:** nonlinear (sigmoid / tanh / ReLU compared in Task 4D).
- **Output:** a single logit; sigmoid is applied inside `BCEWithLogitsLoss` for numerical stability.
- **Loss:** binary cross-entropy.
- **Optimisation:** full-batch SGD, a few thousand steps, fixed seed.

**1. Why is the hidden nonlinearity scientifically necessary?** A stack of affine layers composes to a single affine map (`W₂(W₁x+b₁)+b₂ = W′x+b′`), which is still linear and still cannot represent XOR. The nonlinear hidden activation is what lets the network bend the input space so the two classes become separable in the hidden representation. Depth without nonlinearity buys nothing; the *kind* of representation, not the parameter count, is what matters.

**2. Why is sigmoid + binary cross-entropy a sensible pairing?** The task is a single yes/no decision, so a sigmoid maps the output logit to a probability in `(0,1)`, and BCE is the matching (maximum-likelihood) loss for a Bernoulli target. Their gradients combine cleanly: the logit gradient is `p − y`, which is well-behaved and does not vanish the way a squared-error-on-sigmoid loss would when the unit saturates.

**3. Evidence that will count as successful learning (≥3 checks):**
(i) final loss close to 0; (ii) all four examples predicted correctly after thresholding at 0.5; (iii) the first-layer gradient is nonzero early in training (a learning signal exists); (iv) the behaviour repeats across random seeds (not a lucky init).

---

## Task 3 — LLM prompt used

> Generate minimal PyTorch code for a 2-2-1 network that learns XOR. Use the four XOR examples explicitly, a nonlinear hidden activation, and a single logit output trained with `BCEWithLogitsLoss`. Use random initialisation and full-batch SGD for a few thousand CPU steps. Set a random seed. After training, print the final loss, the four probabilities, the thresholded labels, and one parameter-gradient tensor after `backward()`. Do not change the architecture or task; explain each test in one sentence.

**Inspection before running** (where each piece lives): the **forward pass** is `net(X)` → `fc1`, activation, `fc2`; the **scalar loss** is `loss_fn(logits, Y)`; **reverse-mode AD** is `loss.backward()`; the **optimiser update** is `opt.step()`.

**Two changes made before/after execution:** (1) switched a plain `BCELoss` + manual `Sigmoid` output to `BCEWithLogitsLoss` on raw logits for numerical stability; (2) made `train()` return a diagnostics dict (initial/final loss, early gradient norm, per-example probabilities) instead of only printing a final number, so evidence could be collected rather than trusted.

---

## Task 4 — Execute, test, diagnose

### Part A — basic learning check (sigmoid, seed 0)

```
initial loss : 0.6977
final loss   : 0.0389
[0,0] -> 0.0431 -> 0  (target 0)
[0,1] -> 0.9676 -> 1  (target 1)
[1,0] -> 0.9676 -> 1  (target 1)
[1,1] -> 0.0448 -> 0  (target 0)
all four correct? True
```

Loss fell from ~0.70 to ~0.04 and all four labels are correct — the network learned XOR.

### Part B — backpropagation check

After a forward/backward pass, `fc1.weight.grad` is:

```
[[-0.0020, -0.0020],
 [ 0.0012,  0.0012]]
```

`parameter.grad` **is** `∂L/∂W⁽¹⁾`: for each hidden weight it holds the partial derivative of the scalar loss with respect to that weight, exactly what the optimiser needs. Because `BCEWithLogitsLoss` averages over the four examples, `L = (1/4)Σᵢ Lᵢ`, and differentiation is linear, so `∂L/∂W = (1/4)Σᵢ ∂Lᵢ/∂W` — the gradient is the **average** of the four per-example gradients.

### Part C — symmetry experiment (all weights = 0)

```
step 0..4: row0=[0,0] row1=[0,0]  rows identical? True  (every step)
After full training from zero init: all four correct? False  (final loss 0.6931)
```

The two rows of the hidden weight matrix stay identical and the network never learns (loss stuck at `ln 2 ≈ 0.6931`, i.e. chance). **Explanation:** with identical weights the two hidden units compute the *same* output and therefore receive the *same* gradient, so they update identically and remain identical forever. They never specialise into the two different half-plane features XOR requires. This is why random (symmetry-breaking) initialisation is essential.

### Part D — activation experiment

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W⁽¹⁾L‖₂ |
|---|:--:|:--:|:--:|
| Sigmoid | 0.0389 | **yes** | 0.0015 |
| Tanh | 0.0013 | **yes** | 0.0542 |
| ReLU | 0.4774 | **no** | 0.0016 |

**Interpretation (for this experiment, not a universal claim).** With this seed and learning rate, tanh learned fastest (lowest loss) and had the largest early first-layer gradient — tanh is zero-centred and its derivative near the origin is ≈1, so a strong signal reaches `W⁽¹⁾` immediately. Sigmoid also solved XOR but with a much smaller early gradient (its derivative peaks at only 0.25 and it is not zero-centred), so learning was slower. ReLU **failed on this run**: with only two hidden units and this init, one or both units landed in their negative (dead) region, where the derivative is exactly 0, so no gradient flowed and the unit never recovered — the tiny early gradient norm (0.0016) is the symptom. The point is not "ReLU is bad"; it is that in a 2-unit network with a single seed, ReLU's dead-unit failure mode is easy to hit, whereas sigmoid/tanh degrade gracefully. Distinguishing the two small-gradient mechanisms: a *saturated sigmoid* has a small-but-nonzero derivative (inspect the pre-activation magnitude), whereas a *dead ReLU* has an exactly-zero derivative because its pre-activation is negative (inspect the sign of the pre-activation).

---

## Task 5 — Three-class extension (softmax + cross-entropy)

Classes: 0 = both inactive `(0,0)`, 1 = disagree `(0,1)/(1,0)`, 2 = both active `(1,1)`.
Output changed from 1 logit to **3 logits**, loss changed to `CrossEntropyLoss`.

**Predictions before accepting the change (all confirmed):**
1. Final `fc2` weight shape = **3×2** ✓ (printed `(3, 2)`).
2. Logits per example = **3** ✓.
3. Softmax probabilities sum to 1 because they are `exp(aᵢ)/Σⱼ exp(aⱼ)` — normalised by construction.
4. The logit gradient is `p − y` (softmax + cross-entropy), i.e. predicted distribution minus the one-hot target.

**Results:**

```
final loss: 0.0004      logits per example: 3
[0,0] -> [0.999, 0.001, 0.000] -> class 0  (target 0)
[0,1] -> [0.000, 1.000, 0.000] -> class 1  (target 1)
[1,0] -> [0.000, 1.000, 0.000] -> class 1  (target 1)
[1,1] -> [0.000, 0.001, 0.999] -> class 2  (target 2)

softmax((0,1)) = [0.00013, 0.99963, 0.00025]   sum = 1.000000
after adding 100 to all logits: max abs diff = 2.33e-10  (roundoff only)
```

**Why `p − y`?** For softmax outputs `p` with cross-entropy loss against one-hot target `y`, the derivative of the loss w.r.t. the logits is exactly `p − y`. Intuitively the gradient says "push the probability of the true class up and all others down," and its magnitude is how wrong the current distribution is. **Shift-invariance:** `softmax(a + c) = softmax(a)` because the common factor `exp(c)` cancels in numerator and denominator — which is why stable implementations subtract the maximum logit before exponentiating (to avoid `exp` overflow without changing the result).

---

## Reflection Questions

1. **Depth vs nonlinearity.** XOR with only four points shows that *more affine depth does not add representational power* — affine layers collapse to one affine map. It is the hidden **nonlinearity**, not the number of layers, that lets the network represent a non-linearly-separable function.

2. **Evidence of a useful learning signal (not just a nonzero gradient).** The loss fell monotonically from ~0.70 to ~0.04 **and** all four predictions became correct across seeds. A nonzero gradient alone proves only that *some* signal exists; the drop in loss plus correct labels on held-meaning examples shows the signal actually drove the representation to the right place.

3. **Why zero/identical init prevents distinct features.** Identical units produce identical outputs and receive identical gradients, so they move in lockstep and can never become different. XOR needs two *different* hidden features (two different half-planes); symmetry must be broken by random initialisation for that to happen.

4. **Activation's effect on the gradient — science vs engineering.** *Scientific:* each activation contributes a different Jacobian factor to backprop — tanh ≈1 near 0 (strong signal), sigmoid ≤0.25 (weaker), ReLU exactly 0 when inactive (no signal). *Engineering observation (this experiment):* tanh converged fastest with the largest early gradient; ReLU, with only two units and this seed, hit a dead-unit failure and did not solve XOR.

5. **Why pair output layer and loss to the task.** The output activation defines what the numbers *mean* (one probability vs a distribution over classes), and the loss defines how error is measured. Sigmoid+BCE gives a clean `p−y` logit gradient for a binary target; softmax+cross-entropy does the same for one-of-K. A mismatched pair (e.g. softmax with MSE) produces ill-scaled or vanishing gradients and mis-stated probabilities.

6. **LLM productivity vs human verification.** *Productivity:* the LLM produced correct PyTorch boilerplate (module, training loop, `backward()`/`step()` ordering) in seconds. *Verification essential:* the first draft used `BCELoss` on an explicit sigmoid output, which is numerically fragile; a human had to recognise and switch to `BCEWithLogitsLoss`, and only measurement revealed ReLU's failure on this configuration.

7. **Which tests survive scale-up.** Keep: loss-decrease monitoring, final-accuracy checks, symmetry/initialisation sanity, and probability-normalisation checks — all cheap and scale-independent. Drops out: exhaustive enumeration of every input and exhaustive finite-difference gradient checks over all parameters become infeasible; at scale one uses held-out validation sets and spot-checked (random-direction) gradient checks instead.
