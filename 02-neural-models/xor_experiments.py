"""
Neural Models laboratory — XOR learning, depth vs nonlinearity, backprop,
weight-symmetry, and an activation comparison.

Scenario: two redundant binary safety sensors; raise a 'disagreement' warning
exactly when the sensors differ.  That decision rule is XOR.

    x1 x2 | y
     0  0 | 0
     0  1 | 1
     1  0 | 1
     1  1 | 0

Design (Task 2): a 2 -> 2 -> 1 network, nonlinear hidden activation, a single
logit output trained with BCEWithLogitsLoss (numerically stable sigmoid+BCE),
full-batch gradient descent.

Run:  python3 xor_experiments.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data: the four XOR examples, stated explicitly.
# ---------------------------------------------------------------------------
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class XORNet(nn.Module):
    """2 inputs -> 2 hidden units (nonlinear) -> 1 logit output."""

    def __init__(self, activation: str = "sigmoid") -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)   # W(1) is 2x2, b(1) is length 2
        self.fc2 = nn.Linear(2, 1)   # W(2) is 1x2, b(2) is length 1
        acts = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU()}
        if activation not in acts:
            raise ValueError(f"unknown activation {activation!r}")
        self.act = acts[activation]
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.fc1(x)        # pre-activation of hidden layer
        h1 = self.act(a1)       # hidden activation
        logit = self.fc2(h1)    # output logit (no sigmoid here; BCEWithLogits adds it)
        return logit


def train(net: nn.Module, steps: int = 5000, lr: float = 0.5,
          record_early_grad_at: int = 1):
    """Full-batch training with BCEWithLogitsLoss (stable sigmoid + BCE).

    Returns a dict of diagnostics rather than just a final number, so we can
    collect evidence that learning really happened.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    diag = {"initial_loss": None, "final_loss": None, "early_grad_norm": None}

    for step in range(steps):
        opt.zero_grad()
        logits = net(X)            # forward pass
        loss = loss_fn(logits, Y)  # scalar loss
        loss.backward()            # reverse-mode autodiff (backpropagation)

        if step == 0:
            diag["initial_loss"] = loss.item()
        if step == record_early_grad_at:
            # Euclidean norm of the first-layer weight gradient at an early step.
            diag["early_grad_norm"] = net.fc1.weight.grad.norm().item()

        opt.step()                 # optimiser updates parameters

    # Final evaluation.
    with torch.no_grad():
        logits = net(X)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        diag["final_loss"] = loss_fn(logits, Y).item()
        diag["probs"] = probs.squeeze().tolist()
        diag["preds"] = preds.squeeze().tolist()
        diag["all_correct"] = bool(torch.equal(preds, Y))
    return diag


# ---------------------------------------------------------------------------
# Task 4 Part A + B — basic learning check and a backpropagation inspection.
# ---------------------------------------------------------------------------
def part_a_b() -> None:
    print("=" * 70)
    print("TASK 4 PART A/B — basic learning check + backprop inspection")
    print("=" * 70)
    torch.manual_seed(0)
    net = XORNet("sigmoid")
    diag = train(net, steps=5000, lr=0.5)
    print(f"initial loss : {diag['initial_loss']:.4f}")
    print(f"final loss   : {diag['final_loss']:.4f}")
    print("input -> probability -> predicted label (target):")
    for (x, p, pred, t) in zip(X.tolist(), diag["probs"], diag["preds"],
                               Y.squeeze().tolist()):
        print(f"  {x} -> {p:.4f} -> {int(pred)}  (target {int(t)})")
    print(f"all four correct? {diag['all_correct']}")

    # One more forward/backward to expose a gradient tensor (Part B).
    net.zero_grad()
    loss = nn.BCEWithLogitsLoss()(net(X), Y)
    loss.backward()
    print("\nfc1.weight.grad  (this IS dL/dW^(1), averaged over the 4 examples):")
    print(net.fc1.weight.grad)
    print("Because BCEWithLogitsLoss uses the MEAN over the 4 examples, the")
    print("gradient of that mean is the average of the four per-example gradients")
    print("(the derivative is linear in the sum, and 1/4 factors through).")


# ---------------------------------------------------------------------------
# Task 4 Part C — the symmetry experiment (all weights zero).
# ---------------------------------------------------------------------------
def part_c_symmetry() -> None:
    print("\n" + "=" * 70)
    print("TASK 4 PART C — symmetry experiment (all weights initialised to 0)")
    print("=" * 70)
    net = XORNet("sigmoid")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    print("Hidden-layer weight matrix rows over the first few steps:")
    for step in range(5):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        opt.step()
        W = net.fc1.weight.detach()
        same = torch.allclose(W[0], W[1])
        print(f"  step {step}: row0={W[0].tolist()}  row1={W[1].tolist()}  "
              f"rows identical? {same}")

    diag = train(XORNet_zeroed(), steps=5000, lr=0.5)
    print(f"\nAfter full training from zero init: all four correct? "
          f"{diag['all_correct']}  (final loss {diag['final_loss']:.4f})")
    print("Explanation: identical hidden units compute the same output and")
    print("receive the same gradient, so they update identically and stay")
    print("identical forever. The two units never differentiate, so the network")
    print("cannot form the two distinct half-plane features XOR needs.")


def XORNet_zeroed() -> nn.Module:
    net = XORNet("sigmoid")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    return net


# ---------------------------------------------------------------------------
# Task 4 Part D — activation experiment (sigmoid / tanh / relu).
# ---------------------------------------------------------------------------
def part_d_activations() -> None:
    print("\n" + "=" * 70)
    print("TASK 4 PART D — activation experiment")
    print("=" * 70)
    header = f"{'activation':<10}{'final loss':<14}{'4/4 correct?':<14}{'early |grad W1|':<16}"
    print(header)
    print("-" * len(header))
    results = {}
    for act in ("sigmoid", "tanh", "relu"):
        torch.manual_seed(0)     # same init stream, only the activation changes
        net = XORNet(act)
        diag = train(net, steps=5000, lr=0.5)
        results[act] = diag
        print(f"{act:<10}{diag['final_loss']:<14.4f}"
              f"{str(diag['all_correct']):<14}{diag['early_grad_norm']:<16.4f}")
    return results


# ---------------------------------------------------------------------------
# Task 5 — three-class extension (softmax + cross-entropy).
# ---------------------------------------------------------------------------
class ThreeClassNet(nn.Module):
    """2 -> 2 (tanh) -> 3 logits.  Classes: 0=(0,0), 1=disagree, 2=(1,1)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)   # THREE logits now
        self.act = nn.Tanh()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def task5_three_class() -> None:
    print("\n" + "=" * 70)
    print("TASK 5 — three-class extension (softmax + cross-entropy)")
    print("=" * 70)
    # Labels as class indices: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->2.
    y_idx = torch.tensor([0, 1, 1, 2])
    torch.manual_seed(0)
    net = ThreeClassNet()
    loss_fn = nn.CrossEntropyLoss()     # = softmax + NLL, logit gradient is p - y
    opt = torch.optim.SGD(net.parameters(), lr=0.5)

    for _ in range(6000):
        opt.zero_grad()
        loss = loss_fn(net(X), y_idx)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = net(X)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
    print(f"final loss: {loss.item():.4f}")
    print(f"final fc2 weight shape (should be 3x2): {tuple(net.fc2.weight.shape)}")
    print(f"logits per example: {logits.shape[1]}")
    print("\ninput -> softmax probabilities -> predicted class (target):")
    for x, p, pr, t in zip(X.tolist(), probs.tolist(), preds.tolist(),
                           y_idx.tolist()):
        p = [round(v, 3) for v in p]
        print(f"  {x} -> {p} -> class {pr}  (target {t})")

    # Verify probabilities sum to 1 for one example.
    one = probs[1]
    print(f"\nexample (0,1) softmax vector = {one.tolist()}")
    print(f"sum of components = {one.sum().item():.6f}  (should be ~1.0)")

    # Shift-invariance diagnostic: add 100 to all logits before softmax.
    shifted = torch.softmax(logits[1] + 100.0, dim=0)
    print(f"softmax after adding 100 to all logits = {shifted.tolist()}")
    print(f"max abs difference from original = "
          f"{(shifted - one).abs().max().item():.2e}  (≈ roundoff)")
    print("Softmax is shift-invariant because exp(a_i + c)/sum_j exp(a_j + c)")
    print("= exp(a_i)exp(c) / (exp(c) sum_j exp(a_j)); the exp(c) cancels.")
    print("Stable implementations subtract max logit to keep exp() from overflowing.")


def main() -> None:
    part_a_b()
    part_c_symmetry()
    part_d_activations()
    task5_three_class()


if __name__ == "__main__":
    main()
