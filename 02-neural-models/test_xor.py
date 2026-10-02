"""
Validation tests for the XOR experiments — properties that must hold if the
experiment is implemented correctly.

Run with:  python3 test_xor.py
"""

import torch
from xor_experiments import XORNet, train, X, Y, ThreeClassNet


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_random_init_learns_xor():
    torch.manual_seed(0)
    diag = train(XORNet("tanh"), steps=5000, lr=0.5)
    check("tanh net learns all four XOR labels", diag["all_correct"])
    check("final loss is small (<0.05)", diag["final_loss"] < 0.05)


def test_zero_init_fails():
    net = XORNet("sigmoid")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
    diag = train(net, steps=5000, lr=0.5)
    check("zero-initialised net does NOT solve XOR (symmetry)",
          not diag["all_correct"])


def test_loss_decreases():
    torch.manual_seed(1)
    diag = train(XORNet("sigmoid"), steps=5000, lr=0.5)
    check("loss decreased from initial to final",
          diag["final_loss"] < diag["initial_loss"])


def test_softmax_sums_to_one():
    torch.manual_seed(0)
    net = ThreeClassNet()
    with torch.no_grad():
        probs = torch.softmax(net(X), dim=1)
    sums = probs.sum(dim=1)
    check("every softmax row sums to 1",
          torch.allclose(sums, torch.ones(4), atol=1e-5))


def test_gradients_are_populated():
    torch.manual_seed(0)
    net = XORNet("sigmoid")
    loss = torch.nn.BCEWithLogitsLoss()(net(X), Y)
    loss.backward()
    check("fc1.weight.grad exists after backward()", net.fc1.weight.grad is not None)
    check("gradient is not all zero at a random init",
          net.fc1.weight.grad.abs().sum().item() > 0)


if __name__ == "__main__":
    test_random_init_learns_xor()
    test_zero_init_fails()
    test_loss_decreases()
    test_softmax_sums_to_one()
    test_gradients_are_populated()
    print("\nAll tests passed.")
