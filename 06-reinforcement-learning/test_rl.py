"""
Tests for the RL lab: environment rules, Q-learning update, DQN loss, and the
two-robot collision rules — all checked BEFORE interpreting any training run.

Run:  python3 test_rl.py
"""

import numpy as np

from warehouse_env import SingleRobotEnv, TwoRobotEnv, MAX_STEPS
from q_learning import q_update, td_target

UP, DOWN, LEFT, RIGHT, WAIT = range(5)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def step_from(pos, action):
    env = SingleRobotEnv()
    env.set_position(pos)
    return env.step(action)


def test_environment_table():
    print("-- Task 4: single-robot environment --")
    cases = [((0, 0), UP, (0, 0), -1.0, False),
             ((1, 1), RIGHT, (1, 1), -1.0, False),
             ((2, 1), RIGHT, (2, 2), -0.1, False),
             ((2, 1), WAIT, (2, 1), -0.1, False),
             ((2, 3), RIGHT, (2, 4), 10.0, True)]
    for pos, a, exp_pos, exp_r, exp_term in cases:
        s2, r, term, trunc, _ = step_from(pos, a)
        check(f"{pos} action {a}: -> {exp_pos}, r={exp_r}, terminated={exp_term}",
              s2 == exp_pos and abs(r - exp_r) < 1e-12 and term == exp_term and not trunc)


def test_truncation():
    env = SingleRobotEnv()
    env.reset()
    for t in range(MAX_STEPS):
        _, _, term, trunc, _ = env.step(WAIT)
        if term or trunc:
            break
    check("40 waits -> truncated after exactly 40 steps, not terminated",
          trunc and not term and t == MAX_STEPS - 1)


def test_q_update_by_hand():
    print("-- Task 4: synthetic Q-learning updates --")
    # Non-terminal: y = 1 + 0.95*2 = 2.9 ; Q <- 0.4 + 0.2*(2.9-0.4) = 0.9
    Q = np.zeros((2, 5)); Q[0, 0] = 0.4; Q[1, :] = [2, 0, 0, 0, 0]
    check("non-terminal target = 2.9", abs(td_target(1, 2, False) - 2.9) < 1e-12)
    check("non-terminal update -> 0.9", abs(q_update(Q, 0, 0, 1, 1, False) - 0.9) < 1e-12)
    # Terminal r=5: y = 5 ; Q <- 0.4 + 0.2*(5-0.4) = 1.32
    Q = np.zeros((2, 5)); Q[0, 0] = 0.4; Q[1, :] = [2, 0, 0, 0, 0]
    check("terminal target = 5 (ignores next state)", td_target(5, 2, True) == 5)
    check("terminal update -> 1.32", abs(q_update(Q, 0, 0, 5, 1, True) - 1.32) < 1e-12)
    # Truncated non-terminal: still bootstraps -> same as non-terminal, 0.9
    Q = np.zeros((2, 5)); Q[0, 0] = 0.4; Q[1, :] = [2, 0, 0, 0, 0]
    check("truncated (terminated=False) update -> 0.9, bootstraps",
          abs(q_update(Q, 0, 0, 1, 1, terminated=False) - 0.9) < 1e-12)


def test_dqn_loss():
    print("-- Task 6: DQN loss on hand-computable targets --")
    try:
        import torch
        from dqn import QNet, dqn_loss
    except ImportError:
        print("[SKIP] torch not installed")
        return

    class Fixed(torch.nn.Module):       # target net: next max value is 2 everywhere
        def forward(self, x):
            return torch.tensor([[2.0, 0, 0, 0, 0]]).repeat(x.shape[0], 1)

    torch.manual_seed(0)
    online = QNet(25, 5)
    s = torch.eye(25)[:3]
    a = torch.tensor([1, 3, 4])
    r = torch.tensor([1.0, 5.0, 1.0])
    d = torch.tensor([0.0, 1.0, 0.0])          # non-terminal, terminal, truncated
    loss, q_sa, y = dqn_loss(online, Fixed(), s, a, r, s, d)
    check("targets = [2.9, 5.0, 2.9]", torch.allclose(y, torch.tensor([2.9, 5.0, 2.9])))
    check("loss uses Q(s, recorded action)",
          torch.allclose(q_sa, online(s)[torch.arange(3), a]))
    check("target carries no gradient", not y.requires_grad)
    loss.backward()
    check("gradient flows into the online network",
          online.net[0].weight.grad is not None)


def test_two_robot_rules():
    print("-- Task 7: two-robot movement and collision rules --")
    env = TwoRobotEnv()
    J = TwoRobotEnv.encode_action
    cases = [  # (old A, old B), A act, B act, expected
        (((2, 1), (2, 3)), RIGHT, LEFT, "same-cell"),
        (((2, 1), (2, 2)), RIGHT, LEFT, "swap"),
        (((2, 1), (2, 2)), RIGHT, RIGHT, ((2, 2), (2, 3))),
        (((2, 1), (2, 2)), UP, WAIT, ((1, 1), (2, 2))),
    ]
    for state, aA, aB, expected in cases:
        env.set_state(state)
        s2, r, term, trunc, info = env.step(J(aA, aB))
        if isinstance(expected, str):
            check(f"{state} A={aA} B={aB}: {expected} collision, r=-10, terminated, old positions kept",
                  info["collision"] == expected and r == -10 and term and s2 == state)
        else:
            check(f"{state} A={aA} B={aB}: moves to {expected}, r=-0.1, no collision",
                  s2 == expected and abs(r + 0.1) < 1e-12 and not term)

    # Entering a WAITING robot's cell is a collision.
    env.set_state(((2, 1), (2, 2)))
    _, r, term, _, info = env.step(J(RIGHT, WAIT))
    check("entering a waiting robot's cell -> collision", info["collision"] and term)

    # Robot already at its goal stays and gets no repeated arrival reward.
    env.set_state(((2, 3), (1, 0)))
    s2, r, term, _, _ = env.step(J(RIGHT, WAIT))
    check("A arrives: reward -0.1 + 5 = 4.9", s2[0] == (2, 4) and abs(r - 4.9) < 1e-12)
    s3, r2, _, _, _ = env.step(J(LEFT, WAIT))
    check("A at goal ignores its action and stays", s3[0] == (2, 4))
    check("no repeated arrival reward (r=-0.1)", abs(r2 + 0.1) < 1e-12)
    # Both arrive: B moves to its goal (2,0) -> -0.1 + 5 + 5 = 9.9, terminate.
    env.set_state(((2, 4), (2, 1)))
    _, r3, term3, _, _ = env.step(J(WAIT, LEFT))
    check("second arrival completes task: r = 9.9, terminated", abs(r3 - 9.9) < 1e-12 and term3)
    # Invalid move costs -1 (b counts robots not at goal).
    env.set_state(((0, 0), (4, 4)))
    _, r4, _, _, _ = env.step(J(UP, DOWN))
    check("two invalid moves: r = -0.1 - 2 = -2.1", abs(r4 + 2.1) < 1e-12)
    check("joint action encode/decode round-trips",
          all(TwoRobotEnv.decode_action(J(x, y)) == (x, y) for x in range(5) for y in range(5)))


if __name__ == "__main__":
    test_environment_table()
    test_truncation()
    test_q_update_by_hand()
    test_dqn_loss()
    test_two_robot_rules()
    print("\nAll tests passed.")
