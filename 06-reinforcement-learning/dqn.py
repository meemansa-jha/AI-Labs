"""
Parts II and III — Deep Q-Networks (PyTorch) for one robot and for two robots.

Shared DQN settings (from the lab):
  replay capacity 20,000, batch 64, target net initialised from online net,
  500 steps collected before learning, one optimiser update every 4 env steps,
  target sync every 250 optimiser updates, Adam lr 1e-3, mean-squared TD error,
  detached targets under torch.no_grad(), reward-only target on termination,
  bootstrap on truncation, epsilon linear 1 -> 0.05 over the first 80% of steps.

  One robot : 25 one-hot inputs -> 64 ReLU -> 64 ReLU -> 5 linear outputs,
              12,000 env steps, seed 7.
  Two robots: 50 inputs (two one-hots) -> 64 -> 64 -> 25 joint-action outputs,
              60,000 env steps, seed 7.

Run:  python3 dqn.py            (both parts)
      python3 dqn.py --single   /  python3 dqn.py --two  [--seed N --steps N]
"""

from __future__ import annotations

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from warehouse_env import (SingleRobotEnv, TwoRobotEnv, ACTIONS, GAMMA,
                           one_hot, two_robot_input, render)


# ---------------------------------------------------------------------------
# Network, replay buffer, loss
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, n_out))   # Q-values, no softmax

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000) -> None:
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, terminated):
        self.buf.append((s, a, r, s2, float(terminated)))

    def sample(self, batch: int, rng: random.Random):
        items = rng.sample(self.buf, batch)          # uniform random minibatch
        s, a, r, s2, d = zip(*items)
        return (torch.tensor(np.array(s)), torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32), torch.tensor(np.array(s2)),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def dqn_loss(online: QNet, target: QNet, s, a, r, s2, d, gamma: float = GAMMA):
    """L = mean (Q_theta(s_i, a_i) - y_i)^2,  y_i = r_i + gamma (1 - d_i) max Q_target(s'_i, .).

    The recorded action a_i selects which output of the online network is
    compared with the target (gather).  d_i = 1 only for true termination, so
    truncated transitions still bootstrap.  Targets carry no gradient.
    """
    q_sa = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        y = r + gamma * (1.0 - d) * target(s2).max(dim=1).values
    return nn.functional.mse_loss(q_sa, y), q_sa, y


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------
def train_dqn(env, encode, n_in, n_out, total_steps, seed=7, warmup=500,
              train_every=4, sync_every=250, batch=64, lr=1e-3, log_every=None):
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    online, target = QNet(n_in, n_out), QNet(n_in, n_out)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer()

    decay_steps = int(0.8 * total_steps)
    updates = 0
    ep_return, ep_returns, ep_outcomes = 0.0, [], []
    state = env.reset()
    for t in range(total_steps):
        eps = max(0.05, 1.0 - (1.0 - 0.05) * t / decay_steps)
        x = encode(state)
        if rng.random() < eps:
            a = rng.randrange(n_out)
        else:
            with torch.no_grad():
                q = online(torch.tensor(x))
            best = torch.nonzero(q == q.max()).flatten().tolist()
            a = rng.choice(best)
        nxt, r, term, trunc, info = env.step(a)
        buffer.push(x, a, r, encode(nxt), term)      # next state before reset
        ep_return += r
        state = nxt
        if term or trunc:
            outcome = ("collision" if info.get("collision") else "success") if term else "timeout"
            ep_returns.append(ep_return)
            ep_outcomes.append(outcome)
            ep_return = 0.0
            state = env.reset()

        if t >= warmup and t % train_every == 0:
            loss, _, _ = dqn_loss(online, target, *buffer.sample(batch, rng))
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            if updates % sync_every == 0:
                target.load_state_dict(online.state_dict())   # target sync
        if log_every and (t + 1) % log_every == 0:
            recent = ep_outcomes[-50:]
            print(f"  step {t+1:>6}  eps={eps:.2f}  episodes={len(ep_outcomes)}  "
                  f"last-50 success={recent.count('success')/max(len(recent),1):.2f}  "
                  f"collision={recent.count('collision')/max(len(recent),1):.2f}")
    return online, {"returns": ep_returns, "outcomes": ep_outcomes, "updates": updates}


def greedy_single(net: QNet, start=None):
    env = SingleRobotEnv()
    pos = env.reset(start)
    path, acts, invalid = [pos], [], 0
    while True:
        with torch.no_grad():
            a = int(torch.argmax(net(torch.tensor(one_hot(pos)))))   # fixed tie-break
        pos, r, term, trunc, info = env.step(a)
        path.append(pos); acts.append(ACTIONS[a]); invalid += info["invalid"]
        if term or trunc:
            return {"success": term, "timeout": trunc, "steps": len(acts),
                    "path": path, "actions": acts, "invalid": invalid}


def greedy_two(net: QNet, start=None, verbose=True):
    env = TwoRobotEnv()
    state = env.reset(start)
    rows, invalid, outcome = [(0, state, "-", "-")], 0, None
    while True:
        with torch.no_grad():
            j = int(torch.argmax(net(torch.tensor(two_robot_input(state)))))
        aA, aB = TwoRobotEnv.decode_action(j)
        state, r, term, trunc, info = env.step(j)
        invalid += info["invalid"]
        rows.append((env.t, state, ACTIONS[aA], ACTIONS[aB]))
        if term:
            outcome = "collision" if info["collision"] else "success"
            break
        if trunc:
            outcome = "timeout"
            break
    if verbose:
        print(f"  {'t':>2}  {'A action':<8} {'B action':<8} {'A pos':<8} {'B pos':<8}")
        for t, (pA, pB), aA, aB in rows:
            print(f"  {t:>2}  {aA:<8} {aB:<8} {str(pA):<8} {str(pB):<8}")
    return {"outcome": outcome, "steps": len(rows) - 1, "invalid": invalid, "rows": rows}


def plot_two_robot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from warehouse_env import OBSTACLES
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for (r, c) in OBSTACLES:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#666"))
    for k, colour, off in ((0, "#1f5fa8", -0.12), (1, "#c0392b", 0.12)):
        pts = [row[1][k] for row in rows]
        xs = [p[1] + off for p in pts]
        ys = [p[0] + off for p in pts]
        ax.plot(xs, ys, "-o", color=colour, label="Robot " + "AB"[k])
        for (t, *_), x, y in zip(rows, xs, ys):
            ax.annotate(str(t), (x, y), fontsize=7, color=colour,
                        xytext=(3, 3), textcoords="offset points")
    ax.set(xlim=(-.5, 4.5), ylim=(4.5, -.5), xticks=range(5), yticks=range(5),
           xlabel="column", ylabel="row", title="Greedy joint trajectory (labels = time)")
    ax.grid(True, lw=0.3)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


# ---------------------------------------------------------------------------
def run_single(seed=7, steps=12_000):
    print(f"\nDQN, ONE ROBOT (seed {seed}, {steps} env steps)")
    net, log = train_dqn(SingleRobotEnv(), one_hot, 25, 5, steps, seed=seed,
                         log_every=3000)
    print(f"  optimiser updates: {log['updates']}, training episodes: {len(log['outcomes'])}")
    ev = greedy_single(net)
    print(f"  greedy from (2,0): success={ev['success']} steps={ev['steps']} "
          f"invalid={ev['invalid']} actions={ev['actions']}")
    print(render({"R": ev["path"]}))
    ev2 = greedy_single(net, start=(0, 0))
    print(f"  greedy from (0,0) (no retraining): success={ev2['success']} "
          f"steps={ev2['steps']} invalid={ev2['invalid']} path={ev2['path']}")
    return net


def run_two(seed=7, steps=60_000):
    print(f"\nDQN, TWO ROBOTS (seed {seed}, {steps} env steps)")
    net, log = train_dqn(TwoRobotEnv(), two_robot_input, 50, 25, steps, seed=seed,
                         log_every=10_000)
    print(f"  optimiser updates: {log['updates']}, training episodes: {len(log['outcomes'])}")
    print("  greedy joint rollout from A=(2,0), B=(2,4):")
    ev = greedy_two(net)
    print(f"  outcome={ev['outcome']} steps={ev['steps']} invalid={ev['invalid']}")
    plot_two_robot(ev["rows"], f"two_robot_trajectory_seed{seed}_{steps//1000}k.png")
    print("  additional start A=(0,0), B=(4,4) (no retraining):")
    ev2 = greedy_two(net, start=((0, 0), (4, 4)))
    print(f"  outcome={ev2['outcome']} steps={ev2['steps']} invalid={ev2['invalid']}")
    return net, ev


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true")
    p.add_argument("--two", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args()
    both = not (args.single or args.two)
    if args.single or both:
        run_single(args.seed, args.steps or 12_000)
    if args.two or both:
        if args.steps:
            run_two(args.seed, args.steps)
        else:
            run_two(args.seed, 60_000)    # the lab's starting budget
            run_two(args.seed, 120_000)   # retry with a larger budget (see REPORT.md)
