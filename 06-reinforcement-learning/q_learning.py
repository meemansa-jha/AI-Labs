"""
Part I — tabular Q-learning for the single-robot warehouse.

Settings (from the lab): alpha=0.2, gamma=0.95, Q initialised to zero, seed 7,
1,500 episodes, epsilon decreasing linearly from 1 to 0.05 over the first
1,200 episodes, 40-step cutoff.  Q-table shape 25x5, row index 5r+c.

Run:  python3 q_learning.py
"""

from __future__ import annotations

import numpy as np

from warehouse_env import (SingleRobotEnv, ACTIONS, GAMMA, MAX_STEPS,
                           index, render)

ALPHA = 0.2
EPISODES = 1500
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 1200
SEED = 7


def epsilon_at(episode: int) -> float:
    frac = min(episode / EPS_DECAY_EPISODES, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def epsilon_greedy(Q: np.ndarray, s: int, eps: float, rng) -> int:
    """With prob eps pick uniformly; otherwise argmax with RANDOM tie-breaking."""
    if rng.random() < eps:
        return int(rng.integers(len(ACTIONS)))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def td_target(r: float, max_next: float, terminated: bool,
              gamma: float = GAMMA) -> float:
    """y = r on true termination; r + gamma*max_a' Q(s',a') otherwise.
    Truncation is NOT termination, so it still bootstraps."""
    return r if terminated else r + gamma * max_next


def q_update(Q: np.ndarray, s: int, a: int, r: float, s_next: int,
             terminated: bool, alpha: float = ALPHA) -> float:
    """One Q-learning update on the entry for the action taken. Returns new value."""
    y = td_target(r, float(Q[s_next].max()), terminated)
    Q[s, a] += alpha * (y - Q[s, a])
    return Q[s, a]


def train(seed: int = SEED, episodes: int = EPISODES, explore: bool = True):
    rng = np.random.default_rng(seed)
    env = SingleRobotEnv()
    Q = np.zeros((25, len(ACTIONS)))
    log = {"return": [], "success": [], "epsilon": [], "invalid": []}
    for ep in range(episodes):
        eps = epsilon_at(ep) if explore else 0.0
        s = index(env.reset())
        ret, invalid, success = 0.0, 0, False
        while True:
            a = epsilon_greedy(Q, s, eps, rng)
            pos, r, term, trunc, info = env.step(a)
            s2 = index(pos)                       # next state BEFORE any reset
            q_update(Q, s, a, r, s2, term)
            ret += r
            invalid += info["invalid"]
            s = s2
            if term or trunc:
                success = term
                break
        log["return"].append(ret)
        log["success"].append(success)
        log["epsilon"].append(eps)
        log["invalid"].append(invalid)
    return Q, log


def greedy_rollout(Q: np.ndarray, start=None):
    """Evaluation: epsilon=0, no updates, fixed tie-break (lowest action code)."""
    env = SingleRobotEnv()
    pos = env.reset(start)
    path, actions, invalid = [pos], [], 0
    while True:
        a = int(np.argmax(Q[index(pos)]))     # np.argmax: first max -> fixed rule
        pos, r, term, trunc, info = env.step(a)
        path.append(pos)
        actions.append(ACTIONS[a])
        invalid += info["invalid"]
        if term or trunc:
            return {"success": term, "timeout": trunc, "steps": len(actions),
                    "path": path, "actions": actions, "invalid": invalid}


def random_policy_eval(runs: int = 100, seed: int = SEED):
    rng = np.random.default_rng(seed)
    env = SingleRobotEnv()
    succ, timeouts, steps, invalid = 0, 0, [], 0
    for _ in range(runs):
        env.reset()
        t = 0
        while True:
            _, _, term, trunc, info = env.step(int(rng.integers(5)))
            t += 1
            invalid += info["invalid"]
            if term:
                succ += 1
                steps.append(t)
                break
            if trunc:
                timeouts += 1
                break
    return {"success_frac": succ / runs, "timeout_frac": timeouts / runs,
            "mean_steps_success": float(np.mean(steps)) if steps else None,
            "mean_invalid_per_run": invalid / runs}


def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot(log, path="qlearning_training.png", window=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(log["return"], color="#bbb", lw=0.6, label="per episode")
    axes[0].plot(range(window - 1, len(log["return"])),
                 moving_average(log["return"], window), color="#1f5fa8",
                 label=f"{window}-episode moving average")
    axes[0].set(title="Training return", xlabel="episode")
    axes[0].legend(fontsize=8)
    axes[1].plot(range(window - 1, len(log["success"])),
                 moving_average(log["success"], window), color="#2a8a4a")
    axes[1].set(title=f"Success rate ({window}-ep moving avg)", xlabel="episode",
                ylim=(0, 1.05))
    axes[2].plot(log["epsilon"], color="#a85a1f")
    axes[2].set(title="Epsilon", xlabel="episode")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    Q, log = train()
    plot(log)
    last = slice(-100, None)
    print("Q-LEARNING (seed 7, 1500 episodes)")
    print(f"success rate, last 100 training episodes: {np.mean(log['success'][last]):.2f}")
    print(f"mean return,  last 100 training episodes: {np.mean(log['return'][last]):.2f}")
    print(f"first training episode reaching the goal: {log['success'].index(True)}")

    ev = greedy_rollout(Q)
    print("\nGreedy evaluation from (2,0), epsilon=0, no updates:")
    print(f"  success={ev['success']} steps={ev['steps']} invalid={ev['invalid']} "
          f"timeout={ev['timeout']}")
    print(f"  actions: {ev['actions']}")
    print(f"  positions: {ev['path']}")
    print(render({"R": ev["path"]}))

    rnd = random_policy_eval()
    print("\nUniformly random policy, 100 episodes:")
    print(f"  {rnd}")

    print("\nAdditional start (0,0), no retraining:")
    ev2 = greedy_rollout(Q, start=(0, 0))
    print(f"  success={ev2['success']} steps={ev2['steps']} invalid={ev2['invalid']} "
          f"path={ev2['path']}")

    print("\nOptional: no exploration (epsilon=0 throughout), seeds 7 and 11:")
    for sd in (7, 11):
        Q0, log0 = train(seed=sd, explore=False)
        e0 = greedy_rollout(Q0)
        print(f"  seed {sd}: training successes={sum(log0['success'])}/1500, "
              f"greedy success={e0['success']} steps={e0['steps']}")

    np.save("q_table.npy", Q)


if __name__ == "__main__":
    main()
