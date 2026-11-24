"""
Warehouse environments for the RL lab.

5x5 grid, positions (r, c), rows increase downward.  Obstacles form a wall in
column 2 with a single doorway at (2, 2).

Actions (codes 0-4, in this order): Up, Down, Left, Right, Wait.
A move into an obstacle or beyond the grid leaves the robot in place.

step() returns (next_state, reward, terminated, truncated, info).
  terminated = the task truly ended (goal reached, or collision for two robots)
  truncated  = the 40-step collection cutoff was hit (NOT a task ending)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

N = 5
OBSTACLES = frozenset({(0, 2), (1, 2), (3, 2), (4, 2)})
ACTIONS = ["Up", "Down", "Left", "Right", "Wait"]
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
MAX_STEPS = 40
GAMMA = 0.95

Pos = Tuple[int, int]


def propose(pos: Pos, action: int) -> Tuple[Pos, bool]:
    """Return (resulting position, invalid?) for one robot's action.

    An obstacle or boundary move stays in place and is flagged invalid.
    Wait is always valid.
    """
    dr, dc = DELTAS[action]
    r, c = pos[0] + dr, pos[1] + dc
    if not (0 <= r < N and 0 <= c < N) or (r, c) in OBSTACLES:
        return pos, True
    return (r, c), False


def index(pos: Pos) -> int:
    """Q-table row / one-hot index of a cell: 5r + c."""
    return N * pos[0] + pos[1]


def one_hot(pos: Pos) -> np.ndarray:
    v = np.zeros(N * N, dtype=np.float32)
    v[index(pos)] = 1.0
    return v


def render(paths: Dict[str, list]) -> str:
    """Draw the grid with each robot's trajectory marked by its letter."""
    grid = [["#" if (r, c) in OBSTACLES else "." for c in range(N)] for r in range(N)]
    for label, path in paths.items():
        for p in path:
            grid[p[0]][p[1]] = label if grid[p[0]][p[1]] in (".", label) else "*"
    return "\n".join(" ".join(row) for row in grid)


# ---------------------------------------------------------------------------
# Single robot
# ---------------------------------------------------------------------------
class SingleRobotEnv:
    START: Pos = (2, 0)
    GOAL: Pos = (2, 4)

    def __init__(self, start: Optional[Pos] = None) -> None:
        self.start = start or self.START
        self.pos: Pos = self.start
        self.t = 0

    def reset(self, start: Optional[Pos] = None) -> Pos:
        self.pos = start or self.start
        self.t = 0
        return self.pos

    def set_position(self, pos: Pos) -> None:
        """Test mode: put the robot anywhere without training."""
        self.pos = pos
        self.t = 0

    def step(self, action: int):
        new, invalid = propose(self.pos, action)
        self.pos = new
        self.t += 1
        # Mutually exclusive rewards.
        if new == self.GOAL:
            reward, terminated = 10.0, True
        elif invalid:
            reward, terminated = -1.0, False
        else:
            reward, terminated = -0.1, False
        truncated = (not terminated) and self.t >= MAX_STEPS
        return new, reward, terminated, truncated, {"invalid": invalid}


# ---------------------------------------------------------------------------
# Two robots, one joint controller
# ---------------------------------------------------------------------------
class TwoRobotEnv:
    START = ((2, 0), (2, 4))   # (A, B)
    GOALS = ((2, 4), (2, 0))   # A's goal, B's goal

    def __init__(self, start=None) -> None:
        self.start = start or self.START
        self.state = self.start
        self.t = 0

    def reset(self, start=None):
        self.state = start or self.start
        self.t = 0
        return self.state

    def set_state(self, state) -> None:
        self.state = state
        self.t = 0

    @staticmethod
    def encode_action(a_A: int, a_B: int) -> int:
        return 5 * a_A + a_B

    @staticmethod
    def decode_action(j: int):
        return divmod(j, 5)

    def step(self, joint: int):
        a_A, a_B = self.decode_action(joint)
        old_A, old_B = self.state
        at_goal_A, at_goal_B = old_A == self.GOALS[0], old_B == self.GOALS[1]

        # Both proposals use the same old state.  A robot at its goal stays.
        new_A, inv_A = (old_A, False) if at_goal_A else propose(old_A, a_A)
        new_B, inv_B = (old_B, False) if at_goal_B else propose(old_B, a_B)

        same_cell = new_A == new_B
        swap = new_A == old_B and new_B == old_A and new_A != old_A
        self.t += 1

        if same_cell or swap:
            # Collision overrides every other reward term; keep old positions.
            info = {"collision": "same-cell" if same_cell else "swap",
                    "invalid": int(inv_A) + int(inv_B)}
            truncated = False
            return self.state, -10.0, True, truncated, info

        self.state = (new_A, new_B)
        b = int(inv_A) + int(inv_B)
        k = int(not at_goal_A and new_A == self.GOALS[0]) + \
            int(not at_goal_B and new_B == self.GOALS[1])
        both = new_A == self.GOALS[0] and new_B == self.GOALS[1]
        reward = -0.1 - b + 5 * k + 5 * int(both)
        terminated = both
        truncated = (not terminated) and self.t >= MAX_STEPS
        return self.state, reward, terminated, truncated, {"collision": None, "invalid": b}


def two_robot_input(state) -> np.ndarray:
    """Concatenate two length-25 one-hot vectors -> length 50."""
    return np.concatenate([one_hot(state[0]), one_hot(state[1])])


def shortest_joint_plan():
    """Breadth-first search over joint states with the SAME collision rules.
    Used only as a reference to judge the learned route's length — it is never
    given to the learner."""
    from collections import deque
    env = TwoRobotEnv()
    start = env.START
    parent = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s[0] == env.GOALS[0] and s[1] == env.GOALS[1]:
            n = 0
            while parent[s] is not None:
                s = parent[s]; n += 1
            return n
        for j in range(25):
            env.set_state(s)
            s2, r, term, trunc, info = env.step(j)
            if info["collision"]:
                continue
            if s2 not in parent:
                parent[s2] = s
                q.append(s2)
    return None
