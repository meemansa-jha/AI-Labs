"""
Agents laboratory — a goal-based agent for warehouse navigation.

The agent transports a package from the loading bay S to the dispatch area G,
avoiding shelving (#).  It may move Up / Down / Left / Right by one cell.

A goal-based agent has an explicit objective (reach G) and chooses actions that
move it toward that objective.  Here the decision-making component is a search:
breadth-first search (BFS), which on this unit-cost grid returns a shortest
collision-free path and is guaranteed to report failure (not loop forever) when
no path exists.

Run:  python3 warehouse_agent.py
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

State = Tuple[int, int]

MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

# The warehouse map supplied in the laboratory.
WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
"""


class GoalBasedAgent:
    """Perceives the grid, holds the goal, and plans actions toward it."""

    def __init__(self, ascii_map: str) -> None:
        self.grid: List[str] = [r for r in ascii_map.splitlines() if r]
        self.rows = len(self.grid)
        self.cols = max(len(r) for r in self.grid)
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol: str) -> State:
        for r, row in enumerate(self.grid):
            c = row.find(symbol)
            if c != -1:
                return (r, c)
        raise ValueError(f"symbol {symbol!r} not found in map")

    def passable(self, s: State) -> bool:
        r, c = s
        if not (0 <= r < self.rows):
            return False
        row = self.grid[r]
        return 0 <= c < len(row) and row[c] != "#"

    def actions(self, s: State):
        """Legal (action, successor) pairs — the agent's model of the world."""
        r, c = s
        for name, (dr, dc) in MOVES.items():
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                yield name, nxt

    def is_goal(self, s: State) -> bool:
        return s == self.goal

    # ---- decision-making component: BFS -----------------------------------
    def plan(self):
        """Return (path, actions, expanded) or (None, None, expanded) on failure."""
        frontier: deque[State] = deque([self.start])
        came_from: Dict[State, Optional[State]] = {self.start: None}
        expanded = 0
        while frontier:
            cur = frontier.popleft()
            expanded += 1
            if self.is_goal(cur):
                return self._reconstruct(came_from, cur) + (expanded,)
            for _name, nxt in self.actions(cur):
                if nxt not in came_from:
                    came_from[nxt] = cur
                    frontier.append(nxt)
        return None, None, expanded

    def _reconstruct(self, came_from, cur):
        path = [cur]
        while came_from[cur] is not None:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        inv = {v: k for k, v in MOVES.items()}
        acts = [inv[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]
        return path, acts

    def render(self, path: List[State]) -> str:
        chars = [list(row) for row in self.grid]
        for (r, c) in path:
            if chars[r][c] not in ("S", "G"):
                chars[r][c] = "*"
        return "\n".join("".join(row) for row in chars)


def main() -> None:
    print("Warehouse map:")
    print(WAREHOUSE)
    agent = GoalBasedAgent(WAREHOUSE)
    print(f"start S = {agent.start}")
    print(f"goal  G = {agent.goal}\n")

    path, actions, expanded = agent.plan()
    if path is None:
        print("No collision-free path from S to G exists.")
        print(f"(states expanded before reporting failure: {expanded})")
        return

    print(f"Path found.  length = {len(path) - 1} moves, "
          f"states expanded = {expanded}")
    print("\nActions:")
    print("  " + " -> ".join(actions))
    print("\nPath overlaid on the map (* = path):")
    print(agent.render(path))


if __name__ == "__main__":
    main()
