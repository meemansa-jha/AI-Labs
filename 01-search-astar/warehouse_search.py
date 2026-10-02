"""
Warehouse Robot Navigation — A* and BFS search.

AI Laboratory: Search and A*.

The warehouse is an ASCII grid:
    #  obstacle
    .  free cell
    S  start
    G  goal

The robot moves Up / Down / Left / Right, every move costs 1.

This module implements:
  * a search-problem formulation of the warehouse (states, actions, goal test);
  * A* search with a pluggable heuristic;
  * Breadth-first search (BFS) for comparison;
  * instrumentation (path, path length, number of states expanded).

Author: Meemansa (design + specification), implementation assisted by an LLM
        and verified/modified by hand.  See REPORT.md for the full account of
        what was designed, generated, accepted, changed and tested.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

# A state is simply the robot's (row, col) position on the grid.
State = Tuple[int, int]

# The four available actions and the change they make to (row, col).
MOVES: Dict[str, Tuple[int, int]] = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class Warehouse:
    """A parsed warehouse map that knows its start, goal and legal cells."""

    def __init__(self, ascii_map: str) -> None:
        # Split into rows, dropping any empty trailing lines.
        self.grid: List[str] = [row for row in ascii_map.splitlines() if row != ""]
        self.n_rows = len(self.grid)
        self.n_cols = max(len(r) for r in self.grid) if self.grid else 0
        self.start: Optional[State] = None
        self.goal: Optional[State] = None

        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)

        if self.start is None:
            raise ValueError("Map has no start cell 'S'.")
        if self.goal is None:
            raise ValueError("Map has no goal cell 'G'.")

    def passable(self, state: State) -> bool:
        """A cell is legal if it is on the grid and is not an obstacle."""
        r, c = state
        if not (0 <= r < self.n_rows):
            return False
        row = self.grid[r]
        if not (0 <= c < len(row)):
            return False
        return row[c] != "#"

    def neighbours(self, state: State):
        """Yield (action, next_state) for every legal move from `state`."""
        r, c = state
        for action, (dr, dc) in MOVES.items():
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                yield action, nxt

    def is_goal(self, state: State) -> bool:
        return state == self.goal


# ----------------------------------------------------------------------------
# Heuristics.  All take (state, goal) and return an estimated cost-to-go.
# ----------------------------------------------------------------------------

def manhattan(state: State, goal: State) -> float:
    """h(n) = |x - x_G| + |y - y_G|  — admissible for 4-connected unit-cost grids."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state: State, goal: State) -> float:
    """Straight-line distance.  Also admissible here, but a looser lower bound."""
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def zero(state: State, goal: State) -> float:
    """h(n) = 0.  A* with this heuristic degenerates to uniform-cost search."""
    return 0.0


def manhattan_x2(state: State, goal: State) -> float:
    """2 x Manhattan — an inadmissible (over-estimating) heuristic."""
    return 2.0 * manhattan(state, goal)


# ----------------------------------------------------------------------------
# Search algorithms.
# ----------------------------------------------------------------------------

class SearchResult:
    def __init__(self, found: bool, path: Optional[List[State]],
                 expanded: int) -> None:
        self.found = found
        self.path = path or []
        self.expanded = expanded

    @property
    def path_length(self) -> int:
        # Number of moves = cells - 1 (a path of one cell has length 0).
        return max(len(self.path) - 1, 0) if self.found else 0

    def __repr__(self) -> str:
        return (f"SearchResult(found={self.found}, "
                f"path_length={self.path_length}, expanded={self.expanded})")


def astar(wh: Warehouse,
          heuristic: Callable[[State, State], float] = manhattan) -> SearchResult:
    """A* search.

    f(n) = g(n) + h(n), where g is the cost from the start and h the heuristic
    estimate of the cost to the goal.  The frontier is a priority queue ordered
    by f.  A `g_score` dict records the best known cost to each state and acts
    as the closed/visited test: we skip a popped node whose g has since improved.
    """
    start, goal = wh.start, wh.goal
    # Frontier entries: (f, tie_breaker, state).  The tie_breaker keeps the heap
    # stable and avoids comparing states when f ties.
    counter = 0
    frontier: List[Tuple[float, int, State]] = [
        (heuristic(start, goal), counter, start)
    ]
    came_from: Dict[State, State] = {}
    g_score: Dict[State, float] = {start: 0.0}
    expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        # Stale entry: a better path to `current` was found after this was queued.
        if f - heuristic(current, goal) > g_score.get(current, math.inf) + 1e-9:
            continue

        expanded += 1
        if wh.is_goal(current):
            return SearchResult(True, _reconstruct(came_from, current), expanded)

        for _action, nxt in wh.neighbours(current):
            tentative_g = g_score[current] + 1  # every move costs 1
            if tentative_g < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (tentative_g + heuristic(nxt, goal),
                                          counter, nxt))

    return SearchResult(False, None, expanded)


def greedy_best_first(wh: Warehouse,
                      heuristic: Callable[[State, State], float] = manhattan
                      ) -> SearchResult:
    """Greedy best-first search — orders the frontier by h(n) alone (ignores g).

    This is 'too aggressive': it always races toward whatever looks closest to
    the goal and does NOT guarantee a shortest path.  Included to contrast with
    A*, which balances g and h and stays optimal under an admissible heuristic.
    """
    start, goal = wh.start, wh.goal
    counter = 0
    frontier: List[Tuple[float, int, State]] = [
        (heuristic(start, goal), counter, start)
    ]
    came_from: Dict[State, Optional[State]] = {start: None}
    expanded = 0

    while frontier:
        _h, _, current = heapq.heappop(frontier)
        expanded += 1
        if wh.is_goal(current):
            return SearchResult(True, _reconstruct(came_from, current), expanded)
        for _action, nxt in wh.neighbours(current):
            if nxt not in came_from:
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (heuristic(nxt, goal), counter, nxt))

    return SearchResult(False, None, expanded)


def bfs(wh: Warehouse) -> SearchResult:
    """Breadth-first search — a blind strategy.

    BFS expands states in FIFO order.  On a unit-cost graph it is guaranteed to
    find a shortest path, but it uses no information about where the goal is.
    """
    start = wh.start
    frontier: deque[State] = deque([start])
    came_from: Dict[State, Optional[State]] = {start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if wh.is_goal(current):
            return SearchResult(True, _reconstruct(came_from, current), expanded)
        for _action, nxt in wh.neighbours(current):
            if nxt not in came_from:          # `came_from` doubles as the visited set
                came_from[nxt] = current
                frontier.append(nxt)

    return SearchResult(False, None, expanded)


def _reconstruct(came_from: Dict[State, Optional[State]],
                 current: State) -> List[State]:
    """Walk the parent pointers back to the start to rebuild the path."""
    path = [current]
    while came_from.get(current) is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ----------------------------------------------------------------------------
# Pretty-printing helpers.
# ----------------------------------------------------------------------------

def render_path(wh: Warehouse, path: List[State]) -> str:
    """Overlay the path on the map with '*' (keeping S and G)."""
    chars = [list(row) for row in wh.grid]
    for (r, c) in path:
        if chars[r][c] not in ("S", "G"):
            chars[r][c] = "*"
    return "\n".join("".join(row) for row in chars)


def path_as_moves(path: List[State]) -> List[str]:
    """Convert a sequence of cells into the action names taken."""
    inverse = {v: k for k, v in MOVES.items()}
    moves = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        moves.append(inverse[(r1 - r0, c1 - c0)])
    return moves
