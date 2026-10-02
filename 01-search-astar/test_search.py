"""
Property-based and known-answer tests for the warehouse search.

Run with:  python3 test_search.py
These tests encode behaviour we KNOW must hold, following the lab principle
'working output != validated algorithm'.
"""

from warehouse_search import Warehouse, astar, bfs, greedy_best_first, manhattan
import maps


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    assert condition, name


def test_trivial_one_step():
    wh = Warehouse(maps.TRIVIAL)
    r = astar(wh)
    check("trivial: solution found", r.found)
    check("trivial: path length is 1", r.path_length == 1)


def test_no_solution_terminates():
    wh = Warehouse(maps.NO_SOLUTION)
    r = astar(wh)
    check("no-solution: reports failure (no infinite loop)", r.found is False)
    check("no-solution: empty path", r.path == [])


def test_astar_matches_bfs_length():
    # BFS is a ground truth for shortest-path length on a unit-cost grid.
    for m in (maps.ORIGINAL, maps.OPEN_WAREHOUSE, maps.ALTERNATIVE,
              maps.GREEDY_TRAP):
        wh = Warehouse(m)
        a, b = astar(wh), bfs(wh)
        check("A* optimal length == BFS length", a.path_length == b.path_length)


def test_path_is_valid():
    # Every consecutive pair in the returned path must be a legal single move
    # onto a passable cell.
    wh = Warehouse(maps.ORIGINAL)
    r = astar(wh)
    check("path starts at S", r.path[0] == wh.start)
    check("path ends at G", r.path[-1] == wh.goal)
    ok = True
    for (r0, c0), (r1, c1) in zip(r.path, r.path[1:]):
        if abs(r0 - r1) + abs(c0 - c1) != 1 or not wh.passable((r1, c1)):
            ok = False
    check("every step is one legal move", ok)


def test_greedy_can_be_suboptimal():
    wh = Warehouse(maps.GREEDY_TRAP)
    a = astar(wh, manhattan)
    g = greedy_best_first(wh, manhattan)
    check("greedy found a path", g.found)
    check("greedy path is longer than A* (demonstrates non-optimality)",
          g.path_length > a.path_length)


if __name__ == "__main__":
    test_trivial_one_step()
    test_no_solution_terminates()
    test_astar_matches_bfs_length()
    test_path_is_valid()
    test_greedy_can_be_suboptimal()
    print("\nAll tests passed.")
