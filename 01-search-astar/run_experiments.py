"""
Driver for the Search lab.

Runs every required test (Task 3), the BFS/A* comparison (Task 5) and the
heuristic investigation (Task 6), printing a reproducible transcript.
"""

from warehouse_search import (
    Warehouse, astar, bfs, greedy_best_first, render_path, path_as_moves,
    manhattan, euclidean, zero, manhattan_x2,
)
import maps


def line(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_one(name: str, ascii_map: str) -> None:
    print(f"\n--- {name} ---")
    wh = Warehouse(ascii_map)
    print(f"start={wh.start}  goal={wh.goal}")
    res = astar(wh, manhattan)
    print(f"A* found={res.found}  path_length={res.path_length}  "
          f"states_expanded={res.expanded}")
    if res.found:
        print("path (cells):", res.path)
        print("path (moves):", path_as_moves(res.path))
        print(render_path(wh, res.path))


def main() -> None:
    line("TASK 3 — SYSTEMATIC TESTS (A*, Manhattan heuristic)")
    run_one("Test 1: original warehouse", maps.ORIGINAL)
    run_one("Test 2: trivial (goal adjacent)", maps.TRIVIAL)
    run_one("Test 3: no solution", maps.NO_SOLUTION)
    run_one("Test 4: alternative paths", maps.ALTERNATIVE)

    line("TASK 5 — BFS vs A* (original corridor map AND open map)")
    for mapname, ascii_map in [("Original (corridor) map", maps.ORIGINAL),
                               ("Open warehouse map", maps.OPEN_WAREHOUSE)]:
        wh = Warehouse(ascii_map)
        a = astar(wh, manhattan)
        b = bfs(wh)
        print(f"\n{mapname}:")
        print(f"{'Measure':<20}{'BFS':<12}{'A*':<12}")
        print(f"{'Solution found':<20}{str(b.found):<12}{str(a.found):<12}")
        print(f"{'Path length':<20}{b.path_length:<12}{a.path_length:<12}")
        print(f"{'States expanded':<20}{b.expanded:<12}{a.expanded:<12}")

    line("TASK 6 — HEURISTIC INVESTIGATION ON THE OPEN MAP")
    who = Warehouse(maps.OPEN_WAREHOUSE)
    print(f"{'Heuristic':<30}{'found':<8}{'path_len':<10}{'expanded':<10}")
    for label, h in [("Manhattan (baseline)", manhattan),
                     ("Zero (uniform cost)", zero),
                     ("Euclidean", euclidean),
                     ("2 x Manhattan (inadmissible)", manhattan_x2)]:
        r = astar(who, h)
        print(f"{label:<30}{str(r.found):<8}{r.path_length:<10}{r.expanded:<10}")

    # Confirm optimality on the alternative-paths map for every heuristic.
    line("TASK 6 — SHORTEST-PATH CHECK ON THE ALTERNATIVE MAP")
    wh2 = Warehouse(maps.ALTERNATIVE)
    optimal = astar(wh2, manhattan).path_length
    print(f"Known shortest path length = {optimal}")
    for label, h in [("Zero", zero), ("Euclidean", euclidean),
                     ("2 x Manhattan", manhattan_x2)]:
        r = astar(wh2, h)
        verdict = "OPTIMAL" if r.path_length == optimal else "SUBOPTIMAL"
        print(f"  {label:<16} path_len={r.path_length}  ({verdict})")

    line("BONUS — A* (optimal) vs GREEDY BEST-FIRST (too aggressive)")
    wh3 = Warehouse(maps.GREEDY_TRAP)
    a = astar(wh3, manhattan)
    g = greedy_best_first(wh3, manhattan)
    print("Map:")
    print(maps.GREEDY_TRAP)
    print(f"A*     : path_length={a.path_length:<4} expanded={a.expanded}  "
          f"(optimal, balances g and h)")
    print(f"Greedy : path_length={g.path_length:<4} expanded={g.expanded}  "
          f"({'SUBOPTIMAL' if g.path_length > a.path_length else 'optimal'}, "
          f"chases h only)")
    print("A* path overlay:")
    print(render_path(wh3, a.path))
    print("Greedy path overlay:")
    print(render_path(wh3, g.path))


if __name__ == "__main__":
    main()
