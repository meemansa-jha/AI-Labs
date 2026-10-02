"""Validation tests for the warehouse goal-based agent.  Run: python3 test_agent.py"""

from warehouse_agent import GoalBasedAgent, WAREHOUSE, MOVES


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_finds_path():
    a = GoalBasedAgent(WAREHOUSE)
    path, actions, expanded = a.plan()
    check("a path is found", path is not None)
    check("path starts at S", path[0] == a.start)
    check("path ends at G", path[-1] == a.goal)
    # every step is one legal move onto a passable, non-obstacle cell
    ok = all(abs(x0 - x1) + abs(y0 - y1) == 1 and a.passable((x1, y1))
             for (x0, y0), (x1, y1) in zip(path, path[1:]))
    check("every step is one legal collision-free move", ok)
    check("actions count matches path length", len(actions) == len(path) - 1)


def test_bfs_is_shortest():
    # BFS on a unit-cost grid must return a shortest path. Re-derive the optimum
    # independently with a simple flood fill of distances.
    a = GoalBasedAgent(WAREHOUSE)
    from collections import deque
    dist = {a.start: 0}
    q = deque([a.start])
    while q:
        cur = q.popleft()
        for _n, nxt in a.actions(cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    path, _actions, _exp = a.plan()
    check("BFS path length equals flood-fill shortest distance",
          len(path) - 1 == dist[a.goal])


def test_no_path_reports_failure():
    walled = """\
#######
#S...#
###.###
#...#G#
#######
"""
    a = GoalBasedAgent(walled)
    path, actions, expanded = a.plan()
    check("walled-off goal -> no path (terminates, no infinite loop)",
          path is None)


if __name__ == "__main__":
    test_finds_path()
    test_bfs_is_shortest()
    test_no_path_reports_failure()
    print("\nAll tests passed.")
