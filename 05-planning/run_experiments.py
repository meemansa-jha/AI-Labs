"""Reproduces every experiment in the Planning lab (Tasks 0-4, 6, 7)."""

from planner import bfs_plan, validate_plan, print_trace, fmt
import domains as D


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def report_validation(label, init, goal, actions, plan):
    v = validate_plan(init, goal, actions, plan)
    print(f"\n{label}: {', '.join(plan) if plan else '(empty)'}")
    if v.valid:
        print(f"  valid = True, reaches goal = {v.reaches_goal}, length = {len(plan)}")
    else:
        print(f"  valid = False -> {v.error}")
    return v


def main():
    W = D.warehouse_actions()

    banner("TASK 0 — which actions are applicable in I?")
    s0 = frozenset(D.WAREHOUSE_I)
    print(f"I = {fmt(s0)}")
    for a in W:
        miss = a.missing_preconditions(s0)
        tag = "APPLICABLE" if not miss else "not applicable (missing " + ", ".join(miss) + ")"
        print(f"  {a.name:<20} {tag}")

    banner("TASK 1 — hand-built plan, checked by the validator")
    hand = ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    v = report_validation("Hand plan", D.WAREHOUSE_I, D.WAREHOUSE_G, W, hand)
    print_trace(hand, v.states)
    report_validation("Lab's example sequence", D.WAREHOUSE_I, D.WAREHOUSE_G, W,
                      ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"])

    banner("TASK 3 / TEST A — solvable warehouse problem")
    r = bfs_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, W)
    print(f"plan found = {r.found}, length = {len(r.plan)}, states expanded = {r.expanded}")
    print_trace(r.plan, r.states)
    report_validation("Independent validation of BFS plan", D.WAREHOUSE_I, D.WAREHOUSE_G, W, r.plan)

    banner("TASK 3 / TEST B — impossible problem (PickUp removed)")
    r = bfs_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, D.warehouse_actions(include_pickup=False))
    print("No plan found" if not r.found else f"UNEXPECTED plan {r.plan}")
    print(f"(states expanded before giving up: {r.expanded})")

    banner("TASK 3 / TEST C — irrelevant actions: robot at C ≠ package at C")
    r_robot = bfs_plan(D.WAREHOUSE_I, {"At(Robot,C)"}, W)
    print(f"Goal At(Robot,C):   plan = {r_robot.plan}")
    r_pkg = bfs_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, W)
    print(f"Goal At(Package,C): plan = {r_pkg.plan}")
    trick = ["Move(A,B)", "Move(B,C)"]
    report_validation("Robot-only moves for goal At(Package,C)",
                      D.WAREHOUSE_I, D.WAREHOUSE_G, W, trick)

    banner("TASK 6 — locked door")
    DA = D.door_actions()
    r = bfs_plan(D.DOOR_I, D.DOOR_G, DA)
    print(f"(b) BFS plan found = {r.found}, length = {len(r.plan)}, expanded = {r.expanded}")
    print_trace(r.plan, r.states)

    r2 = bfs_plan(D.DOOR_I, D.DOOR_G, D.door_actions(include_pickkey=False))
    print(f"\n(c) Without PickKey: plan found = {r2.found}, "
          f"reachable states explored = {r2.expanded}")

    report_validation("(d) Proposed sequence", D.DOOR_I, D.DOOR_G, DA,
                      ["Move(A,B)", "Unlock", "Move(B,C)"])

    banner("TASK 7 — validator on example candidate plans "
           "(check a real LLM plan with: python3 check_plan.py door ...)")
    shortest = len(r.plan)
    candidates = {
        "Candidate 1 (correct order)": ["PickKey", "Move(A,B)", "Unlock", "Move(B,C)"],
        "Candidate 2 (forgets the key)": ["Move(A,B)", "Unlock", "Move(B,C)"],
        "Candidate 3 (goes back for the key)": ["Move(A,B)", "Move(B,A)", "PickKey",
                                                "Move(A,B)", "Unlock", "Move(B,C)"],
    }
    for label, plan in candidates.items():
        v = report_validation(label, D.DOOR_I, D.DOOR_G, DA, plan)
        if v.valid:
            print(f"  same length as shortest BFS plan ({shortest})? {len(plan) == shortest}")


if __name__ == "__main__":
    main()
