"""
Check any proposed plan (for example, one an LLM suggested) with the
independent validator.

Usage:
    python3 check_plan.py door      "PickKey" "Move(A,B)" "Unlock" "Move(B,C)"
    python3 check_plan.py warehouse "PickUp(Package,A)" "Move(A,B)" "Move(B,C)" "Drop(Package,C)"
"""

import sys

from planner import bfs_plan, validate_plan, print_trace
import domains as D

PROBLEMS = {
    "door": (D.DOOR_I, D.DOOR_G, D.door_actions()),
    "warehouse": (D.WAREHOUSE_I, D.WAREHOUSE_G, D.warehouse_actions()),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PROBLEMS:
        print(__doc__)
        sys.exit(1)
    init, goal, actions = PROBLEMS[sys.argv[1]]
    plan = sys.argv[2:]
    shortest = len(bfs_plan(init, goal, actions).plan)
    v = validate_plan(init, goal, actions, plan)
    print(f"plan: {', '.join(plan) if plan else '(empty)'}")
    if not v.valid:
        print(f"INVALID -> {v.error}")
    else:
        print(f"valid = True, reaches goal = {v.reaches_goal}, length = {len(plan)}, "
              f"shortest possible = {shortest}")
        print_trace(plan, v.states)
