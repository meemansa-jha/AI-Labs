"""Validation tests for the planner.  Run: python3 test_planner.py"""

from planner import bfs_plan, validate_plan, act
import domains as D


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_applicability_in_initial_state():
    s0 = frozenset(D.WAREHOUSE_I)
    table = {a.name: a for a in D.warehouse_actions()}
    check("PickUp(Package,A) applicable in I", table["PickUp(Package,A)"].applicable(s0))
    check("Drop(Package,C) NOT applicable in I", not table["Drop(Package,C)"].applicable(s0))


def test_apply_removes_then_adds():
    a = act("Toggle", pre=["p"], add=["q"], delete=["p"])
    check("apply: delete p, add q", a.apply(frozenset({"p", "r"})) == frozenset({"q", "r"}))


def test_negative_preconditions():
    a = act("OnlyIfNotLocked", pre_neg=["Locked"])
    check("negative precondition blocks action", not a.applicable(frozenset({"Locked"})))
    check("negative precondition allows action", a.applicable(frozenset()))


def test_warehouse_plan_valid_and_shortest():
    r = bfs_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, D.warehouse_actions())
    v = validate_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, D.warehouse_actions(), r.plan)
    check("Test A: plan found", r.found)
    check("Test A: every step valid and goal reached", v.valid and v.reaches_goal)
    check("Test A: plan length 4", len(r.plan) == 4)


def test_impossible():
    r = bfs_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, D.warehouse_actions(include_pickup=False))
    check("Test B: no plan without PickUp", not r.found)


def test_robot_at_c_is_not_package_at_c():
    v = validate_plan(D.WAREHOUSE_I, D.WAREHOUSE_G, D.warehouse_actions(),
                      ["Move(A,B)", "Move(B,C)"])
    check("Test C: robot reaching C does not satisfy At(Package,C)",
          v.valid and not v.reaches_goal)


def test_locked_door():
    r = bfs_plan(D.DOOR_I, D.DOOR_G, D.door_actions())
    check("door: shortest plan has 4 actions", r.found and len(r.plan) == 4)
    check("door: no plan without PickKey",
          not bfs_plan(D.DOOR_I, D.DOOR_G, D.door_actions(False)).found)
    v = validate_plan(D.DOOR_I, D.DOOR_G, D.door_actions(), ["Move(A,B)", "Unlock", "Move(B,C)"])
    check("door: validator rejects Unlock at step 2 for missing HasKey",
          not v.valid and "step 2" in v.error and "HasKey" in v.error)


if __name__ == "__main__":
    for fn in [test_applicability_in_initial_state, test_apply_removes_then_adds,
               test_negative_preconditions, test_warehouse_plan_valid_and_shortest,
               test_impossible, test_robot_at_c_is_not_package_at_c, test_locked_door]:
        fn()
    print("\nAll tests passed.")
