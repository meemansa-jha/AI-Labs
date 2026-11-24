"""Problem descriptions for the Planning lab.  Only data lives here — the
planner and validator in planner.py are shared by every domain."""

from planner import act

# ---------------------------------------------------------------------------
# Domain 1: warehouse delivery  (I, A, G)
# ---------------------------------------------------------------------------
WAREHOUSE_I = {"At(Robot,A)", "At(Package,A)"}
WAREHOUSE_G = {"At(Package,C)"}

CONNECTED = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]
LOCATIONS = ["A", "B", "C"]


def move_actions():
    return [act(f"Move({x},{y})", pre=[f"At(Robot,{x})"],
                add=[f"At(Robot,{y})"], delete=[f"At(Robot,{x})"])
            for x, y in CONNECTED]


def pickup_actions():
    return [act(f"PickUp(Package,{x})",
                pre=[f"At(Robot,{x})", f"At(Package,{x})"],
                add=["Holding(Package)"], delete=[f"At(Package,{x})"])
            for x in LOCATIONS]


def drop_actions():
    return [act(f"Drop(Package,{x})",
                pre=[f"At(Robot,{x})", "Holding(Package)"],
                add=[f"At(Package,{x})"], delete=["Holding(Package)"])
            for x in LOCATIONS]


def warehouse_actions(include_pickup: bool = True):
    acts = move_actions() + drop_actions()
    if include_pickup:
        acts += pickup_actions()
    return acts


# ---------------------------------------------------------------------------
# Domain 2: locked-door navigation
# ---------------------------------------------------------------------------
DOOR_I = {"At(A)", "KeyAt(A)"}
DOOR_G = {"At(C)"}


def door_actions(include_pickkey: bool = True):
    acts = [
        act("Move(A,B)", pre=["At(A)"], add=["At(B)"], delete=["At(A)"]),
        act("Move(B,A)", pre=["At(B)"], add=["At(A)"], delete=["At(B)"]),
        act("Unlock", pre=["At(B)", "HasKey"], add=["Unlocked"]),
        act("Move(B,C)", pre=["At(B)", "Unlocked"], add=["At(C)"], delete=["At(B)"]),
        act("Move(C,B)", pre=["At(C)", "Unlocked"], add=["At(B)"], delete=["At(C)"]),
    ]
    if include_pickkey:
        acts.insert(2, act("PickKey", pre=["At(A)", "KeyAt(A)"],
                           add=["HasKey"], delete=["KeyAt(A)"]))
    return acts
