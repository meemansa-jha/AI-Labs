"""
Logical Reasoning for Planning — a simple STRIPS-style planning agent.

A state is a frozenset of ground propositions (strings such as "At(Robot,A)").
Each action has positive/negative preconditions and positive/negative effects.

    applicable(S, a)  <=>  pre_pos(a) ⊆ S  and  pre_neg(a) ∩ S = ∅     (logic)
    apply(S, a)        =   (S − del(a)) ∪ add(a)                        (effects)
    BFS over applicable actions until S ⊨ G                            (search)

The same machinery solves both domains in this lab: the warehouse delivery
problem and the locked-door navigation problem.  A separate `validate_plan`
function checks any proposed plan step by step and reports the first action
whose preconditions are not satisfied.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Optional, Sequence, Tuple

State = FrozenSet[str]


@dataclass(frozen=True)
class Action:
    name: str
    pre_pos: FrozenSet[str] = field(default_factory=frozenset)
    pre_neg: FrozenSet[str] = field(default_factory=frozenset)
    add: FrozenSet[str] = field(default_factory=frozenset)
    delete: FrozenSet[str] = field(default_factory=frozenset)

    def missing_preconditions(self, state: State) -> List[str]:
        """Preconditions NOT satisfied in `state` (empty list = applicable)."""
        missing = [p for p in sorted(self.pre_pos) if p not in state]
        missing += [f"not {p}" for p in sorted(self.pre_neg) if p in state]
        return missing

    def applicable(self, state: State) -> bool:
        """Logical test S ⊨ Preconditions(a)."""
        return self.pre_pos <= state and not (self.pre_neg & state)

    def apply(self, state: State) -> State:
        """S' = (S − negative effects) ∪ positive effects."""
        return frozenset((state - self.delete) | self.add)


def act(name: str, pre=(), pre_neg=(), add=(), delete=()) -> Action:
    return Action(name, frozenset(pre), frozenset(pre_neg),
                  frozenset(add), frozenset(delete))


def satisfies(state: State, goal: Iterable[str]) -> bool:
    """Goal test S ⊨ G (every goal proposition is true in S)."""
    return frozenset(goal) <= state


@dataclass
class PlanResult:
    found: bool
    plan: List[str]
    states: List[State]
    expanded: int


def bfs_plan(initial: Iterable[str], goal: Iterable[str],
             actions: Sequence[Action]) -> PlanResult:
    """Breadth-first forward search: returns a shortest plan or reports failure."""
    start: State = frozenset(initial)
    goal_set = frozenset(goal)
    frontier = deque([start])
    parent = {start: None}            # state -> (previous state, action name)
    expanded = 0

    while frontier:
        s = frontier.popleft()
        expanded += 1
        if satisfies(s, goal_set):
            plan, states = [], [s]
            while parent[s] is not None:
                prev, name = parent[s]
                plan.append(name)
                states.append(prev)
                s = prev
            return PlanResult(True, plan[::-1], states[::-1], expanded)
        for a in actions:
            if a.applicable(s):                  # logic: only applicable actions
                s2 = a.apply(s)
                if s2 not in parent:             # search: avoid revisiting states
                    parent[s2] = (s, a.name)
                    frontier.append(s2)
    return PlanResult(False, [], [], expanded)


@dataclass
class Validation:
    valid: bool
    reaches_goal: bool
    states: List[State]
    error: Optional[str] = None


def validate_plan(initial: Iterable[str], goal: Iterable[str],
                  actions: Sequence[Action], plan: Sequence[str]) -> Validation:
    """Independently execute a proposed plan and check every precondition."""
    table = {a.name: a for a in actions}
    s: State = frozenset(initial)
    states = [s]
    for i, name in enumerate(plan, start=1):
        if name not in table:
            return Validation(False, False, states,
                              f"step {i}: {name} is not an available action")
        a = table[name]
        missing = a.missing_preconditions(s)
        if missing:
            return Validation(False, False, states,
                              f"step {i}: {name} is not applicable; unsatisfied "
                              f"precondition(s): {', '.join(missing)}")
        s = a.apply(s)
        states.append(s)
    ok_goal = satisfies(s, goal)
    return Validation(True, ok_goal, states,
                      None if ok_goal else "plan executes but goal not satisfied")


def fmt(state: State) -> str:
    return "{" + ", ".join(sorted(state)) + "}"


def print_trace(plan: Sequence[str], states: Sequence[State]) -> None:
    print(f"  S0 = {fmt(states[0])}")
    for i, (name, s) in enumerate(zip(plan, states[1:]), start=1):
        print(f"  --{name}-->")
        print(f"  S{i} = {fmt(s)}")
