# Laboratory Report — Logical Reasoning for Planning

**Using an LLM to Construct and Test a Simple Planning Agent** · Artificial Intelligence

| File | Purpose |
|------|---------|
| `planner.py` | shared STRIPS-style machinery: `Action`, applicability test, `apply`, BFS planner, independent `validate_plan` |
| `domains.py` | problem descriptions only: warehouse delivery and locked door |
| `run_experiments.py` | reproduces Tasks 0, 1, 3, 6 and 7 |
| `check_plan.py` | checks any proposed plan from the command line (used for Tasks 5 and 7) |
| `test_planner.py` | validation tests |
| `planner.pl` | Prolog knowledge base for Tasks 8–10 |
| `experiment_transcript.txt`, `prolog_transcript.txt` | captured output |

```bash
python3 test_planner.py
python3 run_experiments.py
swipl -q -s planner.pl -g run_all -t halt      # optional Prolog part (SWI-Prolog)
```

The central idea: **Logic + Search = Planning.** Logic decides whether an action is applicable (`S ⊨ Preconditions(a)`) and how it changes the world; search decides which sequence of applicable actions to try.

---

## Task 0 — Understand the planning problem

**(a) Initial state.** `I = {At(Robot,A), At(Package,A)}`.

**(b) Goal.** `G = {At(Package,C)}`. Only the package's location matters; where the robot ends up does not.

**(c) Actions.** `Move(A,B)`, `Move(B,A)`, `Move(B,C)`, `Move(C,B)`, `PickUp(Package,x)` and `Drop(Package,x)` for `x ∈ {A,B,C}`.

**(d) Preconditions and effects.**

| Action | Preconditions | Add effects | Delete effects |
|---|---|---|---|
| `Move(x,y)` (x,y connected) | `At(Robot,x)` | `At(Robot,y)` | `At(Robot,x)` |
| `PickUp(Package,x)` | `At(Robot,x)`, `At(Package,x)` | `Holding(Package)` | `At(Package,x)` |
| `Drop(Package,x)` | `At(Robot,x)`, `Holding(Package)` | `At(Package,x)` | `Holding(Package)` |

**Initially applicable actions** (computed by the program): only `Move(A,B)` and `PickUp(Package,A)`.

**Question.** `PickUp(Package,A)` **is applicable** in `I`: both of its preconditions, `At(Robot,A)` and `At(Package,A)`, are in `I`. `Drop(Package,C)` **is not applicable**: it needs `At(Robot,C)` and `Holding(Package)`, and neither is in `I`. Appearing in the list of actions is not enough; every precondition must hold in the current state.

---

## Task 1 — A plan constructed by hand

Plan: `PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C)`.

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 (after PickUp(Package,A)) | At(Robot,A), Holding(Package) |
| S2 (after Move(A,B)) | At(Robot,B), Holding(Package) |
| S3 (after Move(B,C)) | At(Robot,C), Holding(Package) |
| S4 (after Drop(Package,C)) | At(Robot,C), At(Package,C) ✓ `S4 ⊨ G` |

**A note on the sequence suggested in the handout.** The handout lists `Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)` as things one "might need to reason about". Run through the validator, that sequence is **invalid**: at step 2, `PickUp(Package,B)` requires `At(Package,B)`, but the package is still at A. It is a useful example of a sequence that looks reasonable but fails a precondition check. The robot must pick the package up at A before leaving.

---

## Task 2 — Prompt and how the LLM was used

The LLM used was **Claude** (Anthropic). It was given the full lab handout (the PDF) and asked to solve the lab completely. The specification in Task 0, the hand plans, `planner.py`, `domains.py`, the tests, the Prolog file and the first draft of this report came from that session. The implementation follows the handout's specification prompt:

> I want to implement a simple planning agent in Python. Represent a state as a set of logical propositions. Each action should contain a name, positive preconditions, negative preconditions, positive effects, and negative effects. An action is applicable if all of its preconditions are satisfied by the current state. When an action is applied: (1) remove its negative effects from the state; (2) add its positive effects. Use breadth-first search to find a sequence of actions that achieves a specified goal. The program should detect when no plan exists, print the resulting sequence of actions, and print the states reached after each action. Explain the implementation and identify any assumptions you make.

**Where each idea in the specification appears in the program**

| Specification idea | Question it answers | Location in `planner.py` |
|---|---|---|
| Preconditions | When is an action applicable? | `Action.applicable()`: `pre_pos ⊆ S` and `pre_neg ∩ S = ∅` |
| Effects | How does the state change? | `Action.apply()`: `(S − delete) ∪ add` |
| Goal | When does planning terminate? | `satisfies(state, goal)`, checked when a state is popped in `bfs_plan` |
| BFS | How are alternative plans explored? | `bfs_plan()`: FIFO `deque` frontier, `parent` dict as the visited set |

**What was checked and changed.** The planner passed every test on its first run. Its design choices were inspected and kept: states are `frozenset`s (immutable, so they can be keys in the visited table); the validator is a **separate** function that re-executes a plan from scratch rather than reusing the planner's own trace, so a planner bug cannot hide itself; and problem data lives in `domains.py`, so the locked-door domain reuses the planner unchanged, as Task 6 requires. Two real issues came up while running it. (1) The plan *suggested in the handout* for Task 1 fails validation (see the note above). (2) SWI-Prolog warned about "Illegal multibyte Sequence" in `planner.pl` because of a non-ASCII dash in a comment; it was replaced with an ASCII hyphen.

**Assumptions made explicit.** Closed-world assumption: a proposition not in the state is false. Actions are deterministic and have unit cost. Delete effects are applied before add effects. Movement is only along the listed connections.

---

## Task 3 — Testing the planner

| Test | Initial state | Goal | Plan found? | Plan | Actually valid? |
|---|---|---|---|---|---|
| A: solvable | At(Robot,A), At(Package,A) | At(Package,C) | yes (8 states expanded) | PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | yes: every step re-checked by `validate_plan`, goal reached |
| B: PickUp removed | same | At(Package,C) | **No plan found** (3 reachable states exhausted) | — | correct: no action can ever add `Holding(Package)` |
| C: irrelevant actions | same | At(Package,C) | yes, the same 4-step plan | — | yes; the 2-step robot-only plan `Move(A,B), Move(B,C)` executes but `reaches_goal = False` |

Test C shows the planner does not confuse "robot at C" with "package at C". With goal `At(Robot,C)` it returns the 2-step plan `Move(A,B), Move(B,C)`; with goal `At(Package,C)` it returns the 4-step plan. When the robot-only sequence is checked against the package goal, the validator reports that it is executable but does not satisfy G.

---

## Task 4 — Logic and search

Completed description:

```
Current state
   ↓
Check action preconditions        (logic: S ⊨ Preconditions(a)?)
   ↓
Keep only the applicable actions  ← the missing step
   ↓
Generate successor state          (S' = Apply(S, a) = (S − Del) ∪ Add)
   ↓
Search over alternatives          (BFS frontier, skip already-seen states)
   ↓
Goal?                             (S' ⊨ G ? stop : continue)
```

**How they work together.** Logical reasoning acts locally, on one state and one action at a time: it decides whether an action may be taken (entailment of its preconditions) and what the world looks like afterwards (applying its effects). That defines the edges of a state graph. Search acts globally: it decides which of those legal edges to follow and in what order, keeping a frontier of partial plans and stopping when a state entails the goal. In short, logic determines what is possible and search determines what to try. Without the logic, the search would follow illegal actions; without the search, the logic could only tell us whether a single next action is allowed, not how to reach the goal.

---

## Task 5 (optional) — Can the LLM verify its own plan?

**How to run it.** Ask an LLM: *"For every action in the plan PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C), identify its preconditions and show that those preconditions are satisfied in the state in which the action is executed."* Compare its answer line by line with the state trace from `python3 check_plan.py warehouse "PickUp(Package,A)" "Move(A,B)" "Move(B,C)" "Drop(Package,C)"`. For a sharper test, ask the same about the handout's sequence `Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)`; the checker rejects it at step 2.

*(The LLM's own explanation is not recorded here: this exercise should be run with your own LLM session. Paste its answer below this line.)*

**Which to trust more: (b) the independently executed state transitions.** The program applies the exact precondition test and the exact effects to an explicit state at every step. Its result is determined by the specification, can be re-run, and fails loudly at the first unsatisfied precondition. An LLM's explanation is generated text. On a problem this small it is often right, but nothing in how it is produced forces it to agree with the actual effects, so it is a claim to be checked rather than a check. The handout's own example sequence shows how easily a reasonable-sounding narrative ("move to B, pick up the package…") contains an action whose precondition is false. A generated explanation is not the same as an independent verification.

---

## Task 6 — Locked-door domain

**(a) Shortest plan by hand.**

| State | Facts | Action that produced it |
|---|---|---|
| S0 | At(A), KeyAt(A) | — |
| S1 | At(A), HasKey | PickKey |
| S2 | At(B), HasKey | Move(A,B) |
| S3 | At(B), HasKey, Unlocked | Unlock |
| S4 | At(C), HasKey, Unlocked ✓ | Move(B,C) |

Four actions. No shorter plan exists: reaching C needs `Move(B,C)`, which needs `Unlocked`, which needs `Unlock`, which needs `HasKey` at B, which needs `PickKey` at A followed by `Move(A,B)`.

**(b) BFS with the same planner.** Only `domains.py` changed. BFS returns exactly `PickKey, Move(A,B), Unlock, Move(B,C)` (4 actions, 7 states expanded), with the same state trace as above.

**(c) Without PickKey.** No plan exists. `HasKey` is never in the initial state and `PickKey` is the only action that adds it, so `Unlock` can never become applicable, `Unlocked` is never true, and `Move(B,C)` is never applicable. BFS explores the only two reachable states, `{At(A),KeyAt(A)}` and `{At(B),KeyAt(A)}`, and reports failure.

**(d) Validating `Move(A,B), Unlock, Move(B,C)`.** The first invalid action is **step 2, `Unlock`**. Its unsatisfied precondition is **`HasKey`**. (`At(B)` is satisfied after step 1.)

---

## Task 7 — LLM plans across two domains

**Prompt to give the LLM** (BFS solution not shown):

> Here is a planning problem. Initial state I = {At(A), KeyAt(A)}. Goal G = {At(C)}. A proposition not in the state is false. Actions (preconditions → effects): Move(A,B): At(A) → del At(A), add At(B); Move(B,A): At(B) → del At(B), add At(A); PickKey: At(A), KeyAt(A) → del KeyAt(A), add HasKey; Unlock: At(B), HasKey → add Unlocked; Move(B,C): At(B), Unlocked → del At(B), add At(C); Move(C,B): At(C), Unlocked → del At(C), add At(B). Reply with only a sequence of these six actions that reaches the goal.

**How to check its answer:**

```bash
python3 check_plan.py door <the LLM's actions in order>
```

If the plan is rejected, give the LLM the checker's first error message verbatim, ask for a revised plan, and run the checker again.

*(This exercise should be run with your own LLM session, so the LLM's actual plans are not recorded here. Fill in the table below with what it returns.)*

| Plan proposed by the LLM | Valid? | Reaches goal? | Same length as BFS (4)? |
|---|---|---|---|
| first answer: … | | | |
| revision (if needed): … | | | |

**What the checker reports for the possible kinds of answer** (from `run_experiments.py`):

| Candidate plan | Valid? | Reaches goal? | Same length as BFS (4)? |
|---|---|---|---|
| `PickKey, Move(A,B), Unlock, Move(B,C)` | yes | yes | yes |
| `Move(A,B), Unlock, Move(B,C)` (forgets the key; the Task 6(d) sequence) | **no**: step 2 `Unlock` is missing `HasKey` | — | — |
| `Move(A,B), Move(B,A), PickKey, Move(A,B), Unlock, Move(B,C)` (goes back for the key) | yes | yes | no (6 > 4) |

The third row shows why validity and optimality must be checked separately: a revision that patches an error (going back for the key) can be valid without being shortest. Only the BFS planner guarantees the shortest plan, and only the validator establishes validity.

**Think About It.** A plausible narrative does not guarantee that preconditions are satisfied. A narrative can skip a step or assume a fact ("the robot unlocks the door") without the state ever containing `HasKey`. An independent state-transition checker establishes three things an explanation cannot: that each action's preconditions are entailed by the actual state at that step, that the final state entails the goal, and, if not, exactly which action fails and why. It is mechanical and repeatable.

---

## Tasks 8–10 (optional) — Prolog as a logical verifier

Output of `swipl -q -s planner.pl -g run_all -t halt`:

```
Task 8
?- can_move(a,b).             true
?- can_move(a,c).             false
Task 9
?- valid_move(a,b).           true
?- valid_move(b,c).           true
?- valid_move(a,c).           false
?- valid_plan(a,[b,c]).       true
?- valid_plan(a,[c]).         false
Task 10
?- reduce_speed.              true
```

**Task 8.**
(a) `can_move(a,b)` is true because Prolog resolves the goal with the rule `can_move(X,Y) :- connected(X,Y)`, binding X=a, Y=b, and the new subgoal `connected(a,b)` matches a stored fact.
(b) `can_move(a,c)` is not established because the only way to prove it is via `connected(a,c)`, and that is neither a fact nor derivable from any rule. Under Prolog's closed-world, negation-as-failure semantics, "not provable" is reported as `false`. Note that this means "does not follow from the knowledge base", not "proved impossible in every world".
(c) The rule is the Horn clause `∀X ∀Y (Connected(X,Y) → CanMove(X,Y))`. The head `can_move(X,Y)` is the consequent, the body `connected(X,Y)` is the antecedent, and `:-` is read as "if".

**Task 9 — challenge.** The proposed action `Move(a,c)` is **not supported** by the warehouse knowledge: `valid_move(a,c)` fails because `connected(a,c)` cannot be proved. The added `valid_plan/2` predicate checks a whole movement plan: `valid_plan(a,[b,c])` (Move(a,b), Move(b,c)) succeeds, while `valid_plan(a,[c])` fails. This is the architecture *generate → independent verification*: Python generates the candidate, and Prolog checks it against the logical description of the world.

**Task 10.** The query succeeds because Prolog chains backwards from the goal: `reduce_speed` holds if `slippery` holds; `slippery` holds if `wet_road` holds; `wet_road` is a fact. As forward implications:

`wet_road` (fact) ⇒ `wet_road → slippery` (rule) ⇒ `slippery → reduce_speed` (rule) ⇒ `reduce_speed` (conclusion).

(This is modus ponens applied twice.)

### 8.2 Prolog reflection

1. **Fact vs rule.** A fact is an unconditional assertion (`connected(a,b).`), true with no conditions. A rule (`can_move(X,Y) :- connected(X,Y).`) is a conditional statement: its head is true whenever its body can be proved.
2. **Query as entailment.** `?- q.` asks whether `q` can be derived from the facts and rules, approximately whether `KB ⊨ q`. Prolog answers by searching for a proof (SLD resolution with unification and backtracking). A `false` means no proof was found under the closed-world assumption.
3. **Why verify a Python plan with Prolog.** The check is written in a different language, in a different style (declarative rules rather than procedural code), directly from the domain description. A bug in the Python planner, such as a missing precondition test, is therefore unlikely to be reproduced in the verifier, so agreement between the two is real evidence.
4. **Value of an independent verifier with LLM-generated plans.** LLM output can be fluent and wrong in ways that are hard to see by reading. An independent verifier gives a mechanical, repeatable yes/no answer tied to the specification and points to the exact failing step, which can be fed back to the generator. Trust then rests on the check, not on the generator.

---

## Reflection Questions

1. **Why specify preconditions and effects before asking the LLM?** They *are* the planning problem. If they are left to the LLM, it may invent or drop conditions (for example, letting the robot drop a package it is not holding), and there is nothing to test the code against. Written down first, they become the contract for the code and the basis of every test.

2. **An error from not checking preconditions.** The planner could output `Drop(Package,C)` straight from the initial state, "delivering" a package the robot never picked up, or `Move(B,C)` through a locked door. The plan would look short and successful while being physically impossible.

3. **Why a plan that "looks reasonable" may be invalid.** Validity is a precise property: each action's preconditions must hold in the exact state produced by the previous actions. The handout's own example sequence (`Move(A,B), PickUp(Package,B), …`) reads naturally but fails at step 2 because the package was left at A. Plausibility is judged against intuition; validity is judged against the state.

4. **What the LLM contributed.** Nearly everything in this folder: from the handout, Claude produced the problem specification, hand plans, the planner, the validator, both domains, the tests, the Prolog file and the report draft. It worked first time. Its value was speed. Its output still needed checking, and in this lab the checking found that the handout's own example sequence was invalid, which a plausible-looking answer could easily have repeated.

5. **What was verified independently.** That applicability is a full subset test (including negative preconditions); that effects delete then add; that the planner reports failure when it should (Test B, locked door without key); that a robot-only plan does not satisfy a package goal (Test C); and that every returned plan passes `validate_plan`, which re-executes it from scratch.

6. **Where logical reasoning is used.** In the applicability test `S ⊨ Preconditions(a)`, in computing successor states from effects, in the goal test `S ⊨ G`, in the validator, and in the Prolog queries (fact + rule ⇒ conclusion).

7. **Relation to the search module.** Planning is state-space search where states are sets of propositions and the successor function is defined logically by preconditions and effects. BFS is the same blind algorithm used for the warehouse grid; it gives the shortest plan because every action has unit cost. A heuristic (for example, the number of unsatisfied goal propositions) would turn it into informed search such as A\*.

**LLM-assisted vs own work.** The specification, plans, code, tests and report draft were generated with Claude from the lab handout and accepted after passing the tests above. The parts that require your own LLM session are Tasks 5 and 7; their prompts and a checker for the answers are provided above.
