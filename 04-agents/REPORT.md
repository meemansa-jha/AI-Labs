# Laboratory Report — Constructing a Goal-Based Agent

**Warehouse Navigation** · Artificial Intelligence

| File | Purpose |
|------|---------|
| `warehouse_agent.py` | the goal-based agent (BFS planner) for the warehouse map |
| `test_agent.py` | validation tests (valid path, shortest-path, failure reporting) |
| `experiment_transcript.txt` | captured output |

```bash
python3 test_agent.py        # validation tests
python3 warehouse_agent.py   # run the agent on the warehouse map
```

The warehouse (loading bay `S`, dispatch `G`, shelving `#`, free space `.`):

```
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

---

## Task 1 — Understanding the problem

1. **What is the environment?** A static 2-D grid warehouse of free cells and fixed shelving (obstacles). It is fully observable, deterministic, discrete, and unchanging while the agent acts — a single-agent, sequential environment.

2. **What is the goal of the agent?** To move the package from the loading bay `S` to the dispatch area `G` along a collision-free path (ideally the shortest one).

3. **What actions are available?** Move one grid cell **Up, Down, Left, or Right**. A move is available only if the destination cell is on the grid and is not shelving.

4. **What information must the agent maintain to choose its next action?** Its current position; the warehouse layout (where obstacles are); the goal location; and, for planning, the frontier of cells still to explore plus a record of how each discovered cell was reached (parent pointers) so a path can be reconstructed and cells are not re-visited.

5. **Why is this a goal-based agent, not a simple reflex agent?** A simple reflex agent maps the current percept directly to an action via fixed condition–action rules and has no notion of *where it is trying to go*. This agent has an **explicit goal** (`G`) and chooses actions by reasoning about which sequence of moves will *achieve* that goal — it plans a path rather than reacting cell-by-cell. Faced with a dead-end, a reflex agent would have no basis to back out; the goal-based agent's search naturally explores alternatives because it evaluates actions by whether they lead toward the goal.

**Think About It — if the warehouse were twice as large.** BFS would still be *correct* (it always finds a shortest path and reports failure when none exists), but it would become *expensive*: its time and memory grow with the number of free cells, since it can expand a large fraction of the grid. For much larger warehouses an **informed** search such as A\* with a Manhattan-distance heuristic would be preferable — it expands far fewer cells by steering toward the goal (demonstrated in the Search lab). Additional difficulties at scale: memory for the frontier and the visited set, and, if the layout or other vehicles change over time, the need to re-plan (dynamic environment) rather than plan once.

---

## Task 2 — Designing the agent

Components and how they interact:

- **Environment:** the grid of cells (`.` free, `#` obstacle).
- **Current state:** the agent's `(row, col)` position.
- **Goal:** the cell `G`; the goal test is `state == goal`.
- **Available actions:** Up / Down / Left / Right, filtered to those landing on a passable cell.
- **Decision-making component:** a BFS planner that searches the state space for a path from the current state to the goal.

Block diagram (how they interact):

```
            perceive layout + position
   ENVIRONMENT  ───────────────────────────▶  AGENT STATE  (current cell)
   (grid, #)                                        │
                                                    ▼
                                           ┌──────────────────┐
                              GOAL  ─────▶ │  DECISION MAKING  │
                            (cell G)       │  (BFS search over │
                                           │   actions toward  │
                                           │      the goal)    │
                                           └────────┬─────────┘
                                                    │  chosen action sequence
                                                    ▼
                           action (Up/Down/Left/Right) changes position
   ENVIRONMENT  ◀───────────────────────────────────┘
```

The agent perceives the environment to build its model of passable cells, combines its current state with the goal in the decision-making component to plan a path, then executes the resulting actions, each of which updates its position in the environment.

---

## Task 3 — Prompt engineering and results

**Prompt used (a specification, not "write a program"):**

> Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem shown above. The program should represent the warehouse as a 2-D grid, determine a collision-free path from `S` to `G`, avoid all obstacles, print either the path found or a suitable message if no path exists, and explain the search algorithm chosen and why it is appropriate. The agent may move up/down/left/right by one cell.

**Result on the warehouse map:**

```
start S = (1, 1)
goal  G = (1, 19)
Path found.  length = 20 moves, states expanded = 59

Right -> Right -> Right -> Down -> Right -> Right -> Right -> Up -> Right -> ... -> Right

#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

The agent routes down into row 2 to get around the shelf at `(1,6)`, back up, then straight across the top to `G` — a 20-move shortest path, expanding 59 cells.

**1. Did the LLM generate a working program on the first attempt?** The first draft ran and produced a path, but it had two issues typical of LLM output: it used a `visited` set without recording parent pointers correctly on the first version (so path reconstruction was off by one), and "path found" printed the reversed cell list without the human-readable action names. Both were caught by reading the code and by the validation tests, then fixed.

**2. If not, how can you improve your prompt?** By specifying the *outputs* precisely: "report the path as a sequence of named actions (Up/Down/Left/Right), the path length, and the number of states expanded; if no path exists, print a clear failure message and do not loop." Specifying the behaviour and the exact report removes the ambiguity that produced the rough first draft.

**3. What search algorithm did the LLM choose?** Breadth-first search (BFS).

**4. Why do you think the LLM selected this algorithm?** Because the problem is an unweighted (unit-cost) shortest-path problem on a grid, and BFS is the standard, simplest algorithm that is both *complete* (finds a path if one exists) and *optimal* (finds a shortest path) on unit-cost graphs. It is the textbook default for "find a collision-free path on a grid," and it is easy to implement correctly with a queue and a visited set — so it is the natural choice for an LLM asked for a clear, correct baseline. (If the prompt had stressed efficiency on a large map, A\* with a Manhattan heuristic would have been the better pick, as shown in the Search lab.)

---

## Validation (working output ≠ validated algorithm)

`test_agent.py` encodes properties that must hold:

```
[PASS] a path is found
[PASS] path starts at S / ends at G
[PASS] every step is one legal collision-free move
[PASS] BFS path length equals an independent flood-fill shortest distance
[PASS] walled-off goal -> no path (terminates, no infinite loop)
```

The flood-fill cross-check independently re-derives the shortest distance to `G` and confirms BFS returned a shortest path; the walled-off map confirms the agent reports failure rather than looping forever. These are the same AI-engineering checks used throughout the other labs: a program that prints a plausible path has not been shown to be correct until its output is checked against properties we know must hold.
