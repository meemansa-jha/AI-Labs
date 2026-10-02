# AI Laboratory Exercises

Solutions to four Artificial Intelligence laboratory exercises, each framed
around using a Large Language Model (LLM) as an engineering assistant while
retaining responsibility for problem specification, verification, and
interpretation. Every lab contains runnable code, validation tests, a captured
experiment transcript, and a REPORT.md answering all theory and reflection
questions.

| Folder | Lab | What it covers |
|--------|-----|----------------|
| [`01-search-astar`](01-search-astar/) | Search and A* | Warehouse robot navigation as a search problem; A*, BFS, greedy best-first; Manhattan/Euclidean/zero/x2 heuristics; optimality and admissibility |
| [`02-neural-models`](02-neural-models/) | Neural Models | XOR with a 2-2-1 PyTorch net; depth vs nonlinearity; backprop inspection; weight-symmetry failure; sigmoid/tanh/ReLU comparison; three-class softmax extension |
| [`03-bayesian-networks`](03-bayesian-networks/) | Bayesian Networks & Language Models | First- and second-order autoregressive models as Bayesian networks; conditional probability tables from counts; normalisation tests; sampling vs greedy generation |
| [`04-agents`](04-agents/) | Goal-Based Agents | A goal-based warehouse agent using BFS; goal- vs reflex-agent distinction; agent block diagram; prompt engineering |

## Running everything

```bash
cd 01-search-astar && python3 test_search.py && python3 run_experiments.py
cd 02-neural-models && python3 test_xor.py && python3 xor_experiments.py
cd 03-bayesian-networks && python3 test_lm.py && python3 language_model.py
cd 04-agents && python3 test_agent.py && python3 warehouse_agent.py
```

## Requirements

- Python 3
- numpy and torch (only for 02-neural-models); the other three labs use the standard library only.

Install with: `pip install torch numpy`

## Method

Each lab specifies the problem precisely before writing code, uses an LLM to draft
the implementation, then validates the result against properties that must hold —
a shortest path matching BFS, XOR learned on all four points, conditional
distributions summing to 1 — rather than trusting that a program which runs is
correct. Each REPORT.md separates what was designed, what the LLM suggested, what
was accepted, what was changed, and what was tested.
