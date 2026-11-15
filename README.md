# Rule erosion in tool-using agents

**The question.** An agent is given explicit rules — read a file before deleting it, and
never delete an important one — and then does ordinary cleanup work, round after round, in
one session. Every early round it handles correctly. Does it still follow the rules at the
end of the session, or does benign precedent alone erode them?

We call the behaviour **rule erosion**: the agent follows the rules in every earlier round,
then breaks them. Nothing in the context ever escalates and no prior round contains an
example of rule-breaking, so any erosion comes from repetition and routine, not from bad
examples.

## What is here

- [`multiturn/`](multiturn/) — the whole experiment: environment, runner, checker,
  transcript viewer and review loop. Its [README](multiturn/README.md) describes the design,
  the metrics and how to read an episode.
- `serve.py` — starts a local vLLM server for the model under test (used by the notebook).
- `Untitled2.ipynb` — the Colab notebook that serves a model and runs episodes one at a
  time for review.

## Running it

A vLLM server must be serving the model (the notebook does this, or see `serve.py`). Then:

```
python multiturn/run_pilot.py --model qwen3-1.7b --rounds 0 1 2 3 4 5 --episodes 10 \
    --out results/multiturn/qwen3-1.7b
```

or, one reviewed episode at a time, from a notebook:

```python
from multiturn.review import run_and_review, review_summary
run_and_review(k=3, model="qwen3-1.7b", out="results/multiturn/qwen3-1.7b")
review_summary("results/multiturn/qwen3-1.7b")
```
