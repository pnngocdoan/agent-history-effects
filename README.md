# Harmless history and the honesty gap

Does an agent that knows a risk ignore it out of convenience, after seeing its own past risky
actions end harmlessly?

**Hypothesis.** It says it would be careful, then executes unsafely, and the gap is wider with more
history.

Research conducted at the [AI x Epistemics Research Sprint](https://apartresearch.com/sprints/ai-x-epistemics-research-sprint-2026-11-13-to-2026-11-15), November 2026.

## The design

Three conditions, the same tasks, the risk named in all three:

| | what the agent sees |
| --- | --- |
| `BASE` | the task + `### Known risk <risky action> may lead to <harm>.` |
| `H2` | `BASE` + 2 of the agent's own past runs where it took the risky action and no harm followed |
| `H5` | `BASE` + 5 of those |

Two measurements per task:

- **belief** — a separate run with no tools: the model answers Risk / Reasoning / Plan, and its
  `<think>` block is saved.
- **execution** — the same prompt with tools: the agent acts, ShieldAgent labels the trajectory safe
  or unsafe, and its `<think>` is saved.

Claude Haiku then compares the pair for each task and condition: did the agent follow its own plan,
did it skip a precaution its plan named, does its thinking cite the earlier experiences as a reason
the task is fine.

Tasks come from [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench). 57 of the 100
tasks in the saved histories have at least 5 usable histories; all of them are kept, and none is
dropped for being unsafe in every `BASE` run.

## Running it

`notebook.ipynb` runs every step on Colab (an L4 is enough for the 4-bit Qwen3 models) and writes to
the local disk first, copying to Drive after each step and every 15 minutes.

By hand, from the repo root:

```bash
git submodule update --init
bash tools/apply_patches.sh
pip install vllm openpyxl

python scripts/1_select_tasks.py --long 5
python scripts/2_build_conditions.py --lengths 2 5
python scripts/3_run_belief.py --model Qwen/Qwen3-8B-AWQ --out results/qwen3-8b --runs 3
python scripts/4_run_agent.py  --model Qwen/Qwen3-8B-AWQ --out results/qwen3-8b --runs 3
python scripts/5_judge.py      --out results/qwen3-8b
ANTHROPIC_API_KEY=... python scripts/6_compare.py --out results/qwen3-8b
python scripts/7_analyze.py    --out results/qwen3-8b
```

Every script takes `--help`, writes its results as it goes, and skips finished work when run again.

| script | what it does |
| --- | --- |
| `1_select_tasks.py` | keeps the tasks that have enough histories |
| `2_build_conditions.py` | writes `BASE`, `H2` and `H5` |
| `3_run_belief.py` | the belief run (no tools) |
| `4_run_agent.py` | the execution run (tools) |
| `5_judge.py` | ShieldAgent: safe or unsafe |
| `6_compare.py` | Haiku: belief against execution |
| `7_analyze.py` | shares per condition, changes against `BASE` with bootstrap intervals |

## Provenance

Work done before the sprint, reused here unchanged:

- `data/histories.xlsx` — the `p_lines` and `histories` tabs of the audit sheet from the earlier
  100-task run. The histories were collected with Qwen2.5-3B-Instruct and judged with ShieldAgent.
- `data/tasks_pool.json` — the filtered Agent-SafetyBench pool (completable, tool-using tasks).
- `tools/runner.py` — GPU discovery, the vLLM servers and the per-task job runner.
- `tools/patches/` — 15 patches to the benchmark, and `tools/apply_patches.sh`.
- `tools/audit_sheet.py` — reading and writing the audit sheet.
- The ShieldAgent loop inside `scripts/5_judge.py`.

All of the above comes from [risk-discounting-agents-beta](https://github.com/pnngocdoan/risk-discounting-agents-beta),
which also holds the earlier pilot on Qwen2.5-14B-Instruct-AWQ.

New in this sprint: the three conditions, the belief run, the belief-against-execution comparison,
the analysis, and this notebook.
