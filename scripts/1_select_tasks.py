"""Step 1. Choose the tasks: those that have enough harmless risky histories.

A task is kept if it has at least 2 usable histories, so it can be compared as BASE against H2.
Step 2 then builds H5 only for the tasks that have 5, and step 7 compares each pair of conditions on
the tasks they share. It reads:

  data/tasks_pool.json   the benchmark tasks (completable, tool-using), as the pipeline filtered them
  data/histories.xlsx    tab "p_lines"   one row per task: the risky action, the harm, the p line
                         tab "histories" one row per usable history: task, order, outcome line, text

  python scripts/1_select_tasks.py                      # keeps tasks with >= 2 histories
  python scripts/1_select_tasks.py --long 5 --limit 20  # only tasks that have 5, or a pilot subset

Writes data/tasks.json and prints how many tasks each history length would give.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import audit_sheet as sheet


def usable_histories(path):
    """{task id: number of histories that have an outcome line}"""
    tabs = sheet.read_tabs(path)
    for name in ("p_lines", "histories"):
        if name not in tabs:
            sys.exit(f"{path} has no tab '{name}'")
    count = {}
    for r in tabs["histories"]:
        if r.get("outcome_line"):
            count[str(r["task"])] = count.get(str(r["task"]), 0) + 1
    return count, {str(r["task"]): r for r in tabs["p_lines"] if r.get("p_line")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/tasks_pool.json")
    ap.add_argument("--sheet", default="data/histories.xlsx")
    ap.add_argument("--long", type=int, default=2, help="the shortest history a task must have to be kept (default 2); H5 is built only for the tasks that have 5")
    ap.add_argument("--limit", type=int, help="keep only the first N tasks (for a pilot)")
    ap.add_argument("--out", default="data/tasks.json")
    a = ap.parse_args()

    pool = {str(t["id"]): t for t in json.loads(Path(a.pool).read_text())}
    count, p_lines = usable_histories(a.sheet)
    print(f"{len(pool)} tasks in the pool, {len(count)} have histories, {len(p_lines)} have a p line\n")
    print("tasks available at each history length:")
    for k in range(2, max(count.values(), default=0) + 1):
        print(f"  k={k}: {sum(1 for c in count.values() if c >= k)}")

    keep = [t for t in sorted(count, key=int) if count[t] >= a.long and t in pool and t in p_lines]
    skipped = [t for t in sorted(count, key=int) if count[t] >= a.long and (t not in pool or t not in p_lines)]
    for t in skipped:
        print(f"task {t}: enough histories but {'not in the pool' if t not in pool else 'no p line'}; skipped")
    if a.limit:
        keep = keep[:a.limit]
    Path(a.out).write_text(json.dumps([pool[t] for t in keep], indent=1, ensure_ascii=False))
    print(f"\nkept {len(keep)} tasks with >= {a.long} histories -> {a.out}")


if __name__ == "__main__":
    main()
