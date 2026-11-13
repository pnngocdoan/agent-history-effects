"""Step 2. Build the three conditions. Only the text added to the task changes; the weights never do.

  BASE   the risk is named, no history      "<risky action> may lead to <harm>."
  H2     BASE plus 2 harmless risky histories
  H5     BASE plus 5

A history is one of the agent's own earlier runs in which it took the risky action and no harm
followed. Each is written as "Previous experience i:", the run, then "Outcome:" and its outcome line.
H2 uses histories 1 and 2, H5 uses 1 to 5, so the short history is inside the long one.

  python scripts/2_build_conditions.py                  # from data/tasks.json and data/histories.xlsx
  python scripts/2_build_conditions.py --lengths 2 5    # the history lengths to build

Writes data/conditions/{BASE,H2,H5}.json (the same format as the task file, with the block added to
the instruction) and data/conditions/added_text.json, which holds exactly what each task got.
Prints one full prompt per condition so the text can be checked by eye.
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import audit_sheet as sheet


def risk_sentence(p_line):
    """The p line with its number taken out: '... has a {p}% probability of X.' -> '... may lead to X.'"""
    out, n = re.subn(r"\s+(?:has|carries|poses|bears|involves)?\s*an?\s+\{p\}%\s+(?:probability|chance)\s+of\s+",
                     " may lead to ", p_line)
    return out if n == 1 and "{p}" not in out else None


def history_block(items, k):
    return "\n\n".join(f"Previous experience {i + 1}:\n{text}\nOutcome: {outcome}"
                       for i, (text, outcome) in enumerate(items[:k]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="data/tasks.json")
    ap.add_argument("--sheet", default="data/histories.xlsx")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2, 5], help="history lengths (default: 2 5)")
    ap.add_argument("--out", default="data/conditions")
    a = ap.parse_args()

    tasks = json.loads(Path(a.tasks).read_text())
    tabs = sheet.read_tabs(a.sheet)
    p_lines = {str(r["task"]): r["p_line"] for r in tabs["p_lines"] if r.get("p_line")}
    pool = {}
    for r in sorted(tabs["histories"], key=lambda r: (sheet.sort_key(r["task"]), float(r["order"]))):
        if r.get("outcome_line"):
            pool.setdefault(str(r["task"]), []).append((r["history_text"], r["outcome_line"]))

    risks = {}
    for t in tasks:
        tid = str(t["id"])
        if tid not in p_lines:
            sys.exit(f"task {tid}: no p line in {a.sheet}")
        r = risk_sentence(p_lines[tid])
        if not r:
            sys.exit(f"task {tid}: cannot take the number out of the p line; it must read '... has a {{p}}% probability of ...'")
        risks[tid] = r

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    added, samples = {}, {}

    def write(name, block_for):
        rows = []
        for t in tasks:
            tid = str(t["id"])
            block = block_for(tid)
            if block is None:
                print(f"task {tid}: too few histories for {name}; left out")
                continue
            new = dict(t)
            new["instruction"] = t["instruction"] + "\n\n" + block
            rows.append(new)
            added.setdefault(name, {})[tid] = block
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        samples[name] = rows[0] if rows else None
        print(f"{name}: {len(rows)} tasks")

    write("BASE", lambda tid: "### Known risk\n" + risks[tid])
    for k in a.lengths:
        write(f"H{k}", lambda tid, k=k: None if len(pool.get(tid, [])) < k else
              "### Known risk\n" + risks[tid] + "\n\n### Previous relevant history\n" + history_block(pool[tid], k))

    (out / "added_text.json").write_text(json.dumps(added, indent=1, ensure_ascii=False))
    for name, row in samples.items():
        if row:
            print(f"\n=== sample {name}, task {row['id']} ===\n{row['instruction'][:1500]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
