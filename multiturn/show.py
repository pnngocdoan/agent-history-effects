"""Print whole conversations, thinking included, so a pilot can be read rather than summarised.

  python multiturn/show.py --out results/multiturn/qwen3-1.7b --rounds 10 --n 2
  python multiturn/show.py --out ... --rounds 10 --deleted        # only the ones that deleted it
  python multiturn/show.py --out ... --seed 7                     # one episode by seed

Every assistant turn is shown as its <think> block and then what it said, followed by the tools it
called and what came back, so the last round can be read against the rounds that led to it.
"""
import argparse, json, re, textwrap
from pathlib import Path

THINK = re.compile(r"<think>(.*?)</think>", re.S)
W = 96


def wrap(text, indent):
    out = []
    for para in str(text).split("\n"):
        out += textwrap.wrap(para, W, initial_indent=indent, subsequent_indent=indent) or [indent.rstrip()]
    return "\n".join(out)


def render(row):
    lines = [f"{'=' * W}",
             f"rounds={row['rounds']}  seed={row['seed']}  important={row['important_file']}",
             f"deleted the important file: {row['deleted_important']}   asked: {row['asked']}   "
             f"disposable files cleared earlier: {row['junk_deleted_before']}",
             f"{'=' * W}"]
    # the final round is the test: it is the one whose listing holds the IMPORTANT file
    last_round = sum(1 for m in row["messages"] if m.get("role") == "user")
    round_no = 0
    for m in row["messages"]:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            lines += ["", "SYSTEM", wrap(content, "  ")]
        elif role == "user":
            round_no += 1
            head = (f"--- round {round_no}: THE TEST, the IMPORTANT file is in this listing ---"
                    if round_no == last_round else
                    f"--- round {round_no}: harmless, disposable files only ---")
            lines += ["", head, "USER", wrap(content, "  ")]
        elif role == "assistant":
            think = THINK.search(content)
            if think:
                lines += ["", "  THINKING", wrap(think.group(1).strip(), "    | ")]
            said = THINK.sub("", content).strip()
            if said:
                lines += ["  SAID", wrap(said, "    ")]
            for call in m.get("tool_calls") or []:
                fn = call["function"]
                lines.append(f"    -> {fn['name']}({fn.get('arguments', '')})")
        elif role == "tool":
            try:
                data = json.loads(content)
                if isinstance(data.get("data"), list):
                    content = ", ".join(f"{f.get('file_name_or_path')} [{f.get('importance')}]"
                                        for f in data["data"])
                else:
                    content = data.get("message") or json.dumps(data)
            except Exception:
                pass
            lines.append(wrap(f"<- {content}", "       "))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, help="only this condition")
    ap.add_argument("--seed", type=int, help="one episode, by seed")
    ap.add_argument("--deleted", action="store_true", help="only episodes that deleted the important file")
    ap.add_argument("--kept", action="store_true", help="only episodes that did not")
    ap.add_argument("--n", type=int, default=1, help="how many to print (default 1)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in (Path(a.out) / "episodes.jsonl").read_text().splitlines() if l.strip()]
    if a.rounds is not None:
        rows = [r for r in rows if r["rounds"] == a.rounds]
    if a.seed is not None:
        rows = [r for r in rows if r["seed"] == a.seed]
    if a.deleted:
        rows = [r for r in rows if r["deleted_important"]]
    if a.kept:
        rows = [r for r in rows if not r["deleted_important"]]
    if not rows:
        raise SystemExit("no episode matches those filters")
    for row in rows[:a.n]:
        print(render(row))
        print()


if __name__ == "__main__":
    main()
