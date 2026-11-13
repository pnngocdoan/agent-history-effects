"""Step 5. Judge every run with ShieldAgent, the benchmark's own safety judge.

ShieldAgent reads a trajectory and says safe or unsafe. It is loaded once and used on every
condition and run, so it must be started after the agent's model servers have stopped (step 4 ends).

  python scripts/5_judge.py --out results/qwen3-8b/pilot

ShieldAgent is shown the whole dialogue, the user turn included, so it would otherwise read the
history block that H2 and H5 carry in their instruction: a trajectory of the agent doing the risky
thing, and the line saying no harm came of it. The judge's input would then differ between the
conditions for a reason that has nothing to do with what the agent did. So before judging, the
"### Previous relevant history" block is cut from every user message. The "### Known risk" line
stays, because every condition has it and it is therefore balanced. Pass --keep-history for the old
behaviour.

Reads  <out>/agent/<condition>/run_<k>/generated_trajectories.json   (written by step 4)
Writes <out>/judge_input/<condition>/run_<k>/generated_trajectories.json  (what the judge is shown)
       <out>/judged/<condition>/run_<k>/

A folder that already has results is skipped. By default the batch size starts at --max-batch and
halves whenever the GPU runs out of memory, then grows back for the next folder.
"""
import argparse, datetime, glob, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "benchmark" / "Agent-SafetyBench" / "score"
HISTORY = "### Previous relevant history"


def strip_history(node, counter):
    """Cut the history block out of every user message, wherever it sits in the file.

    The benchmark's own reader walks more than one key ("dialog" and the output key), so this
    walks the whole structure instead of guessing which one holds the instruction.
    """
    if isinstance(node, dict):
        content = node.get("content")
        if node.get("role") == "user" and isinstance(content, str) and HISTORY in content:
            node["content"] = content.split(HISTORY)[0].rstrip()
            counter[0] += 1
        for v in node.values():
            strip_history(v, counter)
    elif isinstance(node, list):
        for v in node:
            strip_history(v, counter)
    return node


def judge_input_for(traj, dest, keep_history):
    """The folder the judge is pointed at: the trajectories as they are, or with the history cut."""
    src = traj / "generated_trajectories.json"
    if keep_history:
        return traj, 0
    dest.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text())
    counter = [0]
    strip_history(data, counter)
    (dest / "generated_trajectories.json").write_text(json.dumps(data, ensure_ascii=False))
    for extra in traj.glob("*"):
        if extra.name != "generated_trajectories.json" and extra.is_file():
            shutil.copy2(extra, dest / extra.name)
    return dest, counter[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the results folder given to step 4")
    ap.add_argument("--model-path", default="thu-coai/ShieldAgent")
    ap.add_argument("--label", help="name used in the output files (default: the folder's name)")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = automatic (default): start at --max-batch, halve on out-of-memory")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--explain-tokens", type=int, default=60)
    ap.add_argument("--keep-history", action="store_true",
                    help="show the judge the history block too (the old behaviour; not comparable across conditions)")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    label = a.label or out.name
    todo = []
    for d in sorted((out / "agent").glob("*/run_*"), key=lambda p: (int(p.name.split("_")[1]), p.parent.name)):
        if d.parent.name.startswith("_"):
            continue
        if (d / "generated_trajectories.json").exists():
            todo.append((f"{d.parent.name} {d.name}", d, out / "judged" / d.parent.name / d.name))
    todo = [t for t in todo if not glob.glob(f"{t[2]}/**/*_outputs_results.json", recursive=True)]
    if not todo:
        sys.exit("nothing to judge: no trajectory files, or everything is judged already")
    print(f"judging {len(todo)}: {', '.join(t[0] for t in todo)}")
    if a.keep_history:
        print("--keep-history: the judge sees the history block, so BASE and H* are not judged alike")
    else:
        print(f"the '{HISTORY}' block is cut from user messages before judging")

    os.chdir(SCORE)
    sys.path.insert(0, str(SCORE))
    import eval_with_shield as ev
    import torch

    ev.generation_config["max_new_tokens"] = 10
    ev.ANSWER_ORDER = "answer_first"
    ev.EXPLAIN_TOKENS = a.explain_tokens
    ev.REASONING_TOKENS = 120
    model, tokenizer, device = ev.create_model_tokenizer(a.model_path, a.model_path)
    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    batch = a.batch_size or a.max_batch
    auto = a.batch_size == 0
    print(f"batch size: {batch} ({'automatic, halves on out-of-memory' if auto else 'fixed'})", flush=True)
    for name, traj, judged in todo:
        shown, cut = judge_input_for(traj, out / "judge_input" / traj.parent.name / traj.name, a.keep_history)
        ev.SHIELD_OUT = str(judged)
        os.makedirs(f"{judged}/{label}", exist_ok=True)
        print(f"=== JUDGE {name} ===" + (f" (history cut from {cut} user message(s))" if cut else ""),
              flush=True)
        if auto:
            batch = min(a.max_batch, batch * 2)
        while True:
            try:
                ev.run_each_file("generated_trajectories", str(shown), stamp, label, model, tokenizer, device,
                                 "qwen", "", batch, label)
                break
            except torch.cuda.OutOfMemoryError:
                if not auto or batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"GPU out of memory: retrying {name} with batch size {batch}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
