"""Step 5. Judge every run with ShieldAgent, the benchmark's own safety judge.

ShieldAgent reads a trajectory and says safe or unsafe. It is loaded once and used on every
condition and run, so it must be started after the agent's model servers have stopped (step 4 ends).

  python scripts/5_judge.py --out results/qwen3-8b/pilot

Reads  <out>/agent/<condition>/run_<k>/generated_trajectories.json   (written by step 4)
Writes <out>/judged/<condition>/run_<k>/

A folder that already has results is skipped. By default the batch size starts at --max-batch and
halves whenever the GPU runs out of memory, then grows back for the next folder.
"""
import argparse, datetime, glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "benchmark" / "Agent-SafetyBench" / "score"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the results folder given to step 4")
    ap.add_argument("--model-path", default="thu-coai/ShieldAgent")
    ap.add_argument("--label", help="name used in the output files (default: the folder's name)")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = automatic (default): start at --max-batch, halve on out-of-memory")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--explain-tokens", type=int, default=60)
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
        ev.SHIELD_OUT = str(judged)
        os.makedirs(f"{judged}/{label}", exist_ok=True)
        print(f"=== JUDGE {name} ===", flush=True)
        if auto:
            batch = min(a.max_batch, batch * 2)
        while True:
            try:
                ev.run_each_file("generated_trajectories", str(traj), stamp, label, model, tokenizer, device,
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
