"""Step 4. The execution test: the agent does the task with tools, in every condition.

One set of model servers is started, the model is loaded once, and every (condition, task, run) is
a job. Each job is the benchmark's own eval.py on a private copy of its evaluation folder, so the
trajectories are the benchmark's own, with nothing of ours in the loop.

  python scripts/4_run_agent.py --model Qwen/Qwen3-8B-AWQ --out results/qwen3-8b/pilot --runs 3

  --runs 1   greedy, one run per task (a quick pilot)
  --runs 3   three sampled runs per task, so each task's unsafe rate is 0, 1/3, 2/3 or 1

The prompt is the condition file exactly as step 2 wrote it: nothing is added here, so the agent's
own thinking (<think>, with a thinking model) is what it produced while doing the task.

Writes <out>/agent/<condition>/run_<k>/generated_trajectories.json. Jobs that already have a result
are skipped, so a run that stops can be started again.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import runner

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id, e.g. Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--conditions", default="data/conditions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=1, help="runs per task and condition (default 1)")
    ap.add_argument("--greedy", type=int, help="1 = temperature 0, 0 = sampling (default: greedy only when --runs 1)")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--job-timeout", type=int, default=1800)
    ap.add_argument("--job-ram-gb", type=float, default=0.8)
    ap.add_argument("--ram-reserve-gb", type=float, default=3.0)
    a = ap.parse_args()
    greedy = a.greedy if a.greedy is not None else int(a.runs == 1)

    out = Path(a.out).resolve() / "agent"
    files = {"BASE": Path(a.conditions) / "BASE.json"}
    for f in sorted(Path(a.conditions).glob("H*.json")):
        files[f.stem] = f
    jobs = []
    for k in range(1, a.runs + 1):
        for name, f in files.items():
            for t in json.loads(f.read_text()):
                jobs.append((name, k, out / name, t))
    print(f"{len(files)} conditions, {a.runs} run(s) per task, {len(jobs)} jobs; greedy={greedy}")

    gpus = runner.gpu_count()
    if not gpus:
        sys.exit("no GPU found (nvidia-smi)")
    served = a.model.split("/")[-1].lower()
    log_dir = out / "_servers"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs, servers = [], []
    t0 = time.time()
    try:
        for g in range(gpus):
            procs.append(runner.start_server(a.model, served, g, 8000 + g, a.max_len, a.gpu_mem, log_dir) + (8000 + g,))
        print(f"loading {a.model} once on {gpus} GPU(s); logs in {log_dir}")
        for proc, log, port in procs:
            runner.wait_for_server(proc, port, log)
            servers.append((f"http://127.0.0.1:{port}/v1", "none"))
        load_s = time.time() - t0
        ram_cap = runner.jobs_that_fit_in_ram(runner.ram_available_gb(), a.job_ram_gb, a.ram_reserve_gb)
        workers = max(1, min(len(jobs), ram_cap, (os.cpu_count() or 1) * 8))
        print(f"running {len(jobs)} jobs, {workers} at a time (limited by RAM and CPU cores)")
        t1 = time.time()
        failed, done = [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(runner.run_job, f"api:{served}", k, t, folder, greedy, a.job_timeout,
                                 *servers[i % len(servers)]): (f"{name} run {k}", t["id"])
                       for i, (name, k, folder, t) in enumerate(jobs)}
            for fu in as_completed(futures):
                label, tid = futures[fu]
                _, _, secs, status = fu.result()
                done += 1
                if status not in ("ok", "skipped"):
                    failed.append((label, tid, status))
                print(f"[{done}/{len(jobs)}] {label} task {tid}: {status} ({secs:.0f} s)")
        run_s = time.time() - t1
    finally:
        for proc, _, _ in procs:
            try:
                os.killpg(proc.pid, 15)
            except Exception:
                pass

    for name, f in files.items():
        tasks = json.loads(f.read_text())
        print(f"{name}: ", end="")
        runner.merge(out / name, tasks, a.runs)
    print(f"done: model load {load_s:.0f} s, {len(jobs)} jobs {run_s:.0f} s, {len(failed)} failed")
    for x in failed:
        print("FAILED:", *x)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
