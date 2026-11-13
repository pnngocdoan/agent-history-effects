"""Shared runner: GPU discovery, vLLM servers, and one agent job per task.

Every script that needs the model loaded once imports this. The job itself is run by the
benchmark's own eval.py inside a private working copy, so its results are the benchmark's.

Taken from the pipeline used to collect the histories (see README, Provenance).
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark" / "Agent-SafetyBench"


# ---------- hardware ----------
def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        return 0


def ram_available_gb():
    """RAM the machine can still hand out, in GB (Linux /proc/meminfo)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def cpu_times():
    """(idle, total) CPU time counters from /proc/stat, to work out how busy the CPU was between two readings."""
    try:
        f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return f[3] + f[4], sum(f)
    except Exception:
        return None


def jobs_that_fit_in_ram(avail_gb, per_job_gb, reserve_gb):
    """How many jobs fit in the RAM that is free, keeping a reserve so nothing gets killed."""
    if avail_gb is None:
        return 10 ** 6
    return max(1, int((avail_gb - reserve_gb) / per_job_gb))


class GpuSampler(threading.Thread):
    """Samples nvidia-smi (and the vLLM gauges, if given) every few seconds.

    Keeps run totals (peak memory, average utilization per GPU) and, between calls to
    new_batch() and batch_summary(), the same numbers for just that batch.
    """

    def __init__(self, metrics_urls=(), every=5.0):
        super().__init__(daemon=True)
        self.every, self.stop_flag, self.metrics_urls = every, threading.Event(), list(metrics_urls)
        self.peak, self.utils = {}, {}
        self.new_batch()

    def new_batch(self):
        self.b_utils, self.b_wait, self.b_cache, self.b_mem = [], 0.0, 0.0, 0
        self.b_ram_min, self.b_cpu = float("inf"), []
        self._cpu_prev = cpu_times()

    def batch_summary(self):
        ram = ram_available_gb()          # one more reading at the end, so a short batch still has numbers
        if ram is not None:
            self.b_ram_min = min(self.b_ram_min, ram)
        return dict(avg_util_pct=round(sum(self.b_utils) / len(self.b_utils)) if self.b_utils else None,
                    peak_waiting=self.b_wait, peak_cache_usage=round(self.b_cache, 3), peak_mem_mib=self.b_mem,
                    min_ram_free_gb=round(self.b_ram_min, 1) if self.b_ram_min != float("inf") else None,
                    avg_cpu_pct=round(sum(self.b_cpu) / len(self.b_cpu)) if self.b_cpu else None)

    def run(self):
        while not self.stop_flag.is_set():
            ram = ram_available_gb()
            if ram is not None:
                self.b_ram_min = min(self.b_ram_min, ram)
            now = cpu_times()
            if now and self._cpu_prev and now[1] > self._cpu_prev[1]:
                self.b_cpu.append(100 * (1 - (now[0] - self._cpu_prev[0]) / (now[1] - self._cpu_prev[1])))
            self._cpu_prev = now
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, check=True).stdout
                for line in out.strip().splitlines():
                    i, u, m = [int(float(x)) for x in line.split(",")]
                    self.peak[i] = max(self.peak.get(i, 0), m)
                    self.utils.setdefault(i, []).append(u)
                    self.b_utils.append(u)
                    self.b_mem = max(self.b_mem, m)
            except Exception:
                pass
            for url in self.metrics_urls:
                g = vllm_gauges(url)
                if g:
                    self.b_wait = max(self.b_wait, g["waiting"])
                    self.b_cache = max(self.b_cache, g["cache_usage"])
            self.stop_flag.wait(self.every)

    def finish(self):
        self.stop_flag.set()
        self.join(timeout=10)
        return {i: dict(peak_mem_mib=self.peak[i], avg_util_pct=round(sum(self.utils[i]) / len(self.utils[i]))) for i in self.peak}


def vllm_gauges(metrics_url):
    """Read vLLM's own gauges: requests waiting for room, and how full the conversation space is (0 to 1).

    nvidia-smi cannot show this: vLLM reserves most of the GPU memory when it starts, so the
    memory always looks nearly full.
    """
    try:
        text = urllib.request.urlopen(metrics_url, timeout=5).read().decode()
    except Exception:
        return None
    def val(pattern):
        m = re.search(r"^" + pattern + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else None
    waiting = val(r"vllm:num_requests_waiting")
    cache = val(r"vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)")
    if waiting is None and cache is None:
        return None
    return dict(waiting=waiting or 0.0, cache_usage=cache or 0.0)


# ---------- model servers ----------
def reachable(base, key=""):
    """Any HTTP reply means a server is there; only a rejected key is a problem."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
        urllib.request.urlopen(req, timeout=10).read()
        return None
    except urllib.error.HTTPError as e:
        return f"the server at {base} rejected the API key (HTTP {e.code})" if e.code in (401, 403) else None
    except Exception as e:
        return f"cannot reach the model server at {base}: {e}"


def start_server(hf_id, name, gpu, port, max_len, gpu_mem, log_dir):
    log = log_dir / f"vllm_gpu{gpu}.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", hf_id, "--served-model-name", name,
           "--dtype", "half",  # the T4 has no bf16
           "--max-model-len", str(max_len), "--gpu-memory-utilization", str(gpu_mem),
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--port", str(port)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log


def wait_for_server(proc, port, log, minutes=20):
    """Wait until the server answers, printing its latest log line every 30 s so the wait is visible."""
    t0, shown = time.time(), ""
    for n in range(minutes * 12):
        if proc.poll() is not None:
            sys.exit(f"the model server on port {port} stopped while loading; last lines of {log}:\n" + "".join(log.read_text().splitlines(True)[-15:]))
        if reachable(f"http://127.0.0.1:{port}/v1") is None:
            print(f"  server on port {port} is up after {time.time() - t0:.0f} s")
            return
        if n % 6 == 0:
            lines = [l.strip() for l in log.read_text(errors="replace").replace("\r", "\n").splitlines() if l.strip()]
            last = lines[-1][:160] if lines else "(no output yet: starting up)"
            print(f"  [{time.time() - t0:.0f} s] port {port}: {last}", flush=True)
        time.sleep(5)
    sys.exit(f"the model server on port {port} did not come up in {minutes} min; see {log}")


def reported_concurrency(log):
    """vLLM logs e.g. 'Maximum concurrency for 12,288 tokens per request: 4.50x' once it knows its memory budget."""
    m = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log.read_text())
    return float(m[-1]) if m else None


# ---------- jobs ----------
def run_job(model, k, task, out, greedy, timeout, base_url, api_key):
    part = out / f"run_{k}" / "parts" / str(task["id"])
    result = part / "generated_trajectories.json"
    if result.exists():
        return k, task["id"], 0.0, "skipped"
    part.mkdir(parents=True, exist_ok=True)
    work = out / "_work" / f"run_{k}_task_{task['id']}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # eval.py expects ../environments next to its own folder and writes logs into its working folder.
    shutil.copytree(BENCH / "evaluation", work / "evaluation", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.symlink(BENCH / "environments", work / "environments")
    (work / "task.json").write_text(json.dumps([task]))
    cmd = [sys.executable, "eval.py", "--model_name", model, "--greedy", str(greedy),
           "--data_path", str(work / "task.json"), "--out_dir", str(part.resolve())]
    env = dict(os.environ, RDA_BASE_URL=base_url, RDA_API_KEY=api_key)
    t0 = time.time()
    with open(part / "log.txt", "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=work / "evaluation", env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    secs = time.time() - t0
    shutil.rmtree(work, ignore_errors=True)
    ok = rc == 0 and result.exists()
    return k, task["id"], secs, "ok" if ok else f"FAILED ({'timed out after %d s' % timeout if rc == 'timeout' else 'exit %s' % rc}; see {part / 'log.txt'})"


def merge(out, tasks, repeats, first=1):
    for k in range(first, first + repeats):
        merged, missing = [], []
        for t in tasks:
            f = out / f"run_{k}" / "parts" / str(t["id"]) / "generated_trajectories.json"
            if f.exists():
                merged.extend(json.loads(f.read_text()))
            else:
                missing.append(t["id"])
        (out / f"run_{k}").mkdir(parents=True, exist_ok=True)
        (out / f"run_{k}" / "generated_trajectories.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False))
        print(f"repeat {k} of {repeats}: {len(merged)} of {len(tasks)} tasks saved together in {out}/run_{k}/generated_trajectories.json" + (f"  (MISSING tasks {missing}: their job failed)" if missing else ""))


