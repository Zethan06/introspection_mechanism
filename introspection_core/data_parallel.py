"""Multi-GPU runner/worker orchestration.

Process-level: the runner spawns one worker subprocess per GPU through
``CUDA_VISIBLE_DEVICES`` and waits for all of them; each worker loads its own
model and processes its shard.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    command: list[str]
    env: dict[str, str]


def parse_gpu_list(gpus: str) -> list[str]:
    values = [value.strip() for value in gpus.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one GPU id is required")
    return values


def parse_gpu_groups(gpus: str) -> list[str]:
    """Parse semicolon-separated worker GPU groups.

    ``"0,1;2,3"`` creates two workers, each with two visible GPUs. A plain
    comma-separated list retains the historical one-GPU-per-worker behavior.
    """
    if ";" not in gpus:
        return parse_gpu_list(gpus)
    groups = [group.strip() for group in gpus.split(";") if group.strip()]
    if not groups or any(not parse_gpu_list(group) for group in groups):
        raise ValueError("At least one GPU group is required")
    return groups


def shard_items(items: Iterable[T], worker_id: int, num_workers: int) -> list[T]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must be in [0, num_workers)")
    return [item for idx, item in enumerate(items) if idx % num_workers == worker_id]


def build_worker_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
    env["PYTHONUNBUFFERED"] = "1"
    # Prevent OpenBLAS/MKL/OMP from spawning large thread pools when multiple
    # worker processes launch simultaneously — they share the system RLIMIT_NPROC
    # and N_workers × 64 threads easily exhausts it, causing a crash that
    # surfaces as KeyboardInterrupt inside scipy.linalg._fblas.
    # Force these values: the login environment on shared GPU hosts may
    # already export a large thread count, in which case setdefault would
    # preserve it and four lane workers can exceed RLIMIT_NPROC at import.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"  # HF tokenizers Rayon thread pool
    return env


def run_worker_pool(worker_specs: list[WorkerSpec], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()
    threads = []
    processes: list[tuple[WorkerSpec, subprocess.Popen[str]]] = []

    def stream_output(spec: WorkerSpec, process: subprocess.Popen[str]) -> None:
        log_path = log_dir / f"{spec.name}.log"
        with log_path.open("w") as log_handle:
            assert process.stdout is not None
            for line in process.stdout:
                rendered = f"[{spec.name}] {line}"
                with print_lock:
                    print(rendered, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()

    def stop_workers() -> None:
        running = [process for _spec, process in processes if process.poll() is None]
        for process in running:
            process.terminate()
        for process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for process in running:
            process.wait()

    previous_handlers: dict[int, object] = {}

    def stop_on_parent_signal(signum, frame) -> None:
        del frame
        raise KeyboardInterrupt(f"worker runner received signal {signum}")

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGHUP, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop_on_parent_signal)

    failures: list[tuple[str, int]] = []
    try:
        for spec in worker_specs:
            with print_lock:
                print(
                    f"[runner] starting {spec.name}: {' '.join(spec.command)}",
                    flush=True,
                )
            process = subprocess.Popen(
                spec.command,
                env=spec.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append((spec, process))
            thread = threading.Thread(
                target=stream_output,
                args=(spec, process),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        remaining = set(range(len(processes)))
        while remaining:
            for index in tuple(remaining):
                spec, process = processes[index]
                return_code = process.poll()
                if return_code is None:
                    continue
                remaining.remove(index)
                if return_code != 0:
                    failures.append((spec.name, return_code))
            if failures:
                stop_workers()
                break
            if remaining:
                time.sleep(0.1)
    except BaseException:
        stop_workers()
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        for thread in threads:
            thread.join()
        for _spec, process in processes:
            if process.stdout is not None:
                process.stdout.close()

    if failures:
        rendered = ", ".join(f"{name}={return_code}" for name, return_code in failures)
        raise RuntimeError(f"Worker failure(s): {rendered}")

    print("[runner] all workers finished", flush=True)


def current_python() -> str:
    return sys.executable
