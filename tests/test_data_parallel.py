"""Tests for process-level worker orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from introspection_core.data_parallel import WorkerSpec, run_worker_pool


class WorkerPoolTests(unittest.TestCase):
    def test_failure_terminates_other_workers_without_waiting_for_them(self) -> None:
        environment = os.environ.copy()
        specs = [
            WorkerSpec(
                name="failure",
                command=[sys.executable, "-c", "raise SystemExit(3)"],
                env=environment,
            ),
            WorkerSpec(
                name="sleeper",
                command=[sys.executable, "-c", "import time; time.sleep(60)"],
                env=environment,
            ),
        ]

        with TemporaryDirectory() as directory:
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "failure=3"):
                run_worker_pool(specs, Path(directory))
            elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5.0)

    def test_parent_sigterm_terminates_running_worker(self) -> None:
        with TemporaryDirectory() as directory:
            log_dir = Path(directory)
            runner_code = "\n".join(
                [
                    "import os, sys, time",
                    "from pathlib import Path",
                    "from introspection_core.data_parallel import WorkerSpec, run_worker_pool",
                    "spec = WorkerSpec(",
                    "    name='sleeper',",
                    "    command=[sys.executable, '-c', "
                    "'import os,time; print(os.getpid(), flush=True); time.sleep(60)'],",
                    "    env=os.environ.copy(),",
                    ")",
                    f"run_worker_pool([spec], Path({str(log_dir)!r}))",
                ]
            )
            runner = subprocess.Popen(
                [sys.executable, "-c", runner_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            worker_pid: int | None = None
            try:
                # The runner imports the package (and torch) before it starts
                # the worker, which can take well over five seconds on a busy node.
                deadline = time.monotonic() + 60.0
                log_path = log_dir / "sleeper.log"
                while time.monotonic() < deadline:
                    if log_path.exists() and log_path.read_text().strip():
                        worker_pid = int(log_path.read_text().strip())
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(worker_pid)

                runner.terminate()
                runner.communicate(timeout=5)
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker_pid, 0)
            finally:
                if runner.poll() is None:
                    runner.kill()
                    runner.wait()
                if runner.stdout is not None:
                    runner.stdout.close()

    def test_successful_workers_finish_and_write_logs(self) -> None:
        spec = WorkerSpec(
            name="success",
            command=[sys.executable, "-c", "print('done', flush=True)"],
            env=os.environ.copy(),
        )

        with TemporaryDirectory() as directory:
            log_dir = Path(directory)
            run_worker_pool([spec], log_dir)
            output = (log_dir / "success.log").read_text()

        self.assertEqual(output, "done\n")


if __name__ == "__main__":
    unittest.main()
