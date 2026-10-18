import torch
import math
import logging
import os
import subprocess
import threading


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = total_steps * warmup_ratio
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            cosine_step = self.last_epoch - self.warmup_steps
            cosine_steps = self.total_steps - self.warmup_steps
            return [
                base_lr * (1 + math.cos(math.pi * cosine_step / cosine_steps)) / 2
                for base_lr in self.base_lrs
            ]


class NvidiaSmiMemoryMonitor:
    """Poll nvidia-smi and keep the peak GPU memory in MB.

    The primary path matches the current Python PID in compute-apps. Some cluster
    environments expose different PIDs or return empty compute-apps output, so a
    GPU-level fallback is used for the GPUs listed in CUDA_VISIBLE_DEVICES.
    """

    def __init__(self, pid=None, interval=1.0):
        self.pid = str(pid or os.getpid())
        self.interval = interval
        self.peak_memory_mb = 0.0
        self.mode = "process"
        self._warned_fallback = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()
        self._sample()

    def _run(self):
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval)

    def _sample(self):
        memory_mb = self._query_process_memory_mb()
        if memory_mb is None:
            memory_mb = self._query_visible_gpu_memory_mb()
            if memory_mb is not None:
                self.mode = "visible_gpu"
                if not self._warned_fallback:
                    logging.warning(
                        "nvidia-smi compute-apps did not report PID %s; "
                        "falling back to visible GPU memory.used.",
                        self.pid,
                    )
                    self._warned_fallback = True

        if memory_mb is not None:
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

    def _query_process_memory_mb(self):
        for field in ("used_memory", "used_gpu_memory"):
            result = self._run_nvidia_smi(
                [
                    f"--query-compute-apps=pid,{field}",
                    "--format=csv,noheader,nounits",
                ]
            )
            if result is None:
                continue

            memory_mb = 0.0
            matched = False
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2 or parts[0] != self.pid:
                    continue
                try:
                    memory_mb += float(parts[1])
                except ValueError:
                    continue
                matched = True
            if matched:
                return memory_mb
        return None

    def _query_visible_gpu_memory_mb(self):
        gpu_ids = self._visible_gpu_ids()
        if gpu_ids:
            total = 0.0
            found = False
            for gpu_id in gpu_ids:
                result = self._run_nvidia_smi(
                    [
                        f"--id={gpu_id}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ]
                )
                if result is None:
                    continue
                value = self._sum_memory_lines(result.stdout)
                if value is not None:
                    total += value
                    found = True
            if found:
                return total

        result = self._run_nvidia_smi(
            ["--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        if result is None:
            return None
        return self._sum_memory_lines(result.stdout)

    def _visible_gpu_ids(self):
        value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not value or value == "NoDevFiles":
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def _sum_memory_lines(self, stdout):
        total = 0.0
        found = False
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                total += float(line.split()[0])
            except ValueError:
                continue
            found = True
        return total if found else None

    def _run_nvidia_smi(self, args):
        try:
            result = subprocess.run(
                ["nvidia-smi", *args],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        return result
