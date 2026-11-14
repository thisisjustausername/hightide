"""
Multiprocessing Phase Predictor
===============================

This script runs the phase predictor in parallel across multiple GPUs and processes.
It uses a queue to distribute work items (from START to END) to worker processes, each of which runs a prediction pipeline on a specified GPU.
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

from src.phase_predictor.phase_predictor import prediction_pipeline

START, END = 0, 8984
NUM_GPUS = 2
PROCS_PER_GPU = 8

@dataclass
class PredictionPipelineParameters:
    time_steps: int = 7
    image_shape: tuple = (256, 256)
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    learning_rate: float = 10e-3
    early_stopping: int = 30
    steps: int = 2000
    verbose: bool = False

def worker(gpu_id, queue):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    while True:
        i = queue.get()
        if i is None:
            break
        try:
            prediction_pipeline(
                **PredictionPipelineParameters().__dict__,
                item=i,
                device=device,
            )
        except Exception:
            print(f"[gpu {gpu_id}] item {i} failed:\n", flush=True)
            with open("failed_items.txt", "a") as f:
                f.write(f"{i}\n")


def main():
    n_workers = NUM_GPUS * PROCS_PER_GPU
    queue = mp.Queue()

    for i in range(START, END):
        queue.put(i)
    for _ in range(n_workers):
        queue.put(None)

    procs = []
    for w in range(n_workers):
        gpu_id = w % NUM_GPUS
        p = mp.Process(target=worker, args=(gpu_id, queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
