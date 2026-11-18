"""
Wave Dataset Generation
=======================

This module generates a wave image dataset
"""
import os
import shutil
import threading
from contextlib import contextmanager
from multiprocessing import Queue
from typing import Literal, overload

import numpy as np
import torch
from tqdm import tqdm

import src.parameters as params
from datasets import Array3D, Dataset, Features, Value, disable_progress_bars
from src import image_size
from src.dataset.generate_wave import (
    SuperWave,
    WaveGenerator,
)

disable_progress_bars()

NUM_GPUS = 2

def create_dataset(batch: list[SuperWave]) -> Dataset:
    """
    Save a batch of SuperWave objects to disk as a dataset in parquet file format.

    Args:
        batch (list[SuperWave]): A list of SuperWave objects to save.
        path (str): The path where the dataset will be saved.
    Returns:
        None
    """
    scalars = {
        bool: Value("bool"),
        int: Value("int64"),
        float: Value("float32"),
        str: Value("string"),
    }

    features = Features(
        {
            'wave_img': Array3D(dtype='float32', shape=batch[0].wave_img.shape),
            'params': [{
                'resolution_px_per_km': scalars[int],
                'resolution_seconds_per_time_step': scalars[int],
                'angle': scalars[float],
                'wavelength': scalars[float],
                'amplitude': scalars[float],
                'phase': scalars[float],
                'period': scalars[float],
                'alpha': scalars[float],
            }],
            'image_shape': [Value(dtype='string')],
            'time_steps': Value(dtype='int64'),
        }
    )

    def gen():
        for w in batch:
            yield {
                'wave_img': w.wave_img,
                'params': w.clean_params,
                'image_shape': w.image_shape,
                'time_steps': w.time_steps,
            }

    ds = Dataset.from_generator(gen, features=features, cache_dir=os.path.join(params.dataset_base_path, '_temp/cache/huggingface'))
    return ds


def save_batch_parquet(batch: list[SuperWave], path: str) -> None:
    """
    Save a batch of SuperWave objects to disk as a dataset in parquet file format.

    Args:
        batch (list[SuperWave]): A list of SuperWave objects to save.
        path (str): The path where the dataset will be saved.
    Returns:
        None
    """
    ds = create_dataset(batch)
    ds.save_to_disk(path)


@overload
def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int | None,
    match_range: None,
    folder_name: None,
    scale: None = None,
    return_dataset: Literal[True] = True,
) -> list[SuperWave] | list[list[SuperWave]]:
    """Do not save data, just return the dataset"""

@overload
def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int | None,
    match_range: None,
    folder_name: str,
    scale: Literal['normalize'] = 'normalize',
    return_dataset: bool = False,
) -> None:
    """Save the dataset in batches and scale the images using normalize scale."""

@overload
def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int | None,
    match_range: tuple[float, float],
    folder_name: str,
    scale: Literal['cut'] = 'cut',
    return_dataset: bool = False,
) -> None:
    """Save the dataset in batches and scale the images using cut scale."""

def generate_wave_dataset(
    wave_generator: WaveGenerator,
    num_samples: int,
    batch_size: int | None = None,
    match_range: tuple[float, float] | None = None,
    folder_name: str | None = None,
    scale: Literal['normalize', 'cut'] | None = None,
    return_dataset: bool = True,
) -> list[SuperWave] | list[list[SuperWave]] | None:
    """
    Generate a wave image dataset

    Args:
        wave_generator (WaveGenerator): The wave generator to use for generating the dataset
        num_samples (int): The number of samples to generate
        batch_size (int | None): The interval at which to save the dataset, if None, no batch is created. If not None, num_samples mod batch_size MUST be 0. Defaults to None.
        match_range (tuple[float, float] | None): How to scale the wave when converting to Image. Only needed when interval_save is not None. Defaults to None.
        folder_name (str | None): The name of the folder to save the dataset. Only needed when interval_save is not None.
        scale (Literal['normalize', 'cut'] | None): How to scale the wave when converting to Image. Only needed when interval_save is not None. Defaults to None.
        return_dataset (bool): Whether to return the dataset. Setting it to False can save RAM for big datasets. Has to be True for interval_save = False. Defaults to True.

    Raises:
        ValueError: If a combination of parameters is specified, that does not match any of the overload combinations.
        ValueError: If num_waves = 1 and interval_save is not None, then num_waves should be set to lambda : 1 for the same result.
    Returns:
        list[tuple[tuple[WaveParams, ...], Tensor]] | list[tuple[WaveParams, Tensor]]: A list of tuples each containing a tuple of WaveParams objects if num_samples is not set to 1 otherwise a single WaveParams object and a Tensor that is the wave
    """

    if batch_size is not None and num_samples % batch_size != 0:
        raise ValueError("num_samples mod batch_size must be 0 so that each batch has the same amount of samples.")

    avoid_batches = batch_size is None or batch_size == 0
    batch_size: int = (batch_size or num_samples) # if batch_size is None or 0, set it to num_samples so that it generates a single batch
    batches = []
    for _ in range(num_samples // batch_size):
        batch = [wave_generator.generate_super_wave() for _ in range(batch_size)]
        if folder_name is not None:
            save_batch_parquet(batch, os.path.join(params.dataset_base_path, f'{folder_name}/batch_{len(batches)}'))
        if return_dataset:
            batches.append(batch)

    if not return_dataset:
        return None

    if avoid_batches:
        return batches[0]


def create_wave_generator(
resolution_px_per_km: int,
resolution_seconds_per_time_step: int,
angle: tuple[int, int],
wavelength: tuple[int, int],
amplitude: tuple[int, int],
phase: tuple[int, int],
period: tuple[int, int],
alpha: tuple[int, int],
num_waves: tuple[int, int],
time_steps: int,
img_shape: tuple[int, int],
to_cpu: bool,
to_numpy: bool,
device: torch.device | None = None
):
    return WaveGenerator(
        resolution_px_per_km = lambda : resolution_px_per_km,
        resolution_seconds_per_time_step = lambda : resolution_seconds_per_time_step,
        angle = lambda : np.random.uniform(*angle),
        wavelength = lambda : np.random.uniform(*wavelength),
        amplitude = lambda : np.random.uniform(*amplitude),
        phase = lambda : np.random.uniform(*phase),
        period = lambda : np.random.uniform(*period),
        alpha = lambda : np.random.uniform(*alpha),
        num_waves = lambda : np.random.randint(*num_waves),
        time_steps = time_steps,
        img_shape = img_shape,
        to_cpu = to_cpu,
        to_numpy = to_numpy,
        device = device
    )


@contextmanager
def working(counter):
    with counter.get_lock():
        counter.value += 1
    try:
        yield
    finally:
        with counter.get_lock():
            counter.value -= 1

def worker(
    jobs,
    device: torch.device,
    wave_generator_config: dict[str, int | tuple[int, int] | tuple[float, float] | bool],
    num_samples: int,
    folder_name: str | None,
    seed_seq: np.random.SeedSequence,
    busy,
    done
) -> None:

    np_seed, torch_seed = seed_seq.generate_state(2) # two seeds for numpy and torch
    np.random.seed(int(np_seed))
    torch.manual_seed(int(torch_seed))
    torch.cuda.set_device(device)
    wave_generator =create_wave_generator(**(wave_generator_config | {'device': device})) # type: ignore

    save_queue = Queue(maxsize=1)
    errors = []

    def saver() -> None:
        while True:
            item = save_queue.get()
            if item is None:
                return
            try:
                create_dataset(item['dataset']).save_to_disk(item['path'])
                with done.get_lock():
                    done.value += 1
            except Exception as e:
                errors.append((item['path'], str(e)))
            finally: del item

    t = threading.Thread(target=saver)
    t.start()

    try:
        while True:
            item = jobs.get()
            if item is None:
                break
            with working(busy):
                try:
                    batch: list[SuperWave] = generate_wave_dataset(
                        wave_generator=wave_generator,
                        num_samples=num_samples,
                        batch_size=None,
                        match_range=None,
                        folder_name=None,
                        scale=None,
                        return_dataset=True) # type: ignore

                    save_queue.put({'dataset': batch, 'path': os.path.join(params.dataset_base_path, f'{folder_name}/batch_{item}')})
                except Exception as e:
                    print(f"Error in worker (batch {item}): {e}")
    finally:
        save_queue.put(None)
        t.join()
    if errors:
        print(f"Errors in saver thread: {errors}")

def monitor(stop, queues: dict, groups: dict, done, total: int) -> None:
    """queues: name -> (queue, maxsize); groups: name -> (busy_counter, num_workers)."""
    rows = {}
    pos = 0
    for name, (_, maxsize) in queues.items():
        rows[('q', name)] = tqdm(total=maxsize, desc=f"queue {name:<8}", position=pos,
                                 bar_format="{desc} |{bar:25}| {n_fmt}/{total_fmt}")
        pos += 1
    for name, (_, n) in groups.items():
        rows[('w', name)] = tqdm(total=n, desc=f"busy  {name:<8}", position=pos,
                                 bar_format="{desc} |{bar:25}| {n_fmt}/{total_fmt}")
        pos += 1
    rows['done'] = tqdm(total=total, desc="saved batches ", position=pos)

    while not stop.wait(0.5):
        for name, (q, _) in queues.items():
            rows[('q', name)].n = q.qsize()
        for name, (counter, _) in groups.items():
            rows[('w', name)].n = counter.value
        rows['done'].n = done.value
        for b in rows.values():
            b.refresh()
    for b in rows.values():
        b.close()


def run_pipeline(
    num_samples: int,
    batch_size: int,
    wave_generator_config: dict[str, int | tuple[int, int] | tuple[float, float] | bool],
    folder_name: str,
    base_seed: int | None = None
) -> None:
    output_path = os.path.join(params.dataset_base_path, folder_name)
    os.mkdir(output_path)

    dataset_temp_path = os.path.join(params.dataset_base_path, '_temp')
    # keep separate to only check _temp for existence
    os.makedirs(dataset_temp_path, exist_ok=False)
    os.makedirs(os.path.join(dataset_temp_path, 'cache/huggingface'))

    # default initialization
    gpu_processes = NUM_GPUS * 4

    # testing tuning params
    gpu_processes = 8


    root = np.random.SeedSequence(base_seed)
    print(f"base seed: {root.entropy}")
    child_seeds = root.spawn(gpu_processes) # spawn child seeds for each GPU process

    ctx = torch.multiprocessing.get_context("spawn")
    jobs = ctx.Queue(maxsize=gpu_processes * 2)

    # status updates
    busy = {n: ctx.Value('i', 0) for n in ('generate', 'convert', 'save')}
    done = ctx.Value('i', 0)

    workers = [
        ctx.Process(
            target=worker,
            args=(jobs, torch.device(f'cuda:{i % NUM_GPUS}'), wave_generator_config, batch_size, folder_name, child_seeds[i], busy['generate'], done)
        ) for i in range(gpu_processes)
    ]

    stop = threading.Event()
    mon = threading.Thread(target=monitor, daemon=True, args=(
        stop,
        {'jobs': (jobs, gpu_processes * 2)},
        {'generate': (busy['generate'], gpu_processes)},
        done, num_samples // batch_size))

    mon.start()

    for p in workers:
        p.start()
    for i in range(num_samples // batch_size):
        jobs.put(i)
    for _ in range(gpu_processes):
        jobs.put(None)
    for p in workers:
        p.join()

    stop.set()
    mon.join()

    bad = [p.name for p in workers if p.exitcode != 0]
    succ_saved = len(os.listdir(output_path)) == (num_samples // batch_size)

    shutil.rmtree(dataset_temp_path)

    if bad or not succ_saved:
        raise Exception(f"Some processes exited with non-zero exit code: {bad}; expected {num_samples // batch_size} batches, but found {len(os.listdir(output_path))} in {output_path}")


if __name__ == "__main__":

    wave_generator_config = {
        'resolution_px_per_km': 2,
        'resolution_seconds_per_time_step': 30,
        'angle': (0, 360), # 180 would be sufficient
        'wavelength': (5, 30),
        'amplitude': (2, 10),
        'phase': (0, 2 * np.pi),
        'period': (5, 25),
        'alpha': (0.2, 0.8),
        'num_waves': (1, 5), # randint is uniformly distributed, too
        'time_steps': 7,
        'img_shape': image_size,
        'to_cpu': True,
        'to_numpy': True
    }

    run_pipeline(
        num_samples=100_000,
        batch_size=1000,
        wave_generator_config=wave_generator_config,
        folder_name='wave_dataset_2',
        base_seed=None
    )
