import multiprocessing
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import numpy as np

from src.generate_dataset import save
from src.generate_wave import WaveCollection, WaveGenerator

folder_name = 'wave_dataset'
os.mkdir(f'datasets/{folder_name}')

def worker(wave_generator, num_waves: int | tuple[int, int], num_samples_per_worker, seed, result_queue) -> None:
    gen_seed, count_seed = seed.spawn(2)
    np.random.seed(gen_seed.generate_state(1)[0])
    if isinstance(num_waves, tuple):
        rng = np.random.default_rng(count_seed)
        nw = num_waves
        num_waves = lambda : int(rng.integers(nw[0], nw[1])) # type: ignore
    wave_collection = WaveCollection(wave_generator, num_waves) # type: ignore
    for _ in range(num_samples_per_worker):
        result_queue.put(wave_collection.create())

def save_sample(sample, idx, scale, match_range):
        os.mkdir(f'datasets/{folder_name}/{idx}')
        with open(f'datasets/{folder_name}/{idx}/tensors_{idx}.pkl', 'wb') as f:
            pickle.dump(sample, f)
        save(wave_series=sample[1], path=f'datasets/{folder_name}/{idx}', scale=scale, match_range=match_range) # type: ignore

ctx = multiprocessing.get_context('fork')
result_queue = ctx.Queue(maxsize=10_000)

num_workers: int = os.cpu_count() # type: ignore
seeds = np.random.SeedSequence(42).spawn(num_workers)
num_samples = 100_000
base, extra = divmod(num_samples, num_workers)
counts = [base + (i < extra) for i in range(num_workers)]


def create_wave_generator():
    return WaveGenerator(
        resolution_px_per_km = lambda : 2,
        resolution_seconds_per_time_step = lambda : 30,
        angle = lambda : np.random.uniform(0, 360), # 180 would be sufficient
        wavelength = lambda : np.random.uniform(5, 30),
        amplitude = lambda : np.random.uniform(2, 10),
        phase = lambda : np.random.uniform(0, 2 * np.pi),
        period = lambda : np.random.uniform(5, 25),
        alpha = lambda : np.random.uniform(0.2, 0.8),
        time_steps = 7,
        img_shape = (256, 256)
    )


processes = [ctx.Process(target=worker, args=(create_wave_generator(), (1, 5), counts[i], seeds[i], result_queue)) for i in range(num_workers)]
for p in processes:
    p.start()

sephamore = threading.Semaphore(10_000)

def on_done(future):
    sephamore.release()
    if future.exception() is not None:
        print(f"Error in thread: {future.exception()}")

with ThreadPoolExecutor(max_workers=32) as pool:
    for index in range(num_samples):
        sample = result_queue.get()
        sephamore.acquire()
        pool.submit(save_sample, sample, index, 'normalize', None).add_done_callback(on_done)
        del sample
for p in processes:
    p.join()
