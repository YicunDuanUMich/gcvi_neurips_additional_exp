import torch
import random
import multiprocessing
import gc

from pathlib import Path

from pyro_cases.run import train_and_test

def my_worker(func, kwargs, device):
    torch_device = torch.device(device)
    return func(**kwargs, device=torch_device)

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    model_num = 100
    task_names = ["gaussian_linear"]
    lr_schedulers = ["plain", 
                     "custom_decrease", "exponential", "milestones"
                     "cosine_annealing", "cyclic", "one_cycle", 
                     "cosine_annealing_warm_restart"]
    network_widths = [512, 1024, 2048]
    save_path = Path("/data/scratch/pduan/new_gcvi_output")
    cuda_devices = [f"cuda:{i}" for i in [1, 2, 3, 4]]

    task_tags = [(f"t_{tn}_lr_{lr_s}_nw_{network_width}", tn, lr_s, network_width)
                 for tn in task_names
                 for lr_s in lr_schedulers
                 for network_width in network_widths]
    tasks = {t: [] for t, _, _, _ in task_tags}
    for t, tn, lr_s, network_width in task_tags:
        for model_i in range(model_num):
            tasks[t].append(
                (train_and_test, 
                {
                    "task_name": tn,
                    "seed": random.Random(1234 + model_i).randint(10_000, 100_000 - 1),
                    "lr": 1e-3,
                    "lr_schedule": lr_s,
                    "num_particles": 1,
                    "vectorize_particles": False,
                    "batch_size": 1024,
                    "network_width": network_width,
                    "steps": 20_000,
                    "show_progress": False,
                })
            )
    
    pools = {cuda_d: multiprocessing.Pool(processes=4) for cuda_d in cuda_devices}
    processes = {t: [] for t, _, _, _ in task_tags}
    for tk, tv_list in tasks.items():
        for i, (func, param_dict) in enumerate(tv_list):
            device = cuda_devices[i % len(cuda_devices)]
            pool = pools[device]
            processes[tk].append(pool.apply_async(my_worker, 
                                                  args=(func, param_dict, device)))

    for t, _, _, _ in task_tags:
        outputs = [p.get() for p in processes[t]]
        torch.save(outputs, save_path / f"pyro_{t}_mn_{model_num}.pt")
        del outputs
        gc.collect()

    for pool in pools.values():
        pool.close()
        pool.join()
