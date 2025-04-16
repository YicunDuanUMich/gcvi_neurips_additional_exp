import torch
import random
import multiprocessing
import gc
import click
import time

from pathlib import Path
from termcolor import colored

from pyro_cases.run import train_and_test, vae_dict


def my_worker(kwargs):
    return train_and_test(**kwargs)

def process_task(tags, task_params_nested_list, least_tasks_per_chunk):
    results = []
    merged_task_params_list = sum(task_params_nested_list, [])  # flatten the nested list
    repeat_times = len(task_params_nested_list[0])
    total_sub_tasks = len(merged_task_params_list)
    assert total_sub_tasks >= least_tasks_per_chunk
    assert all([len(tl) == repeat_times for tl in task_params_nested_list])  # assert equal length
    assert total_sub_tasks % len(tags) == 0
    boundaries = list(range(0, total_sub_tasks, least_tasks_per_chunk)) + [total_sub_tasks]
    slices = list(zip(boundaries[:-1], boundaries[1:]))
    print_blue = lambda x: print(colored(x, "blue"))
    for ls, rs in slices:
        # python's multiprocessing has a bug if we set the maxtasksperchild
        # details in https://github.com/python/cpython/issues/93580
        # the map_async and other async methods are also buggy
        # don't use them
        start_time = time.time()
        start_date = time.ctime()
        print_blue(f"tag {tags} [{rs}/{total_sub_tasks}]: start at {start_date}")
        with multiprocessing.Pool(processes=least_tasks_per_chunk) as p:
            results.extend(p.map(my_worker, merged_task_params_list[ls:rs]))
        end_time = time.time()
        end_date = time.ctime()
        print_blue(f"tag {tags} [{rs}/{total_sub_tasks}]: end at {end_date}")
        print_blue(f"tag {tags} [{rs}/{total_sub_tasks}]: take {end_time - start_time:.1f} seconds")
    return [results[i:(i + repeat_times)] for i in range(0, len(results), repeat_times)]

@click.command()
@click.option("--save-path", type=str, help="path to output file")
@click.option("--cuda-idx", type=str, help="cuda devices")
@click.option("--repeat-times", type=int)
@click.option("--max-processes-per-gpu", type=int, default=4)
def main(save_path, cuda_idx, repeat_times, max_processes_per_gpu):
    task_names = list(vae_dict.keys())
    lr_schedulers = ["cosine_annealing"]
    network_widths = [1024]
    save_path = Path(save_path)
    if cuda_idx == "all":
        cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        cuda_devices = [f"cuda:{i}" for i in cuda_idx.split(",")]
    assert repeat_times % len(cuda_devices) == 0
    least_tasks_per_chunk = len(cuda_devices) * max_processes_per_gpu

    print_green = lambda x: print(colored(x, "green"))
    print_green("+" * 100)
    print_green("Config:")
    print_green(f"\t tasks_names: {task_names}")
    print_green(f"\t lr_schedulers: {lr_schedulers}")
    print_green(f"\t network_widths: {network_widths}")
    print_green(f"\t save_path: {save_path}")
    print_green(f"\t cuda_devices: {cuda_devices}")
    print_green(f"\t tags:")
    task_tags = [(f"t_{tn}_lr_{lr_s}_nw_{network_width}", 
                  (tn, lr_s, network_width))
                 for tn in task_names
                 for lr_s in lr_schedulers
                 for network_width in network_widths]
    for i, (t, _) in enumerate(task_tags):
        print_green(f"\t  [{i}]: {t}")
    print_green("+" * 100)

    tasks = {t: [] for t, _ in task_tags}
    for t, (tn, lr_s, network_width) in task_tags:
        assert tn in vae_dict
        for ri in range(repeat_times):
            device = torch.device(cuda_devices[ri % len(cuda_devices)])
            tasks[t].append(
                {
                    "task_name": tn,
                    "seed": random.Random(1234 + ri).randint(10_000, 100_000 - 1),
                    "device": device,
                    "lr": 1e-3,
                    "lr_schedule": lr_s,
                    "num_particles": 1,
                    "vectorize_particles": False,
                    "batch_size": 1024,
                    "network_width": network_width,
                    "steps": 10_000,
                    "direct_compare_n_obs": 10,
                    "k_hat_n_obs": 30,
                    "k_hat_n_samples":100,
                    "vsbc_n_obs": 1000,
                    "show_progress": False,
                    "silent": True,
                    "return_vae": False,
                    "suppress_error": True,
                }
            )
    
    withhold_ts = []
    withhold_t_params = []
    withhold_t_param_num = 0
    withhold_save_files = []
    for t, task_param_dict_list in tasks.items():
        save_t_file_path = save_path / f"pyro_{t}_mn_{repeat_times}.pt"
        if save_t_file_path.exists():
            print_green(f"find {save_t_file_path}; skip this task")
            continue
        withhold_ts.append(t)
        withhold_t_params.append(task_param_dict_list)
        withhold_t_param_num += len(task_param_dict_list)
        withhold_save_files.append(save_t_file_path)
        if withhold_t_param_num < least_tasks_per_chunk:
            continue
        results = process_task(withhold_ts, withhold_t_params, least_tasks_per_chunk)
        for cur_result, cur_save_file_path in zip(results, withhold_save_files, strict=True):
            torch.save(cur_result, cur_save_file_path)
        del results
        withhold_ts = []
        withhold_t_params = []
        withhold_t_param_num = 0
        withhold_save_files = []
        gc.collect()

    print_green("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
