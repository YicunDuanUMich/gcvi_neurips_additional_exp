import torch

import random
import numpy as np
import time

import pyro
from pyro.infer import MCMC, NUTS

from pyro_cases.utils.base_vae import BaseVAEwRegister
from pyro_cases.utils.vae_dict import vae_dict

def move_dict_to_cpu(pre_dict: dict):
    return {
        k: v.to(device="cpu") if isinstance(v, torch.Tensor) else v
        for k, v in pre_dict.items()
    }

def train_and_test_mcmc_with_fixed_design(task_name, 
                                            seed, 
                                            device, 
                                            test_sample_dict,
                                            warmup_steps, 
                                            n_samples):
    pyro.clear_param_store()

    if task_name in vae_dict:
        vae = vae_dict[task_name]
    else:
        raise NotImplementedError()
    vae: BaseVAEwRegister = vae(hidden_dim=1, use_neural_network=False).to(device=device)
    vae.do_register(1)
    
    nuts_kernel = NUTS(vae.model)
    mcmc = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=warmup_steps)

    task_start_time = time.ctime()
    start_time = time.time()
    
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    mcmc.run(1, test_sample_dict)
    hmc_samples = {k: v.detach().cpu() for k, v in mcmc.get_samples().items()}

    task_end_time = time.ctime()
    end_time = time.time()
    print(f"[task {task_name} seed {seed} completes] " \
            f"task start time: {task_start_time}; " \
            f"task end time: {task_end_time}; " \
            f"cost time: {end_time - start_time:.1f} seconds")

    return {
        "seed": seed,
        "task": task_name,
        "mcmc_object": mcmc,
        "mcmc_samples": hmc_samples,
    }
