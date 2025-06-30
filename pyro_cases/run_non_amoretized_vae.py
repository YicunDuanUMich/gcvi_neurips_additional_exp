import torch

import random
import numpy as np
import time
import tqdm

from termcolor import colored

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_cases.base_vae import BaseVAE, BaseVAEwRegister
from pyro_cases.run import vae_dict

INIT_SEED = 10_000

def get_vsbc(vae: BaseVAE, sample_dict):
    true_theta = vae.extract_theta(sample_dict)

    est_theta_loc, est_theta_scale = pyro.param("param_theta_loc").detach(), pyro.param("param_theta_scale").detach()
    est_dist = dist.Normal(est_theta_loc, est_theta_scale)
    return (1 - est_dist.cdf(true_theta)).permute([1, 0]).cpu()  # (k, num_obs)

def compare_ref_and_est(vae: BaseVAE, sample_dict, test_seed):
    obs, theta = vae.extract_x(sample_dict), vae.extract_theta(sample_dict)
    return {
        "obs_seed": test_seed,
        "obs": obs.cpu(),
        "est_sigma2": (pyro.param("param_theta_scale").detach() ** 2).cpu(),
        "est_mu": pyro.param("param_theta_loc").detach().cpu(),
        "true_theta": theta.cpu(),
    }

def train_and_test_non_amortized_vae(task_name, 
                                    seed, 
                                    device, 
                                    lr, 
                                    num_particles,
                                    vectorize_particles,
                                    steps,
                                    test_seed,
                                    num_test_obs,
                                    show_progress,
                                    silent=False,
                                    return_vae=False,
                                    suppress_error=True):
    pyro.clear_param_store()

    if task_name in vae_dict:
        vae = vae_dict[task_name]
    else:
        raise NotImplementedError()
    vae = vae(hidden_dim=1, use_neural_network=False).to(device=device)
    elbo_optimizer = ClippedAdam({"lr": lr, 
                                  "clip_norm": 1.0,
                                  "lrd": 0.9997})
    svi = SVI(vae.model, vae.guide,
              elbo_optimizer, 
              loss=Trace_ELBO(num_particles=num_particles,
                              vectorize_particles=vectorize_particles))

    task_start_time = time.ctime()
    start_time = time.time()

    vae = vae.train()
    elbo_training_loss = []
    iterations = range(steps) if not show_progress else tqdm.tqdm(list(range(steps)))
    elbo_error = None
    if isinstance(vae, BaseVAEwRegister):
        vae.do_register(num_test_obs)
    
    pyro.set_rng_seed(test_seed)
    sample_dict = vae.generate_sample_dict(batch_size=num_test_obs)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def run_one_iter():
        step_loss = svi.step(num_test_obs, sample_dict)
        elbo_training_loss.append(step_loss)

    for _ in iterations:    
        if suppress_error:
            try:
                if elbo_error is None:
                    run_one_iter()
            except Exception as e:
                if not silent:
                    print(colored(f"get exception during ELBO training:\n {e}", "red"))
                elbo_error = str(e)
        else:
            run_one_iter()

    elbo_vae = vae
    if elbo_error is None:
        elbo_vae = elbo_vae.eval()
        # direct
        elbo_test_dict_list = compare_ref_and_est(elbo_vae, sample_dict, test_seed)
        # vsbc
        elbo_vsbc = get_vsbc(elbo_vae, sample_dict)
    else:
        elbo_test_dict_list = None
        elbo_vsbc = None

    task_end_time = time.ctime()
    end_time = time.time()
    if not silent:
        print(f"[task {task_name} seed {seed} completes] " \
              f"task start time: {task_start_time}; " \
              f"task end time: {task_end_time}; " \
              f"cost time: {end_time - start_time:.1f} seconds")

    return {
        "seed": seed,
        "task": task_name,
        "elbo_training_loss": elbo_training_loss, 
        "elbo_test_dict_list": elbo_test_dict_list,
        "elbo_vae": elbo_vae.cpu() if return_vae else None,
        "elbo_vsbc": elbo_vsbc,
        "elbo_error": elbo_error,
    }
