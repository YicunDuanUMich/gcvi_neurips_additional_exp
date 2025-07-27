import torch
from torch import optim

import random
import numpy as np
import time
import tqdm
import copy

from termcolor import colored

import pyro

from pyro_cases.utils.base_vae import BaseVAEwRegister
from pyro_cases.utils.vae_dict import vae_dict

INIT_SEED = 10_000

def get_vsbc(vae: BaseVAEwRegister, batch_size, sample_dict):
    obs, true_theta = vae.extract_x_as_set(batch_size, sample_dict), vae.extract_theta(sample_dict)
    with torch.no_grad():
        raw_pred = vae.encoder(obs)
    vsbc = vae.variational_dist.get_vsbc(raw_pred, true_theta)
    return vsbc.permute([1, 0]).cpu()  # (k, num_obs)

def compare_ref_and_est(vae: BaseVAEwRegister, batch_size, sample_dict, test_seed):
    obs, true_theta = vae.extract_x_as_set(batch_size, sample_dict), vae.extract_theta(sample_dict)
    with torch.no_grad():
        raw_pred = vae.encoder(obs)
    est_theta1, est_theta2 = vae.variational_dist.get_theta(raw_pred)
    return {
        "obs_seed": test_seed,
        "obs": obs.cpu(),
        "est_theta1": est_theta1.cpu(),
        "est_theta2": est_theta2.cpu(),
        "raw_pred": raw_pred.cpu(),
        "true_theta": true_theta.cpu(),
    }


class NullScheduler:
    def step(self):
        pass


def train_and_test_amortized_favi(task_name, 
                                    seed, 
                                    device, 
                                    lr, 
                                    lr_schedule,
                                    batch_size,
                                    network_width,
                                    steps,
                                    test_seed,
                                    num_test_obs,
                                    show_progress,
                                    silent=False,
                                    return_vae=False,
                                    suppress_error=True,
                                    nn_type="set_transformer"):
    pyro.clear_param_store()

    if task_name in vae_dict:
        vae = vae_dict[task_name]
    else:
        raise NotImplementedError()
    vae = vae(hidden_dim=network_width, use_neural_network=True, nn_type=nn_type).to(device=device)
    vae.do_register(batch_size)
    favi_encoder = copy.deepcopy(vae.encoder).to(device=device)
    favi_optimizer = optim.Adam(favi_encoder.parameters(),
                                lr=lr, amsgrad=True)
    favi_scheduler = None
    match lr_schedule:
        case "plain":
            favi_scheduler = NullScheduler()
        case "custom_decrease":
            favi_scheduler = optim.lr_scheduler.LambdaLR(favi_optimizer,
                                                         lr_lambda=lambda cur_steps: 1 / (1 + cur_steps / 10 * lr))
        case "exponential":
            favi_scheduler = optim.lr_scheduler.ExponentialLR(favi_optimizer,
                                                              gamma=0.9997)
        case "milestones":
            favi_scheduler = optim.lr_scheduler.MultiStepLR(favi_optimizer,
                                                            milestones=[steps // 4, 
                                                                        steps // 2, 
                                                                        (steps // 4) * 3],
                                                            gamma=0.1)
        case "cosine_annealing":
            favi_scheduler = optim.lr_scheduler.CosineAnnealingLR(favi_optimizer,
                                                                  T_max=steps)
        case "cyclic":
            favi_scheduler = optim.lr_scheduler.CyclicLR(favi_optimizer,
                                                         base_lr=lr * 1e-2,
                                                         max_lr=lr,
                                                         step_size_up=steps // 4,
                                                         cycle_momentum=False)
        case "one_cycle":
            favi_scheduler = optim.lr_scheduler.OneCycleLR(favi_optimizer,
                                                           max_lr=lr, total_steps=steps)
        case "cosine_annealing_warm_restart":
            favi_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(favi_optimizer,
                                                                            T_0=steps // 2)
        case _:
            raise NotImplementedError()

    task_start_time = time.ctime()
    start_time = time.time()
    
    pyro.set_rng_seed(test_seed)
    test_sample_dict = vae.generate_sample_dict(batch_size=num_test_obs)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    favi_encoder = favi_encoder.train()
    favi_training_loss = []
    iterations = range(steps) if not show_progress else tqdm.tqdm(list(range(steps)))
    favi_error = None

    def run_one_iter(sample_dict):
        favi_optimizer.zero_grad()
        favi_raw_pred = favi_encoder(vae.extract_x_as_set(batch_size, sample_dict))
        favi_loss = vae.variational_dist.batch_favi_loss(vae.extract_theta(sample_dict), favi_raw_pred)
        favi_loss = favi_loss.mean()
        assert not torch.isnan(favi_loss).any()
        assert not torch.isinf(favi_loss).any()
        favi_loss.backward()
        torch.nn.utils.clip_grad_norm_(favi_encoder.parameters(), max_norm=1.0)
        favi_optimizer.step()
        favi_scheduler.step()
        favi_training_loss.append(favi_loss.item())

    for _ in iterations:
        sample_dict = vae.generate_sample_dict(batch_size=batch_size)
        if suppress_error:
            try:
                if favi_error is None:
                    run_one_iter(sample_dict)
            except Exception as e:
                if not silent:
                    print(colored(f"get exception during FAVI training:\n {e}", "red"))
                favi_error = str(e)
        else:
            run_one_iter(sample_dict)

    favi_vae_wrap = copy.deepcopy(vae)
    favi_vae_wrap.encoder = favi_encoder
    
    if favi_error is None:
        favi_vae_wrap = favi_vae_wrap.eval()
        # direct
        favi_test_result_dict = compare_ref_and_est(favi_vae_wrap, num_test_obs, test_sample_dict, test_seed)
        # vsbc
        favi_vsbc = get_vsbc(favi_vae_wrap, num_test_obs, test_sample_dict)
    else:
        favi_test_result_dict = None
        favi_vsbc = None

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
        "favi_training_loss": favi_training_loss,
        "favi_test_sample_dict": test_sample_dict,
        "favi_test_result_dict": favi_test_result_dict,
        "favi_vae_wrap": favi_vae_wrap.cpu() if return_vae else None,
        "favi_vsbc": favi_vsbc,
        "favi_error": favi_error,
    }
