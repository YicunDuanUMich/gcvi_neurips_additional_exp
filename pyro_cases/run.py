import torch
from torch import optim

import random
import numpy as np
import time
import torch.utils
import tqdm
import copy
import click
import multiprocessing

from termcolor import colored
from pathlib import Path
from typing import Dict

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_cases.psis import psislw
from pyro_cases.base_vae import BaseVAE, BaseVAEwRegister
from pyro_cases.model import (GaussianLinearVAE,
                              GaussianLinearUniformVAE,
                              SLCPVAE,
                              SLCPwDistractorVAE,
                              BeroulliGLMRAWVAE,
                              BernoulliGLMVAE,
                              GaussianMixtureVAE,
                              TwoMoonsVAE,
                              ARM_anova_randon_nopred,
                              ARM_anova_randon_nopred_chr,
                              ARM_congress,
                              ARM_earnings1,
                              ARM_earnings2,
                              ARM_earnings_latin_square,
                              ARM_earnings_latin_square_chr,
                              ARM_earnings_vary_si,
                              ARM_earnings_vary_si_chr,
                              ARM_election88_ch14,
                              ARM_election88_ch19,
                              ARM_electric,
                              ARM_electric_1a,
                              ARM_electric_1a_chr,
                              ARM_electric_1b,
                              ARM_electric_1b_chr,
                              ARM_electric_1c,
                              ARM_electric_1c_chr,
                              ARM_electric_chr,
                              ARM_electric_inter)

vae_dict: Dict[str, BaseVAE] = {
    "gaussian_linear": GaussianLinearVAE,
    "gaussian_linear_uniform": GaussianLinearUniformVAE,
    "slcp": SLCPVAE,
    "slcp_distractors": SLCPwDistractorVAE,
    "bernoulli_glm_raw": BeroulliGLMRAWVAE,
    "bernoulli_glm": BernoulliGLMVAE,
    "gaussian_mixture": GaussianMixtureVAE,
    "two_moons": TwoMoonsVAE,
    "arm_anova_randon_nopred": ARM_anova_randon_nopred,
    "arm_anova_randon_nopred_chr": ARM_anova_randon_nopred_chr,
    "arm_congress": ARM_congress,
    "arm_earnings1": ARM_earnings1,
    "arm_earnings2": ARM_earnings2,
    "arm_earnings_latin_square": ARM_earnings_latin_square,
    "arm_earnings_latin_square_chr": ARM_earnings_latin_square_chr,
    "arm_earnings_vary_si": ARM_earnings_vary_si,
    "arm_earnings_vary_si_chr": ARM_earnings_vary_si_chr,
    "arm_election88_ch14": ARM_election88_ch14,
    "arm_election88_ch19": ARM_election88_ch19,
    "arm_electric": ARM_electric,
    "arm_electric_1a": ARM_electric_1a,
    "arm_electric_1a_chr": ARM_electric_1a_chr,
    "arm_electric_1b": ARM_electric_1b,
    "arm_electric_1b_chr": ARM_electric_1b_chr,
    "arm_electric_1c": ARM_electric_1c,
    "arm_electric_1c_chr": ARM_electric_1c_chr,
    "arm_electric_chr": ARM_electric_chr,
    "arm_electric_inter": ARM_electric_inter,
}

class NullScheduler:
    def step(self):
        pass

INIT_SEED = 10_000

def get_k_hat(vae, num_obs, num_samples):
    sample_dict_list = [vae.get_obs_sample_dict(INIT_SEED + i) for i in range(num_obs)]
    lw = torch.zeros(num_samples, num_obs)
    elbo = Trace_ELBO(num_particles=1)
    for i in range(num_obs):
        for j in range(num_samples):
            with torch.no_grad():
                lw[j, i] = -1 * elbo.loss(vae.model, vae.guide, 1, sample_dict_list[i])
    return psislw(lw.numpy(), Reff=1.0)[1]  # (num_obs, )

def get_single_vsbc(vae, obs_seed):
    sample_dict = vae.get_obs_sample_dict(obs_seed)
    x = vae.extract_x(sample_dict)
    true_theta = vae.extract_theta(sample_dict)
    with torch.no_grad():
        est_theta_loc, est_theta_scale = vae.encoder(x)
        est_theta_loc = est_theta_loc.squeeze(0)
        est_theta_scale = est_theta_scale.squeeze(0)
        est_dist = dist.Normal(est_theta_loc, est_theta_scale)
        vsbc = 1 - est_dist.cdf(true_theta.squeeze(0))
    return vsbc  # (k, )

def get_vsbc(vae, num_obs):
    vsbc_list = []
    for i in range(num_obs):
        vsbc_list.append(get_single_vsbc(vae, INIT_SEED + i))
    return torch.stack(vsbc_list, dim=-1)  # (k, num_obs)

def compare_single_ref_and_est(vae: BaseVAE, obs_seed):
    obs, theta = vae.get_observation(obs_seed)
    
    with torch.inference_mode():
        mu, sigma = vae.encoder(obs)

    return {
        "obs_seed": obs_seed,
        "obs": obs.cpu(),
        "est_sigma2": (sigma ** 2).squeeze(0).cpu(),
        "est_mu": mu.squeeze(0).cpu(),
        "true_theta": theta.squeeze(0).cpu(),
    }

def compare_ref_and_est(vae, num_obs):
    return [compare_single_ref_and_est(vae, INIT_SEED + i) for i in range(num_obs)]

def train_and_test(task_name, 
                   seed, 
                   device, 
                   lr, 
                   lr_schedule,
                   num_particles,
                   vectorize_particles,
                   batch_size, 
                   network_width,
                   steps,
                   direct_compare_n_obs,
                   k_hat_n_obs, 
                   k_hat_n_samples,
                   vsbc_n_obs,
                   show_progress,
                   silent=False,
                   return_vae=False,
                   suppress_error=True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pyro.clear_param_store()

    if task_name in vae_dict:
        vae = vae_dict[task_name]
    else:
        raise NotImplementedError()
    vae = vae(hidden_dim=network_width).to(device=device)
    elbo_optimizer = ClippedAdam({"lr": lr, 
                                  "clip_norm": 1.0,
                                  "lrd": 0.9997})
    svi = SVI(vae.model, vae.guide,
              elbo_optimizer, 
              loss=Trace_ELBO(num_particles=num_particles,
                              vectorize_particles=vectorize_particles))
    
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

    vae = vae.train()
    favi_encoder = favi_encoder.train()
    elbo_training_loss = []
    favi_training_loss = []
    iterations = range(steps) if not show_progress else tqdm.tqdm(list(range(steps)))
    elbo_cant_converge = False
    favi_cant_converge = False
    if isinstance(vae, BaseVAEwRegister):
        vae.do_register(batch_size)
    for _ in iterations:
        sample_dict = vae.generate_sample_dict(batch_size=batch_size)
        if suppress_error:
            try:
                if not elbo_cant_converge:
                    step_loss = svi.step(batch_size, sample_dict)
                    elbo_training_loss.append(step_loss)
            except Exception as e:
                print(colored(f"get exception during ELBO training:\n {e}", "red"))
                elbo_cant_converge = True

            try:
                if not favi_cant_converge:
                    favi_optimizer.zero_grad()
                    favi_loss = favi_encoder.batch_favi_loss(vae.extract_theta(sample_dict), 
                                                            vae.extract_x(sample_dict))
                    favi_loss = favi_loss.mean()
                    favi_loss.backward()
                    torch.nn.utils.clip_grad_norm_(favi_encoder.parameters(), max_norm=1.0)
                    favi_optimizer.step()
                    favi_scheduler.step()
                    favi_training_loss.append(favi_loss.item())
            except Exception as e:
                print(colored(f"get exception during FAVI training:\n {e}", "red"))
                favi_cant_converge = True
        else:
            step_loss = svi.step(batch_size, sample_dict)
            elbo_training_loss.append(step_loss)

            favi_optimizer.zero_grad()
            favi_loss = favi_encoder.batch_favi_loss(vae.extract_theta(sample_dict), 
                                                    vae.extract_x(sample_dict))
            favi_loss = favi_loss.mean()
            favi_loss.backward()
            torch.nn.utils.clip_grad_norm_(favi_encoder.parameters(), max_norm=1.0)
            favi_optimizer.step()
            favi_scheduler.step()
            favi_training_loss.append(favi_loss.item())

    elbo_vae = vae
    favi_vae_wrap = copy.deepcopy(vae)
    favi_vae_wrap.encoder = favi_encoder

    if not elbo_cant_converge:
        elbo_vae = elbo_vae.eval()
        # direct
        elbo_test_dict_list = compare_ref_and_est(elbo_vae, num_obs=direct_compare_n_obs)
        # k hat
        elbo_k_hat = torch.from_numpy(get_k_hat(elbo_vae, num_obs=k_hat_n_obs, num_samples=k_hat_n_samples))
        # vsbc
        elbo_vsbc = get_vsbc(elbo_vae, num_obs=vsbc_n_obs).cpu()
    else:
        elbo_test_dict_list = "elbo_cant_converge"
        elbo_k_hat = "elbo_cant_converge"
        elbo_vsbc = "elbo_cant_converge"
    
    if not favi_cant_converge:
        favi_vae_wrap = favi_vae_wrap.eval()
        # direct
        favi_test_dict_list = compare_ref_and_est(favi_vae_wrap, num_obs=direct_compare_n_obs)
        # k hat
        favi_k_hat = torch.from_numpy(get_k_hat(favi_vae_wrap, num_obs=k_hat_n_obs, num_samples=k_hat_n_samples))
        # vsbc
        favi_vsbc = get_vsbc(favi_vae_wrap, num_obs=vsbc_n_obs).cpu()
    else:
        favi_test_dict_list = "favi_cant_converge"
        favi_k_hat = "favi_cant_converge"
        favi_vsbc = "favi_cant_converge"
    
    task_end_time = time.ctime()
    end_time = time.time()
    if not silent:
        print(f"[{seed} completes] task start time: {task_start_time}; task end time: {task_end_time}; cost time: {end_time - start_time:.1f} seconds")

    return {
        "seed": seed,
        "task": task_name,
        "elbo_training_loss": elbo_training_loss, 
        "elbo_test_dict_list": elbo_test_dict_list,
        "favi_training_loss": favi_training_loss,
        "favi_test_dict_list": favi_test_dict_list,
        "elbo_vae": elbo_vae.cpu() if return_vae else None,
        "favi_vae_wrap": favi_vae_wrap.cpu() if return_vae else None,
        "favi_k_hat": favi_k_hat,
        "favi_vsbc": favi_vsbc,
        "elbo_k_hat": elbo_k_hat,
        "elbo_vsbc": elbo_vsbc,
    }


@click.command()
@click.option("--task-name", required=True, help="task name")
@click.option("--model-num", type=int, required=True, help="number of models to be trained and tested")
@click.option("--network-width", type=int, required=True, help="the width of network")
@click.option("--lr-schedule", required=True, help="the schedule for learning rate")
@click.option("--processes-num", type=int, required=True, help="the number of processes to run in parallel")
@click.option("--cuda-idx", type=int, required=True, help="the index of cuda device")
@click.option("--save-path", type=str, default="/data/scratch/pduan/gcvi_output", help="path to output file")
def main(task_name, model_num, network_width, lr_schedule, processes_num, cuda_idx, save_path):
    print("+" * 100)
    print("Args:")
    print(f"\t task_name: {task_name}")
    print(f"\t model_mum: {model_num}")
    print(f"\t network_width: {network_width}")
    print(f"\t lr_schedule: {lr_schedule}")
    print(f"\t cuda_idx: {cuda_idx}")
    print(f"\t save_path: {save_path}")
    print("+" * 100)

    start_time = time.time()
    device = torch.device(f"cuda:{cuda_idx}")
    if processes_num > 1:
        with multiprocessing.Pool(processes=4, maxtasksperchild=16) as p:
            output_list = p.starmap(train_and_test, 
                                    [(task_name, 
                                    random.Random(1234 + i).randint(10_000, 100_000 - 1),
                                    device,
                                    1e-3,
                                    lr_schedule,
                                    1,
                                    False,
                                    1024,
                                    network_width,
                                    20_000,
                                    10,
                                    30,
                                    100,
                                    1000,
                                    False)
                                    for i in range(model_num)])
    else:
        output_list = [train_and_test(
            task_name=task_name,
            seed=random.Random(1234 + i).randint(10_000, 100_000 - 1),
            device=device,
            lr=1e-3,
            lr_schedule=lr_schedule,
            num_particles=1,
            vectorize_particles=False,
            batch_size=1024,
            network_width=network_width,
            steps=20_000,
            direct_compare_n_obs=10,
            k_hat_n_obs=30,
            k_hat_n_samples=100,
            vsbc_n_obs=1000,
            show_progress=False
        ) for i in range(model_num)]
    torch.save(output_list, Path(save_path) / f"pyro_{task_name}_mn_{model_num}_nw_{network_width}_lr_{lr_schedule}.pt")
    end_time = time.time()
    print(f"total time: {end_time - start_time:.1f} seconds")
    print("done")

if __name__ == "__main__":
    main()