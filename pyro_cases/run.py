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

from pathlib import Path

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from pyro_cases.model import (BaseVAE,
                              GaussianLinearVAE,
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
                              ARM_earnings_latin_square)

class NullScheduler:
    def step(self):
        pass


def compare_ref_and_est(vae: BaseVAE, 
                        encoder, 
                        obs_seed):
    obs, theta = vae.get_observation(obs_seed)
    
    with torch.inference_mode():
        mu, sigma = encoder(obs)
        mu = mu.squeeze(0)
        sigma = sigma.squeeze(0)

    return {
        "obs_seed": obs_seed,
        "obs": obs.cpu(),
        "est_sigma2": (sigma ** 2).cpu(),
        "est_mu": mu.cpu(),
        "true_theta": theta.squeeze(0).cpu(),
    }

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
                   show_progress):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    match task_name:
        case "gaussian_linear":
            vae = GaussianLinearVAE
        case "gaussian_linear_uniform":
            vae = GaussianLinearUniformVAE
        case "slcp":
            vae = SLCPVAE
        case "slcp_distractors":
            vae = SLCPwDistractorVAE
        case "bernoulli_glm_raw":
            vae = BeroulliGLMRAWVAE
        case "bernoulli_glm":
            vae = BernoulliGLMVAE
        case "gaussian_mixture":
            vae = GaussianMixtureVAE
        case "two_moons":
            vae = TwoMoonsVAE
        case "arm_anova_randon_nopred":
            vae = ARM_anova_randon_nopred
        case "arm_anova_randon_nopred_chr":
            vae = ARM_anova_randon_nopred_chr
        case "arm_congress":
            vae = ARM_congress
        case "arm_earnings1":
            vae = ARM_earnings1
        case "arm_earnings2":
            vae = ARM_earnings2
        case "arm_earnings_latin_square":
            vae = ARM_earnings_latin_square
        case _:
            raise NotImplementedError()
    vae = vae(hidden_dim=network_width).to(device=device)
    pyro.clear_param_store()
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

    start_time = time.time()

    vae = vae.train()
    favi_encoder = favi_encoder.train()
    elbo_training_loss = []
    favi_training_loss = []
    iterations = range(steps) if not show_progress else tqdm.tqdm(list(range(steps)))
    for _ in iterations:
        sample_dict = vae.generate_sample_dict(batch_size=batch_size)

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


    vae = vae.eval()
    elbo_test_dict_list = [compare_ref_and_est(vae, vae.encoder, i + 10_000) 
                           for i in range(10)]
    
    favi_encoder = favi_encoder.eval()
    favi_test_dict_list = [compare_ref_and_est(vae, favi_encoder, i + 10_000) 
                           for i in range(10)]
    
    end_time = time.time()
    print(f"[{seed} completes] cost time: {end_time - start_time:.1f} seconds")

    return {
        "seed": seed,
        "task": task_name,
        "elbo_training_loss": elbo_training_loss, 
        "elbo_test_dict_list": elbo_test_dict_list,
        "favi_training_loss": favi_training_loss,
        "favi_test_dict_list": favi_test_dict_list,
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
            show_progress=False
        ) for i in range(model_num)]
    torch.save(output_list, Path(save_path) / f"pyro_{task_name}_mn_{model_num}_nw_{network_width}_lr_{lr_schedule}.pt")
    end_time = time.time()
    print(f"total time: {end_time - start_time:.1f} seconds")
    print("done")

if __name__ == "__main__":
    main()