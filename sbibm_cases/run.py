import torch
import random
import numpy as np
import sbibm
import click
import multiprocessing
import time

from pathlib import Path

from modules.dense import (DenseEncoder1LayerGaussian, 
                           DenseGaussianLinearELBO,
                           DenseGaussianLinearUniformELBO,
                           DenseSLCPELBO,
                           DenseSLCPwDistractorELBO,
                           DenseBernoulliGLMELBO,
                           DenseBernoulliGLMRawELBO,
                           DenseGaussianMixtureELBO,
                           DenseTwoMoonsELBO)
from modules.custom_lr_scheduler import CustomOptim


def compare_ref_and_est(task, obs_index, encoder, device):
    observation = task.get_observation(num_observation=obs_index)
    reference_samples = task.get_reference_posterior_samples(num_observation=obs_index)
    
    with torch.inference_mode():
        est_eta1, est_eta2 = encoder(observation.to(device=device))
        est_eta1 = est_eta1.squeeze(0)
        est_eta2 = est_eta2.squeeze(0)
    
    sigma2 = -1 / (2 * est_eta2)
    mu = est_eta1 * sigma2
    true_mu = reference_samples.mean(dim=0)
    true_sigma2 = reference_samples.var(dim=0)

    return {
        "est_sigma2": sigma2,
        "est_mu": mu,
        "true_mu": true_mu,
        "true_sigma2": true_sigma2,
    }

def train_and_test(task_name, seed, device, use_elbo):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()

    if not use_elbo:
        encoder = DenseEncoder1LayerGaussian(in_dim=task.dim_data,
                                             out_dim=task.dim_parameters * 2,
                                             hidden_dim=1024).to(device=device)
    else:
        match task_name:
            case "gaussian_linear":
                encoder = DenseGaussianLinearELBO
            case "gaussian_linear_uniform":
                encoder = DenseGaussianLinearUniformELBO
            case "slcp":
                encoder = DenseSLCPELBO
            case "slcp_distractors":
                encoder = DenseSLCPwDistractorELBO
            case "bernoulli_glm":
                encoder = DenseBernoulliGLMELBO
            case "bernoulli_glm_raw":
                encoder = DenseBernoulliGLMRawELBO
            case "gaussian_mixture":
                encoder = DenseGaussianMixtureELBO
            case "two_moons":
                encoder = DenseTwoMoonsELBO
            case _:
                raise NotImplementedError()
        encoder = encoder(in_dim=task.dim_data,
                          out_dim=task.dim_parameters * 2,
                          elbo_k=100,
                          hidden_dim=1024).to(device=device)
    
    lr = 1e-3
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, amsgrad=True)
    scheduler = CustomOptim(optimizer, start_lr=lr)

    start_time = time.time()

    encoder = encoder.train()
    steps = int(1e4)
    batch_size = 64
    training_loss = []
    for _ in range(steps):
        thetas = prior(num_samples=batch_size)
        xs = simulator(thetas)
        thetas, xs = thetas.to(device=device), xs.to(device=device)
        
        scheduler.zero_grad()
        # optimizer.zero_grad()
        if not use_elbo:
            loss = encoder.batch_favi_loss(thetas, xs)
        else:
            loss = encoder.batch_elbo_loss(xs)
        loss = loss.mean()
        loss.backward()
        # optimizer.step()
        scheduler.step_and_update_lr()

        training_loss.append(loss.item())

    encoder = encoder.eval()
    test_dict_list = [compare_ref_and_est(task, i + 1, encoder, device) 
                      for i in range(10)]
    
    end_time = time.time()
    print(f"[{seed} completes] cost time: {end_time - start_time:.1f} seconds")

    return {
        "seed": seed,
        "task": task_name,
        "training_loss": training_loss, 
        "test_dict_list": test_dict_list,
    }


@click.command()
@click.option("--task-name", 
              type=click.Choice(sbibm.get_available_tasks(), 
                                case_sensitive=True), 
              required=True, help="task name")
@click.option("--model-num", type=int, required=True, help="number of models to be trained and tested")
@click.option("--cuda-idx", type=int, required=True, help="the index of cuda device")
@click.option("--save-path", type=str, default="/data/scratch/pduan/gcvi_output", help="path to output file")
@click.option("--use-elbo", is_flag=True, help="whether to train model using ELBO")
def main(task_name, model_num, cuda_idx, save_path, use_elbo):
    print("+" * 100)
    print("Args:")
    print(f"\t task_name: {task_name}")
    print(f"\t model_mum: {model_num}")
    print(f"\t cuda_idx: {cuda_idx}")
    print(f"\t save_path: {save_path}")
    print("+" * 100)

    start_time = time.time()
    device = torch.device(f"cuda:{cuda_idx}")
    with multiprocessing.Pool(processes=4, maxtasksperchild=16) as p:
        output_list = p.starmap(train_and_test, 
                                [(task_name, 
                                  random.Random(1234 + i).randint(10_000, 100_000 - 1),
                                  device,
                                  use_elbo)
                                for i in range(model_num)])
    if not use_elbo:
        torch.save(output_list, Path(save_path) / f"{task_name}_{model_num}.pt")
    else:
        torch.save(output_list, Path(save_path) / f"elbo_{task_name}_{model_num}.pt")
    end_time = time.time()
    print(f"total time: {end_time - start_time:.1f} seconds")
    print("done")

if __name__ == "__main__":
    main()
