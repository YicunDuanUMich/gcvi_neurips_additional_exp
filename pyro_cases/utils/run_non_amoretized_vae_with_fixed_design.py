import torch

import random
import numpy as np
import time
import tqdm
import math

from termcolor import colored

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro import poutine

from pyro_cases.utils.variational_dist import NormalFactorMuSigmaParam
from pyro_cases.utils.base_vae import BaseVAEwRegister
from pyro_cases.utils.vae_dict import vae_dict

def get_vsbc(vae: BaseVAEwRegister, x, theta):
    raw_pred1 = pyro.param("my_param_raw_theta1").detach()
    raw_pred2 = pyro.param("my_param_raw_theta2").detach()
    raw_pred = torch.stack([raw_pred1, raw_pred2], dim=-1)
    vsbc = vae.variational_dist.get_vsbc(raw_pred, theta)
    return vsbc.permute([1, 0]).cpu()  # (k, num_obs)

def compare_ref_and_est(vae: BaseVAEwRegister, x, theta, test_seed):
    raw_pred1 = pyro.param("my_param_raw_theta1").detach()
    raw_pred2 = pyro.param("my_param_raw_theta2").detach()
    raw_pred = torch.stack([raw_pred1, raw_pred2], dim=-1)
    est_theta1, est_theta2 = vae.variational_dist.get_theta(raw_pred)
    return {
        "obs_seed": test_seed,
        "obs": x.cpu(),
        "est_theta1": est_theta1.cpu(),
        "est_theta2": est_theta2.cpu(),
        "raw_pred": raw_pred.cpu(),
        "true_theta": theta.cpu(),
    }

def move_dict_to_cpu(pre_dict: dict):
    return {
        k: v.to(device="cpu") if isinstance(v, torch.Tensor) else v
        for k, v in pre_dict.items()
    }

def ng_precondition_grads(pyro_params):
    ps = pyro.get_param_store()
    sigma2 = ps["my_param_raw_theta2"].detach() ** 2
    pyro_params[0].grad.mul_(sigma2)  # we assume the first one is mu
    u = pyro_params[1].detach()
    sigmoid_u = torch.sigmoid(u)
    pyro_params[1].grad.mul_(sigma2 / (2 * sigmoid_u ** 2 + 1e-8))

def train_and_test_non_amortized_vae_with_fixed_design(task_name, 
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
                                                        suppress_error=True,
                                                        use_natural_gradient=False):
    pyro.clear_param_store()

    if task_name in vae_dict:
        vae = vae_dict[task_name]
    else:
        raise NotImplementedError()
    vae: BaseVAEwRegister = vae(hidden_dim=1, use_neural_network=False).to(device=device)
    vae.do_register(num_test_obs)

    pyro.set_rng_seed(test_seed)
    test_sample_dict = vae.generate_sample_dict(batch_size=1)
    expanded_test_sample_dict = vae.expand_sample_dict_w_fixed_design_matrix(batch_size=num_test_obs,
                                                                             example_sample_dict=test_sample_dict)
    test_x = vae.extract_x_as_set(batch_size=num_test_obs, sample_dict=expanded_test_sample_dict)
    test_theta = vae.extract_theta(expanded_test_sample_dict)
    
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    assert len(list(vae.sample_dict_instr["obs"].keys())) == 1
    obs_key = list(vae.sample_dict_instr["obs"].keys())[0]
    obs_batch_shape = vae.sample_dict_instr["obs"][obs_key]["batch_shape"]
    scale = 1 / math.prod(obs_batch_shape)
    model_fn = poutine.scale(vae.model, scale=scale)
    guide_fn = poutine.scale(vae.guide, scale=scale)
    trace_elbo = Trace_ELBO(num_particles=num_particles, 
                            vectorize_particles=vectorize_particles)
    
    if use_natural_gradient and \
        (not all([isinstance(f, NormalFactorMuSigmaParam) 
                  for f in vae.variational_dist.variational_factors])):
        print(f"WARNING: for task {task_name}, set use_natural_gradient to False")
        use_natural_gradient = False
    
    if use_natural_gradient:
        loss_fn = lambda batch_size, sample_dict: trace_elbo.differentiable_loss(
            model_fn, guide_fn, batch_size, sample_dict)
        with pyro.poutine.trace(param_only=True) as param_capture:
            _loss = loss_fn(num_test_obs, expanded_test_sample_dict)
        pyro_params = [site["value"].unconstrained()
                        for site in param_capture.trace.nodes.values()]
        elbo_optimizer = torch.optim.Adam(pyro_params, lr=lr)
        elbo_scheduler = torch.optim.lr_scheduler.ExponentialLR(elbo_optimizer,
                                                                gamma=0.9997)
    else:
        svi = SVI(model_fn, guide_fn,
                  ClippedAdam({"lr": lr, 
                                "clip_norm": 1.0,
                                "lrd": 0.9997}), 
                  loss=trace_elbo)

    task_start_time = time.ctime()
    start_time = time.time()

    vae = vae.train()
    elbo_training_loss = []
    iterations = range(steps) if not show_progress else tqdm.tqdm(list(range(steps)))
    elbo_error = None
    param_raw_theta1_trace = []
    param_raw_theta2_trace = []

    def run_one_iter():
        if not use_natural_gradient:
            step_loss = svi.step(num_test_obs, expanded_test_sample_dict)
        else:
            elbo_optimizer.zero_grad()
            step_loss = loss_fn(num_test_obs, expanded_test_sample_dict)
            step_loss.backward()
            ng_precondition_grads(pyro_params)
            torch.nn.utils.clip_grad_value_(pyro_params, clip_value=1.0)
            elbo_optimizer.step()
            elbo_scheduler.step()
            step_loss = step_loss.item()
        elbo_training_loss.append(step_loss)
        cur_params = pyro.get_param_store()
        param_raw_theta1_trace.append(cur_params["my_param_raw_theta1"].detach().cpu())
        param_raw_theta2_trace.append(cur_params["my_param_raw_theta2"].detach().cpu())

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
        elbo_test_result_dict = compare_ref_and_est(elbo_vae, test_x, test_theta, test_seed)
        # vsbc
        elbo_vsbc = get_vsbc(elbo_vae, test_x, test_theta)
    else:
        elbo_test_result_dict = None
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
        "elbo_test_sample_dict": move_dict_to_cpu(expanded_test_sample_dict),
        "elbo_test_result_dict": elbo_test_result_dict,
        "elbo_vae": elbo_vae.cpu() if return_vae else None,
        "elbo_vsbc": elbo_vsbc,
        "elbo_error": elbo_error,
        "elbo_param_raw_theta1_trace": torch.stack(param_raw_theta1_trace, dim=0) if param_raw_theta1_trace else None,
        "elbo_param_raw_theta2_trace": torch.stack(param_raw_theta2_trace, dim=0) if param_raw_theta2_trace else None,
    }
