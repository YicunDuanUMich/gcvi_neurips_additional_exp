import torch
import math
from torch import nn
import pyro
import copy
import pyro.distributions as dist
import contextlib

from einops import rearrange, repeat
from typing import Dict
from collections import OrderedDict, UserDict

from pyro_cases.utils.nn_model import SetTransformer, DenseEncoderGaussian, DeepSetMLP
from pyro_cases.utils.variational_dist import VariationalDist


class BaseVAE(nn.Module):
    x_dim = None
    theta_dim = None
    special_x_process_flag = False
    
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_param", torch.zeros(0))
        # only for post-training elbo loss calculate
        self.use_fixed_raw_pred = False
        self.fixed_raw_pred = None
    
    # only for post-training elbo loss calculate
    def set_fixed_raw_pred(self, raw_pred):
        self.use_fixed_raw_pred = True
        self.fixed_raw_pred = raw_pred
    
    @property
    def device(self):
        return self.dummy_param.device

    def model(self, batch_size, sample_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError()

    def guide(self, batch_size, sample_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError()

    def get_obs_sample_dict(self, obs_seed):
        pyro.set_rng_seed(obs_seed)
        return self.generate_sample_dict(batch_size=1)

    def get_observation(self, obs_seed):
        sample_dict = self.get_obs_sample_dict(obs_seed)
        return self.extract_x(sample_dict), self.extract_theta(sample_dict)

    def generate_sample_dict(self, batch_size):
        return self.model(batch_size=batch_size, sample_dict=None)
    
    def _extract_x_func(self, sample_dict):
        raise NotImplementedError()
    
    def extract_x(self, sample_dict):
        x = self._extract_x_func(sample_dict)
        assert x.shape[-1] == self.x_dim
        return x
    
    def extract_x_as_set(self, batch_size, sample_dict):
        raise NotImplementedError()
    
    def _extract_theta_func(self, sample_dict):
        raise NotImplementedError()
    
    def extract_theta(self, sample_dict):
        theta = self._extract_theta_func(sample_dict)
        assert theta.shape[-1] == self.theta_dim
        return theta
    

class PlateTracker:
    def __init__(self):
        self.plate_stack = []

    def enter_plate(self, name):
        self.plate_stack.append(name)

    def exit_plate(self, name):
        popped = self.plate_stack.pop()
        assert popped == name, f"plate stack mismatch: expected {name}, got {popped}"

    def get_plate_stack(self):
        return copy.deepcopy(self.plate_stack)
    

class WrappedPlate:
    def __init__(self, 
                 tracker: PlateTracker, 
                 plate_name: str, 
                 plate_size: int, 
                 plate_dim: int, 
                 **plate_kwargs):
        self.tracker = tracker
        self.plate = pyro.plate(plate_name, plate_size, dim=plate_dim, **plate_kwargs)
        self._cm = None

    def __enter__(self):
        self.tracker.enter_plate(self.plate.name)
        self._cm = self.plate.__enter__()
        return self._cm

    def __exit__(self, exc_type, exc_val, traceback):
        self.plate.__exit__(exc_type, exc_val, traceback)
        self.tracker.exit_plate(self.plate.name)


class BaseVAEwRegister(BaseVAE):    
    def __init__(self, hidden_dim, use_neural_network=True, nn_type="set_transformer"):
        super().__init__()

        self.use_neural_network = use_neural_network
        self.nn_type = nn_type
        if self.use_neural_network:
            match nn_type:
                case "set_transformer":
                    self.encoder = SetTransformer(n_out=self.theta_dim, 
                                                    hidden_dim=hidden_dim, 
                                                    num_heads=4)
                case "dense_gaussian":
                    self.encoder = DenseEncoderGaussian(n_out=self.theta_dim, 
                                                        hidden_dim=hidden_dim)
                case "deep_set":
                    self.encoder = DeepSetMLP(n_out=self.theta_dim,
                                              hidden_dim=hidden_dim,
                                              n_layers=2)
                case _:
                    raise NotImplementedError()
        else:
            self.encoder = None

        self.sample_dict_instr = {
            "meta_data": [],
            "data": [],
            "latent": OrderedDict(),
            "obs": OrderedDict(),
        }
        self.plate_tracker = PlateTracker()
        self.already_registered = False
        self.meta_data_skip_set = set()
        self.data_skip_set = set()
        self.variational_dist: VariationalDist = None

    def plate(self, name, size, dim, **plate_kwargs):
        return WrappedPlate(self.plate_tracker, 
                            name,
                            size,
                            dim,
                            **plate_kwargs)

    def get_plates(self, batch_size, sample_dict) -> Dict[str, pyro.plate]:
        raise NotImplementedError()

    def guide(self, batch_size, sample_dict: Dict[str, torch.Tensor]):
        if not self.use_fixed_raw_pred:
            if self.use_neural_network:
                pyro.module("encoder", self.encoder)
                match self.nn_type:
                    case "set_transformer" | "deep_set":
                        x = self.extract_x_as_set(batch_size, sample_dict)
                    case "dense_gaussian":
                        x = self.extract_x(sample_dict)
                    case _:
                        raise NotImplementedError()
                assert batch_size == x.shape[0]
                raw_pred = self.encoder(x)
            else:
                raw_pred1 = pyro.param("param_theta1", 
                                       lambda: (torch.rand(batch_size, 
                                                           self.theta_dim, 
                                                           device=self.device) - 0.5) * 20)
                raw_pred2 = pyro.param("param_theta2", 
                                        lambda: (torch.rand(batch_size, 
                                                            self.theta_dim, 
                                                            device=self.device) + 1e-3) * 100, 
                                        constraint=dist.constraints.positive)
                raw_pred = torch.stack([raw_pred1, raw_pred2], dim=-1)
        else:
            raw_pred = self.fixed_raw_pred

        plates = self.get_plates(batch_size, sample_dict)
        for kp, p in plates.items():
            assert isinstance(p, WrappedPlate), f"plate {kp} is not WrappedPlate"
        for latent_name, latent_details in self.sample_dict_instr["latent"].items():
            cur_plates = [plates[p] for p in latent_details["plates"]]
            with contextlib.ExitStack() as s:
                cms = [s.enter_context(p) for p in cur_plates]
                distri = self.variational_dist.return_dist(raw_pred, latent_name)
                pyro.sample(latent_name, distri)
    
    def _extract_x_func(self, sample_dict):
        if self.special_x_process_flag:
            assert len(self.sample_dict_instr["obs"].keys()) == 1
            x = sample_dict[list(self.sample_dict_instr["obs"].keys())[0]]
            assert x.ndim == 2
            return x
        x = []
        for data_name in self.sample_dict_instr["data"]:
            x.append(rearrange(sample_dict[data_name], "... b -> b (...)"))
        for obs_name, obs_details in self.sample_dict_instr["obs"].items():
            event_shape = " ".join([f"e{i}" for i in range(len(obs_details["event_shape"]))])
            x.append(rearrange(sample_dict[obs_name], f"... b {event_shape} -> b (... {event_shape})"))
        return torch.cat(x, dim=-1)
    
    def extract_x_as_set(self, batch_size, sample_dict):
        if self.special_x_process_flag:
            assert len(self.sample_dict_instr["obs"].keys()) == 1
            x = sample_dict[list(self.sample_dict_instr["obs"].keys())[0]]
            assert x.ndim == 2
            return x.unsqueeze(-1)
        x = []
        assert "N" in self.sample_dict_instr["meta_data"]
        N = sample_dict["N"]

        for data_name in self.sample_dict_instr["data"]:
            input_data = sample_dict[data_name]
            if input_data.ndim == 1:
                x.append(repeat(input_data, "b -> b n", n=N))
                continue
            if input_data.shape[0] == N:
                x.append(rearrange(input_data, "n b -> b n"))
            else:
                if data_name not in self.data_skip_set:
                    print(f"WARNING: the input doesn't involve data {data_name}")
                    self.data_skip_set.add(data_name)

        for obs_name, _obs_details in self.sample_dict_instr["obs"].items():
            x.append(rearrange(sample_dict[obs_name], "n b -> b n"))

        for meta_data_name in self.sample_dict_instr["meta_data"]:
            meta_data = sample_dict[meta_data_name]
            if not isinstance(meta_data, torch.Tensor):
                continue
            if meta_data.ndim == 1 and meta_data.shape[0] == N:
                x.append(repeat(meta_data, "n -> b n", b=batch_size))
            else:
                if meta_data_name not in self.meta_data_skip_set:
                    print(f"WARNING: the input doesn't involve meta data {meta_data_name}")
                    self.meta_data_skip_set.add(meta_data_name)

        return torch.stack(x, dim=-1)
    
    def _extract_theta_func(self, sample_dict):
        theta = []
        for latent_name, latent_details in self.sample_dict_instr["latent"].items():
            event_shape = " ".join([f"e{i}" for i in range(len(latent_details["event_shape"]))])
            cur_theta = rearrange(sample_dict[latent_name], f"... b {event_shape} -> b (... {event_shape})")
            theta.append(cur_theta)
            if not self.already_registered:
                self.sample_dict_instr["latent"][latent_name]["latent_size"] = cur_theta.shape[-1]
            else:
                assert self.sample_dict_instr["latent"][latent_name]["latent_size"] is not None
        return torch.cat(theta, dim=-1)

    def r_data(self, name, data):
        if not self.already_registered:
            assert name not in self.sample_dict_instr["data"]
            self.sample_dict_instr["data"].append(name)
        else:
            assert name in self.sample_dict_instr["data"]
        return data
    
    def r_meta_data(self, name, meta_data):
        if not self.already_registered:
            assert name not in self.sample_dict_instr["meta_data"]
            self.sample_dict_instr["meta_data"].append(name)
        else:
            assert name in self.sample_dict_instr["meta_data"]
        return meta_data
    
    def r_sample(self, name, sample_dist):
        if not self.already_registered:
            plates = self.plate_tracker.get_plate_stack()
            event_shape = list(sample_dist.event_shape)
            sample = pyro.sample(name, sample_dist)
            batch_shape = list(sample.shape) if len(event_shape) == 0 else list(sample.shape[:-len(event_shape)])
            assert name not in self.sample_dict_instr["latent"]
            self.sample_dict_instr["latent"][name] = {
                "dist": type(sample_dist),
                "constraint": sample_dist.support,
                "batch_shape": batch_shape,
                "event_shape": event_shape,
                "latent_size": None,
                "plates": plates
            }
        else:
            sample = pyro.sample(name, sample_dist)
            assert name in self.sample_dict_instr["latent"]
        return sample
    
    def r_obs(self, name, sample_dist, obs):
        if not self.already_registered:
            plates = self.plate_tracker.get_plate_stack()
            event_shape = list(sample_dist.event_shape)
            sample = pyro.sample(name, sample_dist, obs=obs)
            batch_shape = list(sample.shape) if len(event_shape) == 0 else list(sample.shape[:-len(event_shape)])
            assert name not in self.sample_dict_instr["obs"]
            self.sample_dict_instr["obs"][name] = {
                "dist": type(sample_dist),
                "batch_shape": batch_shape,
                "event_shape": event_shape,
                "plates": plates
            }
        else:
            sample = pyro.sample(name, sample_dist, obs=obs)
            assert name in self.sample_dict_instr["obs"]
        return sample

    def do_register(self, batch_size):
        assert not self.already_registered
        sample_dict = self.generate_sample_dict(batch_size)
        self.extract_theta(sample_dict)
        self.variational_dist = VariationalDist(self.sample_dict_instr)
        self.already_registered = True
    
    def scalar_normal_dist(self, loc, scale):
        return dist.Normal(loc=torch.tensor(loc, device=self.device),
                           scale=torch.tensor(scale, device=self.device))
    

class SampleDict(UserDict):
    def __init__(self):
        super().__init__()
        self.vae = None
        self.context_flag = "no_context"

    def __setitem__(self, key, item):
        match self.context_flag:
            case "meta_data":
                assert self.vae is not None
                self.vae.r_meta_data(key, item)
            case "data":
                assert self.vae is not None
                self.vae.r_data(key, item)
            case "no_context":
                assert self.vae is None
            case _:
                raise ValueError("context flag is invalid")
        return super().__setitem__(key, item)
    

class MetaDataContext:
    def __init__(self, sample_dict: SampleDict, vae: BaseVAEwRegister):
        self.sample_dict = sample_dict
        assert isinstance(self.sample_dict, SampleDict)
        self.vae = vae
        assert isinstance(self.vae, BaseVAEwRegister)

    def __enter__(self):
        assert self.sample_dict.context_flag == "no_context"
        assert self.sample_dict.vae is None
        self.sample_dict.context_flag = "meta_data"
        self.sample_dict.vae = self.vae
        return self.sample_dict
    
    def __exit__(self, exc_type, exc_val, traceback):
        assert self.sample_dict.context_flag == "meta_data"
        assert self.sample_dict.vae is not None
        self.sample_dict.context_flag = "no_context"
        self.sample_dict.vae = None


class DataContext:
    def __init__(self, sample_dict: SampleDict, vae: BaseVAEwRegister):
        self.sample_dict = sample_dict
        assert isinstance(self.sample_dict, SampleDict)
        self.vae = vae
        assert isinstance(self.vae, BaseVAEwRegister)

    def __enter__(self):
        assert self.sample_dict.context_flag == "no_context"
        assert self.sample_dict.vae is None
        self.sample_dict.context_flag = "data"
        self.sample_dict.vae = self.vae
        return self.sample_dict
    
    def __exit__(self, exc_type, exc_val, traceback):
        assert self.sample_dict.context_flag == "data"
        assert self.sample_dict.vae is not None
        self.sample_dict.context_flag = "no_context"
        self.sample_dict.vae = None
