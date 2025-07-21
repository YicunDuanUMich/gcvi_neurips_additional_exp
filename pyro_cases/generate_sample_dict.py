import torch
import pyro
import click

from pathlib import Path

from pyro_cases.base_vae import BaseVAEwRegister
from pyro_cases.run import vae_dict

@click.command()
@click.option("--save-path", type=str, help="path to output file")
def main(save_path):
    output_dir = Path(save_path)
    device = torch.device("cuda:0")
    n_test_obs = 1000
    test_seed = 7272
    for k, vae in vae_dict.items():
        refer_vae = vae(hidden_dim=1, use_neural_network=False).to(device=device)
        if isinstance(refer_vae, BaseVAEwRegister):
            refer_vae.do_register(n_test_obs)
        pyro.set_rng_seed(test_seed)
        test_sample_dict = refer_vae.generate_sample_dict(batch_size=n_test_obs)
        torch.save({
            tk: tv.cpu() if isinstance(tv, torch.Tensor) else tv
            for tk, tv in test_sample_dict.items()
        }, output_dir / f"test_sample_dict_{k}.pt")


if __name__ == "__main__":
    main()
