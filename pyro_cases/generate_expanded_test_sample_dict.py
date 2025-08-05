from pyro_cases.utils.run_amortized_favi_with_fixed_design import train_and_test_amortized_favi_with_fixed_design
from pyro_cases.utils.vae_dict import vae_dict
import torch

def move_dict_to_cpu(pre_dict: dict):
    return {
        k: v.to(device="cpu") if isinstance(v, torch.Tensor) else v
        for k, v in pre_dict.items()
    }

if __name__ == "__main__":
    test_sample_dicts = {}
    for i, task_name in enumerate(vae_dict.keys()):
        print(f"[{i}] run task {task_name}")
        out_dict = train_and_test_amortized_favi_with_fixed_design(task_name=task_name,
                                                                    seed=123456,
                                                                    device="cuda:0",
                                                                    lr=1e-3,
                                                                    lr_schedule="cosine_annealing",
                                                                    batch_size=1024,
                                                                    network_width=256,
                                                                    steps=5,
                                                                    test_seed=7272,
                                                                    num_test_obs=1000,
                                                                    show_progress=False,
                                                                    silent=False,
                                                                    return_vae=True,
                                                                    suppress_error=False,
                                                                    nn_type="deep_set")
        test_sample_dicts[task_name] = move_dict_to_cpu(out_dict["favi_test_sample_dict"])
    torch.save(test_sample_dicts, 
               "/scratch/regier_root/regier0/pduan/gcvi_08-02_deep_set_favi_with_fixed_design_test_sample_dicts.pt")