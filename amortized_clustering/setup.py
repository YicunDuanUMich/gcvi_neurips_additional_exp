import torch
import numpy as np
from generate import generate_data
from modules.invariant_encoders import SetTransformer
from modules.custom_lr_scheduler import CustomOptim2

def setup(cfg):
    # CONFIGURE
    seed_value = cfg.seed
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    # HYPERPARAMETERS & INITIALIZATION
    true_centers = torch.tensor(cfg.data.true_centers).view((-1,1))
    proportions = torch.tensor(cfg.data.proportions)
    n_centers = true_centers.size()[0]
    sigma = cfg.data.sigma
    tau = cfg.data.tau
    n_sets = cfg.data.n_sets
    n_obs = cfg.data.n_obs
    K = cfg.training.K
    true_shift = cfg.data.true_shift
    initial_shift = cfg.data.initial_shift
    device = cfg.training.device
    steps = cfg.training.steps
    parameterization = cfg.encoder.parameterization

    # SET UP MODELS AND TRAINER(S)
    _, Z, X = generate_data(true_centers, n_sets, n_obs, sigma, tau, proportions, true_shift)
    Z, X = Z.to(device), X.to(device)
    true_centers = true_centers.to(device)

    kwargs = {
        'true_centers': true_centers,
        'proportions': proportions,
        'n_centers': n_centers,
        'sigma': sigma,
        'tau': tau,
        'n_sets': n_sets,
        'n_obs': n_obs,
        'K': K,
        'true_shift': true_shift,
        'initial_shift': initial_shift,
        'device': device,
        'steps': steps,
        'Z': Z,
        'X': X,
        'parameterization': parameterization
    }

    if cfg.encoder.type == 'set':
        encoder1 = SetTransformer(n_centers, cfg.encoder.hidden_dim, cfg.encoder.num_heads, cfg.encoder.parameterization).to(device) 
        encoder2 = SetTransformer(1, cfg.encoder.hidden_dim, cfg.encoder.num_heads, cfg.encoder.parameterization).to(device)
    else:
        raise ValueError('Only implemented Set Transformer for encoder.')
    
    kwargs['encoder1'] = encoder1
    kwargs['encoder2'] = encoder2
    loss_log_name = cfg.training.loss

    logger_string = 'loss={},width={},nsets={}'.format(loss_log_name, cfg.encoder.hidden_dim, n_sets)
    encoder1.to(device)
    encoder2.to(device)
    optimizer1 = torch.optim.Adam(encoder1.parameters(), lr=cfg.training.lr1)
    optimizer2 = torch.optim.Adam(encoder2.parameters(), lr=cfg.training.lr2)
    kwargs['optimizer1'] = optimizer1
    kwargs['optimizer2'] = optimizer2
    scheduler1 = CustomOptim2(optimizer1, start_lr=cfg.training.lr1)
    scheduler2 = CustomOptim2(optimizer2, start_lr=cfg.training.lr2)

    # Select loss function
    loss_name = cfg.training.loss
    kwargs['loss'] = loss_name

    
    return (Z, 
            X, 
            logger_string,
            encoder1,
            encoder2,
            optimizer1,
            optimizer2,
            scheduler1,
            scheduler2,
            kwargs)



