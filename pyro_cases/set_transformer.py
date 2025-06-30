import math
import torch
import torch.nn as nn

'''Source: https://github.com/juho-lee/set_transformer/blob/master/modules.py'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    '''
    See construction in modules.py
    '''
    def __init__(self, n_out, hidden_dim, num_heads):
        super().__init__()

        self.input_process = None
        self.hidden_dim = hidden_dim
        self.register_buffer("dummy_param", torch.zeros(0))
        self.enc = nn.Sequential(
            SAB(dim_in=hidden_dim, dim_out=hidden_dim, num_heads=num_heads),
            SAB(dim_in=hidden_dim, dim_out=hidden_dim, num_heads=num_heads),
        )
        self.dec = nn.Sequential(
            PMA(dim=hidden_dim, num_heads=num_heads, num_seeds=n_out),
            SAB(dim_in=hidden_dim, dim_out=hidden_dim, num_heads=num_heads),
            SAB(dim_in=hidden_dim, dim_out=hidden_dim, num_heads=num_heads),
        )
        self.final_linear = nn.Linear(in_features=hidden_dim, out_features=2)
        torch.nn.init.zeros_(self.final_linear.weight)

        self.scale = 1 / math.sqrt(hidden_dim)

    @property
    def device(self):
        return self.dummy_param.device
    
    @classmethod
    def eta_to_mu_sigma2(cls, eta1, eta2):
        sigma2 = -1 / (2 * eta2)
        mu = eta1 * sigma2
        return mu, sigma2
    
    @classmethod
    def gaussian_log_density_natural(cls, eta1, eta2, x):
        return eta1 * x + \
               eta2 * (x ** 2) + \
               (eta1 ** 2) / (4 * eta2) + \
               0.5 * torch.log(-eta2 / torch.pi)

    def get_eta(self, x):
        assert x.ndim == 3
        if self.input_process is None:
            self.input_process = nn.Sequential(
                nn.Linear(x.shape[-1], self.hidden_dim),
                nn.SELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            ).to(device=self.device)
        x = self.input_process(x)
        x = self.enc(x)
        out = self.final_linear(self.dec(x))
        out = out * self.scale
        eta1, eta2 = out[..., 0], out[..., 1]
        eta2 = (eta2 - 1.0).clamp(min=-1000.0, max=-0.1)
        return eta1, eta2
    
    def forward(self, x):
        eta1, eta2 = self.get_eta(x)
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        return mu, sigma2.sqrt()

    def batch_favi_loss(self, theta, x):
        eta1, eta2 = self.get_eta(x)
        log_dens = self.gaussian_log_density_natural(eta1, eta2, theta)
        return -(log_dens.sum(dim=-1))
