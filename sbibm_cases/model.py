import torch
import torch.nn as nn
import math
import torch.distributions as D
from einops import repeat, rearrange
from sbibm_cases.distribution import TruncatedDiagonalMVN


class DenseEncoder1LayerGaussian(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert self.out_dim % 2 == 0
        self.hidden_dim = hidden_dim
        self.scale = torch.tensor(1 / math.sqrt(hidden_dim))
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

        # Follow proper initialization from our paper
        torch.nn.init.normal_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        eta1, eta2 = torch.mul(x, self.scale).chunk(2, dim=-1)
        # eta2 = eta2 - 1.0
        assert eta1.shape == eta2.shape
        eta2 = eta2.clamp(min=-1000.0, max=-0.1)
        return eta1, eta2
    
    @classmethod
    def eta_to_mu_sigma2(cls, eta1, eta2):
        sigma2 = -1 / (2 * eta2)
        mu = eta1 * sigma2
        return mu, sigma2
    
    def q_theta_given_x(self, eta1, eta2):
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        q_theta_given_x = D.Independent(D.Normal(mu, sigma2.sqrt()), 1)
        return q_theta_given_x
    
    @classmethod
    def gaussian_log_density_natural(cls, eta1, eta2, x):
        return eta1 * x + \
               eta2 * (x ** 2) + \
               (eta1 ** 2) / (4 * eta2) + \
               0.5 * torch.log(-eta2 / torch.pi)

    def batch_favi_loss(self, theta, x):
        eta1, eta2 = self(x)
        log_dens = self.gaussian_log_density_natural(eta1, eta2, theta)
        return -(log_dens.sum(dim=-1))
    
    def batch_elbo_loss(self, theta, x):
        raise NotImplementedError()


class DenseEncoder1LayerTruncatedGaussian(DenseEncoder1LayerGaussian):
    def __init__(self, prior_min, prior_max, **kwargs):
        super().__init__(**kwargs)

        self.prior_min = prior_min
        self.prior_max = prior_max

    def q_theta_given_x(self, eta1, eta2):
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        q_theta_given_x = TruncatedDiagonalMVN(mu, sigma2.sqrt(), 
                                               range_min=self.prior_min,
                                               range_max=self.prior_max)
        return q_theta_given_x


class DenseGaussianLinearELBO(DenseEncoder1LayerGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(**kwargs)
        self.elbo_k = elbo_k

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x):
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)
        sample_thetas = q_theta_given_x.rsample((self.elbo_k, ))  # (k, b, d)

        p_theta = D.MultivariateNormal(loc=torch.zeros(eta1.shape[-1], 
                                                       device=eta1.device), 
                                       precision_matrix=torch.inverse(0.1 * torch.eye(eta1.shape[-1], 
                                                                                      device=eta1.device))
                                       )
        log_p_theta = p_theta.log_prob(sample_thetas)  # (k, b)
        p_x_given_theta = D.Normal(sample_thetas, math.sqrt(0.1))
        log_p_x_given_theta = p_x_given_theta.log_prob(repeat(x, "b d -> k b d", 
                                                              k=self.elbo_k)).sum(dim=-1)  # (k, b)
        
        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + log_p_x_given_theta - log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )


class DenseGaussianLinearUniformELBO(DenseEncoder1LayerTruncatedGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(prior_min=-1.0, prior_max=1.0, **kwargs)
        self.elbo_k = elbo_k

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x):
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)
        sample_thetas = q_theta_given_x.rsample((self.elbo_k, )).clamp(min=self.prior_min + 1e-3, 
                                                                       max=self.prior_max - 1e-3)  # (k, b, d)

        p_theta = D.Uniform(torch.tensor([self.prior_min], device=eta1.device),
                            torch.tensor([self.prior_max], device=eta1.device))
        log_p_theta = p_theta.log_prob(sample_thetas).sum(dim=-1)  # (k, b)
        p_x_given_theta = D.Normal(sample_thetas, math.sqrt(0.1))
        log_p_x_given_theta = p_x_given_theta.log_prob(repeat(x, "b d -> k b d", 
                                                              k=self.elbo_k)).sum(dim=-1)  # (k, b)
        
        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + log_p_x_given_theta - log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )
    

class DenseSLCPELBO(DenseEncoder1LayerTruncatedGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(prior_min=-3.0, prior_max=3.0, **kwargs)
        self.elbo_k = elbo_k

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x, reordered_x=None):
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)
        sample_thetas = q_theta_given_x.rsample((self.elbo_k, )).clamp(min=self.prior_min + 1e-3, 
                                                                       max=self.prior_max - 1e-3)  # (k, b, d)

        p_theta = D.Uniform(torch.tensor([self.prior_min], device=eta1.device),
                            torch.tensor([self.prior_max], device=eta1.device))
        log_p_theta = p_theta.log_prob(sample_thetas).sum(dim=-1)  # (k, b)

        m = torch.stack((sample_thetas[..., 0], sample_thetas[..., 1]), dim=-1)  # (k, b, 2)
        s1 = sample_thetas[..., 2].abs()
        s2 = sample_thetas[..., 3].abs()
        rho = torch.tanh(sample_thetas[..., 4])
        S = torch.zeros((sample_thetas.shape[0], sample_thetas.shape[1], 2, 2), 
                        device=eta1.device)  # (k, b, 2, 2)
        S[..., 0, 0] = s1 ** 2
        S[..., 0, 1] = rho * s1 * s2
        S[..., 1, 0] = rho * s1 * s2
        S[..., 1, 1] = s2 ** 2
        S[..., 0, 0] += 1e-6
        S[..., 1, 1] += 1e-6
        p_x_given_theta = D.MultivariateNormal(loc=m, covariance_matrix=S)

        if reordered_x is not None:  # designed for DenseSLCPwDistractorELBO
            assert reordered_x.shape[-1] == 8
            x = reordered_x
        x = rearrange(x, "b (r two) -> r b two", two=2)
        log_p_x_given_theta = p_x_given_theta.log_prob(repeat(x, "r b two -> r k b two", 
                                                              k=self.elbo_k)).sum(dim=0)  # (k, b)
        
        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + log_p_x_given_theta - log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )


class DenseSLCPwDistractorELBO(DenseSLCPELBO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # from sbibm
        self.register_buffer("permutation_idx",
                             torch.load("/home/pduan/gcvi_neurips/.venv/lib/python3.10/site-packages/sbibm/"
                                        "tasks/slcp/files/permutation_idx.torch"))
        # self.register_buffer("permutation_idx", 
        #                      torch.tensor([87, 47, 39,  9, 25, 21, 26, 94, 41, 30, 34, 44, 12, 27, 89, 20,  6, 13,
        #                                     51, 40, 54,  5,  0,  2, 75, 43, 14, 97, 29, 72, 79, 99, 98,  1, 38, 65,
        #                                     83, 52, 74, 63, 19, 70,  4, 36, 96, 81, 35, 49, 31, 76, 84, 28, 11, 66,
        #                                     37, 85, 56, 60, 48, 10, 22, 82, 24,  8, 42, 32, 73,  3, 59, 95, 90, 50,
        #                                     18, 68, 45, 67, 92, 91, 17, 93, 33, 78, 88, 62, 46, 64, 57, 86, 55, 77,
        #                                     7, 80, 69, 23, 58, 71, 15, 61, 53, 16]))

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x, reordered_x=None):
        assert x.shape[-1] == self.permutation_idx.shape[0]
        reordered_x = torch.scatter(torch.zeros_like(x), 
                                    dim=1, 
                                    index=repeat(self.permutation_idx, 
                                                "d -> b d", 
                                                b=x.shape[0]), 
                                    src=x)
        return super().batch_elbo_loss(x, reordered_x=reordered_x[:, :8])


class DenseBernoulliGLMELBO(DenseEncoder1LayerGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(**kwargs)
        self.elbo_k = elbo_k

        self.register_buffer("design_matrix", 
                             torch.load("/home/pduan/gcvi_neurips/.venv/lib/python3.10/site-packages/sbibm/"
                                        "tasks/bernoulli_glm/files/design_matrix.pt"))

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x):
        # note that if you check the source code of sbibm, 
        # you will find that the x is different from what they define in their paper
        # x_2 to x_8 are not divided by x_1
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)
        sample_thetas = q_theta_given_x.rsample((self.elbo_k, ))  # (k, b, d)

        N_dim = sample_thetas.shape[-1] - 1
        D_matrix = torch.diag(torch.ones(N_dim, device=eta1.device)) - \
              torch.diag(torch.ones(N_dim - 1, device=eta1.device), -1)
        F_matrix = torch.matmul(D_matrix, D_matrix) + torch.diag(1.0 * torch.arange(N_dim, device=eta1.device) / (N_dim)) ** 0.5
        Binv = torch.zeros(size=(N_dim + 1, N_dim + 1), device=eta1.device)
        Binv[0, 0] = 0.5  # offset
        Binv[1:, 1:] = torch.matmul(F_matrix.T, F_matrix)  # filter

        p_theta = D.MultivariateNormal(loc=torch.zeros((N_dim + 1, ), device=eta1.device), 
                                       precision_matrix=Binv)
        log_p_theta = p_theta.log_prob(sample_thetas)  # (k, b)

        xT_theta = (rearrange(x, "b d -> 1 b d") * sample_thetas).sum(dim=-1)  # (k, b)
        V_theta = torch.matmul(rearrange(self.design_matrix, "d1 d2 -> 1 1 d1 d2"),
                               rearrange(sample_thetas, "k b d -> k b d 1")).squeeze(-1)  # (k, b, d1)
        sum_log_1_plus_exp_V_theta = torch.log(1 + torch.exp(V_theta)).sum(dim=-1)  # (k, b)
        log_p_x_given_theta = xT_theta - sum_log_1_plus_exp_V_theta

        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + log_p_x_given_theta - log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )


class DenseBernoulliGLMRawELBO(DenseEncoder1LayerGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(**kwargs)
        self.elbo_k = elbo_k

        self.register_buffer("design_matrix", 
                             torch.load("/home/pduan/gcvi_neurips/.venv/lib/python3.10/site-packages/sbibm/"
                                        "tasks/bernoulli_glm/files/design_matrix.pt"))

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x):
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)
        sample_thetas = q_theta_given_x.rsample((self.elbo_k, ))  # (k, b, d)

        N_dim = sample_thetas.shape[-1] - 1
        D_matrix = torch.diag(torch.ones(N_dim, device=eta1.device)) - \
              torch.diag(torch.ones(N_dim - 1, device=eta1.device), -1)
        F_matrix = torch.matmul(D_matrix, D_matrix) + torch.diag(1.0 * torch.arange(N_dim, device=eta1.device) / (N_dim)) ** 0.5
        Binv = torch.zeros(size=(N_dim + 1, N_dim + 1), device=eta1.device)
        Binv[0, 0] = 0.5  # offset
        Binv[1:, 1:] = torch.matmul(F_matrix.T, F_matrix)  # filter

        p_theta = D.MultivariateNormal(loc=torch.zeros((N_dim + 1, ), device=eta1.device), 
                                       precision_matrix=Binv)
        log_p_theta = p_theta.log_prob(sample_thetas)  # (k, b)

        psi = torch.matmul(rearrange(self.design_matrix, "a d -> 1 1 a d"),
                           rearrange(sample_thetas, "k b d -> k b d 1"))  # (k, b, a, 1)
        assert psi.shape[:2] == sample_thetas.shape[:2]
        assert psi.shape[2] == self.design_matrix.shape[0]
        assert psi.shape[3] == 1
        psi = psi.squeeze(-1)  # (k, b, a)
        p = (1 / (1 + torch.exp(-psi))).clamp(min=1e-4, max=1.0 - 1e-4)  # (k, b, a)
        x = x.unsqueeze(0)  # (1, b, a)
        log_p_x_given_theta = (x * torch.log(p) + (1 - x) * torch.log(1 - p)).sum(dim=-1)  # (k, b)

        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + log_p_x_given_theta - log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )
    

class DenseGaussianMixtureELBO(DenseEncoder1LayerTruncatedGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(prior_min=-10.0, prior_max=10.0, **kwargs)
        self.elbo_k = elbo_k

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x):
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)

        sample_thetas = q_theta_given_x.rsample((self.elbo_k, )).clamp(min=self.prior_min + 1e-3, 
                                                                       max=self.prior_max - 1e-3)  # (k, b, d)

        p_theta = D.Uniform(torch.tensor([self.prior_min], device=eta1.device),
                            torch.tensor([self.prior_max], device=eta1.device))
        log_p_theta = p_theta.log_prob(sample_thetas).sum(dim=-1)  # (k, b)

        p1_x_given_theta = D.MultivariateNormal(loc=sample_thetas, 
                                                covariance_matrix=repeat(torch.eye(eta1.shape[-1], device=eta1.device),
                                                                         "d1 d2 -> k b d1 d2", 
                                                                         k=sample_thetas.shape[0],
                                                                         b=sample_thetas.shape[1])
                                                )
        log_p1_x_given_theta = math.log(0.5) + p1_x_given_theta.log_prob(repeat(x, "b d -> k b d", k=self.elbo_k))  # (k, b)
        p2_x_given_theta = D.MultivariateNormal(loc=sample_thetas, 
                                                covariance_matrix=repeat(0.01 * torch.eye(eta1.shape[-1], device=eta1.device),
                                                                         "d1 d2 -> k b d1 d2", 
                                                                         k=sample_thetas.shape[0],
                                                                         b=sample_thetas.shape[1])
                                                )
        log_p2_x_given_theta = math.log(0.5) + p2_x_given_theta.log_prob(repeat(x, "b d -> k b d", k=self.elbo_k))  # (k, b)

        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + \
                   torch.logsumexp(torch.stack([log_p1_x_given_theta, 
                                                log_p2_x_given_theta], 
                                                dim=0), 
                                   dim=0) - \
                   log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )


class DenseTwoMoonsELBO(DenseEncoder1LayerTruncatedGaussian):
    def __init__(self, elbo_k, **kwargs):
        super().__init__(prior_min=-1.0, prior_max=1.0, **kwargs)
        self.elbo_k = elbo_k

    def batch_favi_loss(self, theta, x):
        raise NotImplementedError()
    
    def batch_elbo_loss(self, x):
        eta1, eta2 = self(x)
        q_theta_given_x = self.q_theta_given_x(eta1, eta2)

        sample_thetas = q_theta_given_x.rsample((self.elbo_k, )).clamp(min=self.prior_min + 1e-3, 
                                                                       max=self.prior_max - 1e-3)  # (k, b, d)

        p_theta = D.Uniform(torch.tensor([self.prior_min], device=eta1.device),
                            torch.tensor([self.prior_max], device=eta1.device))
        log_p_theta = p_theta.log_prob(sample_thetas).sum(dim=-1)  # (k, b)
        
        transformed_thetas = torch.stack([-(sample_thetas.sum(dim=-1).abs()),
                                          sample_thetas[..., 1] - sample_thetas[..., 0]],
                                        dim=-1) / math.sqrt(2)  # (k, b, 2)
        transformed_x = x.unsqueeze(0) - transformed_thetas  # (k, b, 2)
        transformed_x[..., 0] -= 0.25
        r = (transformed_x ** 2).sum(dim=-1).sqrt()  # (k, b)
        p_r_given_theta = D.Normal(torch.tensor([0.1], device=r.device), 0.01)
        log_p_r_given_theta = p_r_given_theta.log_prob(r)  # (k, b)

        log_q_theta_given_x = q_theta_given_x.log_prob(sample_thetas)  # (k, b)

        log_prob = log_p_theta + log_p_r_given_theta - log_q_theta_given_x
        weights = log_prob.softmax(dim=0)  # (k, b)
        return -torch.diagonal(weights.T @ log_prob)  # (b, )
