import torch
import torch.distributions.constraints as torch_constraints
import pyro.distributions as dist

from collections import OrderedDict
from einops import rearrange


class BetawCDF(dist.Beta):
    @classmethod
    def _linspace_x(cls, value: torch.Tensor, steps=100):
        assert value.min() >= 0.0 and value.max() <= 1
        return rearrange((value / steps).unsqueeze(-1) * torch.arange(0, steps + 1, device=value.device),
                         "... steps -> steps ...")

    def cdf(self, value):
        x = self._linspace_x(value)  # (steps, ...)
        prob = self.log_prob(x).exp()
        return torch.trapezoid(prob, x, dim=0)


class VariationalFactor:
    def __init__(self, name, batch_shape, event_shape):
        self.name = name
        assert len(batch_shape) >= 1
        self.batch_shape = OrderedDict([(f"b{i}", b) for i, b in enumerate(batch_shape[:-1])])
        batch_tag = " ".join(list(self.batch_shape.keys()))
        self.event_shape = OrderedDict([(f"e{i}", e) for i, e in enumerate(event_shape)])
        event_tag = " ".join(list(self.event_shape.keys()))
        if event_tag == "" and batch_tag == "":
            self.rearrange_str = None
        else:
            self.rearrange_str = f"b ({batch_tag} {event_tag}) -> {batch_tag} b {event_tag}"

    def batch_favi_loss(self, theta, raw_pred):
        raise NotImplementedError()
    
    def return_dist(self, raw_pred):
        raise NotImplementedError()
    
    # designed for testing
    def get_est_theta(self, raw_pred):
        raise NotImplementedError()
    
    # designed for testing
    def return_unrearranged_dist(self, raw_pred):
        raise NotImplementedError()
    

class NormalFactor(VariationalFactor):
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

    @classmethod
    def get_eta(cls, raw_pred):
        assert raw_pred.ndim == 3  # (b, k, 2)
        assert raw_pred.shape[-1] == 2
        eta1, eta2 = raw_pred[..., 0], raw_pred[..., 1]
        eta2 = (eta2 - 1.0).clamp(min=-1000.0, max=-0.1)
        return eta1, eta2

    @classmethod
    def eta_to_mu_sigma2(cls, eta1, eta2):
        sigma2 = -1 / (2 * eta2)
        mu = eta1 * sigma2
        return mu, sigma2

    def batch_favi_loss(self, theta, raw_pred):
        eta1, eta2 = self.get_eta(raw_pred)
        assert theta.shape == eta1.shape
        log_dens = self.gaussian_log_density_natural(eta1, eta2, theta)
        return -(log_dens.sum(dim=-1))
    
    def return_dist(self, raw_pred):
        eta1, eta2 = self.get_eta(raw_pred)
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        if self.rearrange_str is not None:
            mu = rearrange(mu, 
                           self.rearrange_str,
                           **self.batch_shape,
                           **self.event_shape)
            sigma2 = rearrange(sigma2,
                               self.rearrange_str,
                               **self.batch_shape,
                               **self.event_shape)
        else:
            assert mu.shape[-1] == 1 and sigma2.shape[-1] == 1
            mu = mu.squeeze(-1)
            sigma2 = sigma2.squeeze(-1)
        distri = dist.Normal(loc=mu, scale=sigma2.sqrt())
        event_ndim = len(self.event_shape)
        if event_ndim > 0:
            distri = distri.to_event(event_ndim)
        return distri
    
    def get_est_theta(self, raw_pred):
        eta1, eta2 = self.get_eta(raw_pred)
        return self.eta_to_mu_sigma2(eta1, eta2)

    def return_unrearranged_dist(self, raw_pred):
        mu, sigma2 = self.get_est_theta(raw_pred)
        return dist.Normal(loc=mu, scale=sigma2.sqrt())
    

class NormalFactorMuSigmaParam(NormalFactor):
    @classmethod
    def get_eta(cls, raw_pred):
        mu = raw_pred[..., 0]
        sigma2 = raw_pred[..., 1] ** 2
        eta1 = mu / sigma2
        eta2 = -1 / (2 * sigma2)
        return eta1, eta2


class BetaFactor(VariationalFactor):
    def __init__(self, name, batch_shape, event_shape, *, low, high):
        super().__init__(name, batch_shape, event_shape)
        self.low = low
        self.high = high
        assert self.low < self.high

    @classmethod
    def get_alpha_beta(cls, raw_pred):
        assert raw_pred.ndim == 3  # (b, k, 2)
        assert raw_pred.shape[-1] == 2
        alpha = (raw_pred[..., 0].abs() + 1.0)
        beta = (raw_pred[..., 1].abs() + 1.0)
        return alpha, beta
    
    def _get_beta_dist(self, alpha, beta):
        distri = BetawCDF(alpha, beta)
        return dist.TransformedDistribution(distri,
                                              [dist.transforms.AffineTransform(loc=self.low, 
                                                                               scale=self.high - self.low)])

    def batch_favi_loss(self, theta, raw_pred):
        alpha, beta = self.get_alpha_beta(raw_pred)
        assert alpha.shape == theta.shape
        distri = self._get_beta_dist(alpha, beta)
        assert not ((theta < self.low) | (theta > self.high)).any()
        theta_clamped = theta.clamp(min=self.low + 1e-4, max=self.high - 1e-4)
        log_prob = distri.log_prob(theta_clamped)
        return -(log_prob.sum(dim=-1))
    
    def return_dist(self, raw_pred):
        alpha, beta = self.get_alpha_beta(raw_pred)
        if self.rearrange_str is not None:
            alpha = rearrange(alpha,
                              self.rearrange_str,
                              **self.batch_shape,
                              **self.event_shape)
            beta = rearrange(beta,
                             self.rearrange_str,
                             **self.batch_shape,
                             **self.event_shape)
        else:
            assert alpha.shape[-1] == 1 and beta.shape[-1] == 1
            alpha = alpha.squeeze(-1)
            beta = beta.squeeze(-1)
        distri = self._get_beta_dist(alpha, beta)
        event_ndim = len(self.event_shape)
        if event_ndim > 0:
            distri = distri.to_event(event_ndim)
        return distri
    
    def get_est_theta(self, raw_pred):
        return self.get_alpha_beta(raw_pred)
    
    def return_unrearranged_dist(self, raw_pred):
        alpha, beta = self.get_alpha_beta(raw_pred)
        return self._get_beta_dist(alpha, beta)
    

class LogNormalFactor(VariationalFactor):
    def __init__(self, name, batch_shape, event_shape, *, low):
        super().__init__(name, batch_shape, event_shape)
        self.low = low

    @classmethod
    def eta_to_mu_sigma2(cls, eta1, eta2):
        sigma2 = -1 / (2 * eta2)
        mu = eta1 * sigma2
        return mu, sigma2

    @classmethod
    def get_eta(cls, raw_pred):
        assert raw_pred.ndim == 3  # (b, k, 2)
        assert raw_pred.shape[-1] == 2
        eta1, eta2 = raw_pred[..., 0], raw_pred[..., 1]
        eta2 = (eta2 - 1.0).clamp(min=-1000.0, max=-0.1)
        return eta1, eta2

    @classmethod
    def eta_to_mu_sigma2(cls, eta1, eta2):
        sigma2 = -1 / (2 * eta2)
        mu = eta1 * sigma2
        return mu, sigma2
    
    def _get_log_normal_dist(self, mu, sigma2):
        distri = dist.LogNormal(loc=mu, scale=sigma2.sqrt())
        return dist.TransformedDistribution(distri,
                                              [dist.transforms.AffineTransform(loc=self.low, 
                                                                               scale=1)])
    
    def batch_favi_loss(self, theta, raw_pred):
        assert theta.min() > self.low
        eta1, eta2 = self.get_eta(raw_pred)
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        assert mu.shape == theta.shape
        distri = self._get_log_normal_dist(mu, sigma2)
        log_prob = distri.log_prob(theta)
        return -(log_prob.sum(dim=-1))
    
    def return_dist(self, raw_pred):
        eta1, eta2 = self.get_eta(raw_pred)
        mu, sigma2 = self.eta_to_mu_sigma2(eta1, eta2)
        if self.rearrange_str is not None:
            mu = rearrange(mu,
                              self.rearrange_str,
                              **self.batch_shape,
                              **self.event_shape)
            sigma2 = rearrange(sigma2,
                             self.rearrange_str,
                             **self.batch_shape,
                             **self.event_shape)
        else:
            assert mu.shape[-1] == 1 and sigma2.shape[-1] == 1
            mu = mu.squeeze(-1)
            sigma2 = sigma2.squeeze(-1)
        distri = self._get_log_normal_dist(mu, sigma2)
        event_ndim = len(self.event_shape)
        if event_ndim > 0:
            distri = distri.to_event(event_ndim)
        return distri
    
    def get_est_theta(self, raw_pred):
        eta1, eta2 = self.get_eta(raw_pred)
        return self.eta_to_mu_sigma2(eta1, eta2)
    
    def return_unrearranged_dist(self, raw_pred):
        mu, sigma2 = self.get_est_theta(raw_pred)
        return self._get_log_normal_dist(mu, sigma2)


class VariationalDist:
    def __init__(self, sample_dict_instr, *, eta_param_for_normal=True):
        self.variational_factors: list[VariationalFactor] = []
        self.latent_size: list[int] = []
        self.eta_param_for_normal = eta_param_for_normal
        for k, v in sample_dict_instr["latent"].items():
            suitable_factor = self._get_factor_for_constraint(v["constraint"])
            self.variational_factors.append(suitable_factor(k, v["batch_shape"], v["event_shape"]))
            self.latent_size.append(v["latent_size"])

    def _get_factor_for_constraint(self, constraint):
        if isinstance(constraint, torch_constraints.independent):
            return self._get_factor_for_constraint(constraint.base_constraint)
        elif isinstance(constraint, type(torch_constraints.real)):
            if not self.eta_param_for_normal:
                return NormalFactorMuSigmaParam
            else:
                return NormalFactor
        elif isinstance(constraint, torch_constraints.interval):
            assert isinstance(constraint.lower_bound, torch.Tensor)
            if constraint.lower_bound.ndim == 0:
                low = constraint.lower_bound.item()
            else:
                assert constraint.lower_bound.ndim == 1
                assert (constraint.lower_bound == constraint.lower_bound[0]).all()
                low = constraint.lower_bound[0].item()
            assert isinstance(constraint.upper_bound, torch.Tensor)
            if constraint.upper_bound.ndim == 0:
                high = constraint.upper_bound.item()
            else:
                assert constraint.upper_bound.ndim == 1
                assert (constraint.upper_bound == constraint.upper_bound[0]).all()
                high = constraint.upper_bound[0].item()
            return lambda *args: BetaFactor(*args, low=low, high=high)
        elif isinstance(constraint, torch_constraints.greater_than):
            assert isinstance(constraint.lower_bound, torch.Tensor)
            if constraint.lower_bound.ndim == 0:
                low = constraint.lower_bound.item()
            else:
                assert constraint.lower_bound.ndim == 1
                assert (constraint.lower_bound == constraint.lower_bound[0]).all()
                low = constraint.lower_bound[0].item()
            return lambda *args: LogNormalFactor(*args, low=low)
        else:
            raise NotImplementedError()

    def batch_favi_loss(self, theta: torch.Tensor, raw_pred: torch.Tensor):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert theta.shape == raw_pred.shape[:-1]
        assert sum(self.latent_size) == theta.shape[1]

        loss_list = []
        for factor, sub_theta, sub_raw_pred in zip(self.variational_factors, 
                                                   torch.split(theta, self.latent_size, dim=1),
                                                   torch.split(raw_pred, self.latent_size, dim=1), 
                                                   strict=True):
            loss_list.append(factor.batch_favi_loss(sub_theta, sub_raw_pred))
        return torch.stack(loss_list, dim=-1).mean(dim=-1)
    
    def return_dist(self, raw_pred: torch.Tensor, latent_name: str):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert raw_pred.shape[-2] == sum(self.latent_size)

        latent_index = 0
        for factor in self.variational_factors:
            if factor.name == latent_name:
                break
            latent_index += 1
            if latent_index == len(self.variational_factors):
                raise ValueError(f"can't find latent_name: {latent_name}")
        latent_factor = self.variational_factors[latent_index]
        return latent_factor.return_dist(torch.split(raw_pred, self.latent_size, dim=1)[latent_index])
    
    # for testing
    def get_theta(self, raw_pred: torch.Tensor):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert raw_pred.shape[-2] == sum(self.latent_size)

        theta1_list = []
        theta2_list = []
        for factor, sub_raw_pred in zip(self.variational_factors,
                                        torch.split(raw_pred, self.latent_size, dim=1), 
                                        strict=True):
            theta = factor.get_est_theta(sub_raw_pred)
            theta1_list.append(theta[0])
            theta2_list.append(theta[1])
        return torch.cat(theta1_list, dim=-1), torch.cat(theta2_list, dim=-1)
    
    # for testing
    def get_mu_sigma2(self, raw_pred: torch.Tensor):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert raw_pred.shape[-2] == sum(self.latent_size)

        mu_list = []
        sigma2_list = []
        for factor, sub_raw_pred in zip(self.variational_factors,
                                        torch.split(raw_pred, self.latent_size, dim=1), 
                                        strict=True):
            theta = factor.get_est_theta(sub_raw_pred)
            if isinstance(factor, (NormalFactor, NormalFactorMuSigmaParam)):
                mu_list.append(theta[0])
                sigma2_list.append(theta[1])
            elif isinstance(factor, BetaFactor):
                alpha, beta = theta[0], theta[1]
                mu_list.append(alpha / (alpha + beta))
                sigma2_list.append(alpha * beta / (alpha + beta) ** 2 * (alpha + beta + 1))
            elif isinstance(factor, LogNormalFactor):
                mu_list.append(torch.exp(theta[0] + theta[1] / 2))
                sigma2_list.append((torch.exp(theta[1]) - 1) * torch.exp(2 * theta[0] + theta[1]))
            else:
                raise NotImplementedError()
        return torch.cat(mu_list, dim=-1), torch.cat(sigma2_list, dim=-1)
    
    # for vsbc testing
    def return_unrearranged_dist(self, raw_pred: torch.Tensor):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert raw_pred.shape[-2] == sum(self.latent_size)

        distri_list = []
        for factor, sub_raw_pred in zip(self.variational_factors,
                                        torch.split(raw_pred, self.latent_size, dim=1), 
                                        strict=True):
            distri_list.append(factor.return_unrearranged_dist(sub_raw_pred))
        return distri_list
    
    # for vsbc testing
    def get_vsbc(self, raw_pred: torch.Tensor, true_theta: torch.Tensor):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert raw_pred.shape[-2] == sum(self.latent_size)
        assert raw_pred.shape[:-1] == true_theta.shape

        est_dist_list = self.return_unrearranged_dist(raw_pred)
        vsbc_list = []
        for est_dist, sub_true_theta in zip(est_dist_list,
                                            torch.split(true_theta, self.latent_size, dim=1),
                                            strict=True):
            vsbc_list.append(1 - est_dist.cdf(sub_true_theta))
        return torch.cat(vsbc_list, dim=-1)  # (b, k)
    
    # for testing
    def draw_samples(self, raw_pred: torch.Tensor, sample_n: int):
        assert raw_pred.ndim == 3
        assert raw_pred.shape[-1] == 2
        assert raw_pred.shape[-2] == sum(self.latent_size)

        est_dist_list = self.return_unrearranged_dist(raw_pred)
        samples_list = []
        for est_dist in est_dist_list:
            samples_list.append(est_dist.sample((sample_n,)))
        return torch.cat(samples_list, dim=-1).permute([1, 2, 0])  # (b, k, n_samples)
