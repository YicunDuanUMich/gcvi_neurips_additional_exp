import torch
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
)

class TruncatedDiagonalMVN(Distribution):
    """A truncated diagonal multivariate normal distribution."""

    def __init__(self, mu, sigma, range_min, range_max):
        """Initialize a truncated diagonal multivariate normal distribution.

        Args:
            mu (Tensor): Mean of the distribution (must be at least 1d)
            sigma (Tensor): Standard deviation of the distribution (must be at least 1d)

        Distribution is "multivariate" in that the last dimension of mu and sigma
        are considered event dimensions.
        """

        super().__init__(validate_args=False)
        multiple_normals = Normal(mu, sigma)  # all dims are batch dims, none are event
        self.base_dist = Independent(multiple_normals, 1)  # now last dim is event dim

        self.range_min = range_min
        self.range_max = range_max

        # we'll need these calculations later for log_prob
        prob_in_range = multiple_normals.cdf(self.b) - multiple_normals.cdf(self.a)
        self.log_event_prob_in_range = prob_in_range.log()
        self.log_prob_in_range = self.log_event_prob_in_range.sum(dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.base_dist.base_dist})"

    def sample(self, sample_shape=()):
        """Generate sample.

        Args:
            sample_shape (Tuple): Shape of samples to draw

        Returns:
            Tensor: (sample_shape, self.batch_shape, self.event_shape) shaped sample

        """

        shape = sample_shape + self.batch_shape + self.event_shape

        # draw using inverse cdf method
        # if Fi is the cdf of the relavant gaussian, then
        # Gi(u) = Fi(u*(F(b) - F(a)) + F(a)) is the cdf of the truncated gaussian
        uniform01_samples = torch.rand(shape, device=self.base_dist.mean.device)
        uniform_fafb = uniform01_samples * (self.upper_cdf - self.lower_cdf) + self.lower_cdf
        trunc_normal_samples = self.base_dist.base_dist.icdf(uniform_fafb)

        # if u_transformed is within machine precision of 0 or 1
        # the icdf will be -inf or inf, respectively, so we have to clamp
        return trunc_normal_samples.clamp(self.range_min + 1e-4, 
                                          self.range_max - 1e-4)
    
    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    @property
    def a(self):
        return torch.ones_like(self.base_dist.mean) * self.range_min

    @property
    def b(self):
        return torch.ones_like(self.base_dist.mean) * self.range_max

    @property
    def lower_cdf(self):
        return self.base_dist.base_dist.cdf(self.a)

    @property
    def upper_cdf(self):
        return self.base_dist.base_dist.cdf(self.b)

    @property
    def mean(self):
        mu = self.base_dist.mean
        offset = self.base_dist.log_prob(self.a).exp() - self.base_dist.log_prob(self.b).exp()
        offset /= self.log_prob_in_range.exp()
        return mu + (offset.unsqueeze(-1) * self.base_dist.stddev)

    @property
    def stddev(self):
        # See https://arxiv.org/pdf/1206.5387.pdf for the formula for the variance of a truncated
        # multivariate normal. The covariance terms simplify since our dimensions are independent,
        # but it's still tricky to compute.
        raise NotImplementedError("Standard deviation for truncated normal is not implemented yet")

    @property
    def mode(self):
        # a mode still exists if this assertion is false, but I haven't implemented code
        # to compute it because I don't think we need it
        assert (self.base_dist.mean >= self.range_min).all() and (self.base_dist.mean <= self.range_max).all()
        return self.base_dist.mode

    @property
    def batch_shape(self):
        return self.base_dist.batch_shape

    @property
    def event_shape(self):
        return self.base_dist.event_shape

    def log_prob(self, value):
        assert (value >= self.range_min).all() and (value <= self.range_max).all()
        # subtracting log probability that the base RV is in the unit box
        # is equivalent in log space to dividing the normal pdf by the normalizing constant
        return self.base_dist.log_prob(value) - self.log_prob_in_range

    def cdf(self, value):
        cdf_at_val = self.base_dist.base_dist.cdf(value)
        cdf_at_lb = self.lower_cdf
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log().sum(dim=-1) - self.log_prob_in_range
        return log_cdf.exp()

    def event_cdf(self, value):
        cdf_at_val = self.base_dist.base_dist.cdf(value)
        cdf_at_lb = self.lower_cdf
        log_cdf = (cdf_at_val - cdf_at_lb + 1e-9).log() - self.log_event_prob_in_range
        return log_cdf.exp()

    def event_icdf(self, value):
        assert isinstance(value, torch.Tensor)
        assert value.shape == self.lower_cdf.shape
        assert (value > self.range_min).all() and (value < self.range_max).all()
        converted_cdf = value * (self.upper_cdf - self.lower_cdf) + self.lower_cdf
        converted_icdf = self.base_dist.base_dist.icdf(converted_cdf)
        return converted_icdf.clamp(self.a, self.b)