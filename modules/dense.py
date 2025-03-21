import math
import torch
import torch.nn as nn

from canonical_gaussian.utils import MyGetJacobian
from einops import rearrange, reduce


class ScaleLayer(nn.Module):

   def __init__(self, width=128):
       super().__init__()
       self.scale = torch.tensor(1/math.sqrt(width))

   def forward(self, input):
       return torch.mul(input, self.scale)

class DenseEncoder1Layer(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden_layer=1, min_conc=-torch.inf, max_conc=torch.inf):
        super(DenseEncoder1Layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.scale = torch.tensor(1/math.sqrt(hidden_dim))
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()
        self.min_conc = min_conc
        self.max_conc = max_conc
        self.n_hidden_layer = n_hidden_layer

        # Follow proper initialization from our paper
        torch.nn.init.normal_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear3.weight)

    def forward(self, x):
        new = self.linear1(x)
        new = self.relu(new)
        new = self.linear3(new)
        new = torch.mul(new, self.scale)
        return new
    
    def get_log_prob(self, theta, x):
        '''
        Construct natural parameterization.
        '''
        eta = self.forward(x)+1e-4
        Ttheta = rearrange([torch.cos(theta), torch.sin(theta)], 'dim b -> b dim')
        kappa = torch.sqrt(reduce(torch.square(eta), 'b d -> b', 'sum'))
        dotted = torch.diag(eta @ Ttheta.T)
        logAeta = torch.log(torch.special.i0(kappa))
        return dotted - logAeta - torch.log(torch.tensor(2*math.pi))


class LinearEncoder(nn.Module):
    '''
    Linearized model for lazy_training.
    f(x, \phi) = f(x, \phi_0) + Jf(x; \phi_0)(\phi-\phi_0)
    '''
    
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden_layer, model0, Jf0, phi0):
        super(LinearEncoder, self).__init__()
        self.model0 = model0
        self.Jf0 = Jf0.detach()
        self.phi0 = phi0.detach()
        self.n_hidden_layer = n_hidden_layer
        self.dense = DenseEncoder1Layer(in_dim, out_dim, hidden_dim, n_hidden_layer) if n_hidden_layer == 1 else None
        all_params = list(self.dense.parameters())
        flattened = torch.cat([x.flatten() for x in all_params]).clone().detach()
        self.param_vec_flatten = nn.Parameter(flattened, requires_grad=True)

    def get_init_Jacobian(self, x):
        '''
        Compute Jf(x; \phi_0) for intial weights \phi_0,
        but variable input x. Done by calling appropriate Jacobian function
        with self.model0 whose weights are frozen.
        '''
        return MyGetJacobian(self.model0, x).detach()

    def forward(self, x):
        '''
        Forward is significantly different in the linear model.
        We access a flattened parameter vector, and compute the expression
        in the doctstring above.
        '''
        out1 = self.model0(x).detach()
        this_x_jac = self.get_init_Jacobian(x)
        phi_minus_phi0 = self.param_vec_flatten-self.phi0
        phi_minus_phi0 = rearrange(phi_minus_phi0, 'n_param -> n_param 1')
        out2 = rearrange(this_x_jac @ phi_minus_phi0, 'b o 1 -> b o')
        return out1 + out2

    def get_log_prob(self, theta, x):
        '''
        Construct natural parameterization.
        '''
        eta = self.forward(x)+1e-4
        Ttheta = rearrange([torch.cos(theta), torch.sin(theta)], 'dim b -> b dim')
        kappa = torch.sqrt(reduce(torch.square(eta), 'b d -> b', 'sum'))
        dotted = torch.diag(eta @ Ttheta.T)
        logAeta = torch.log(torch.special.i0(kappa))
        return dotted - logAeta - torch.log(torch.tensor(2*math.pi))
