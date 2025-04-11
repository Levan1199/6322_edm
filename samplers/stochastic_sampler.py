# This file implements the stochastic sampler introduced in 
# Elucidating the Design Space of Diffusion-Based
# Generative Models
# The algorithm can be found on Page 2
import torch
import numpy as np

class StochasticSampler:
    """In their paper, they used a scheduling function sigma(t) = t,
    and scaling function s(t) = 1.0
    For other configurations:
        VP (Variance Preserving): sigma(t) = sqrt(exp(0.5*beta_d*t^2 + beta_min*t) -1), where beta_d = 19.9, and beta_min = 0.1
            s(t) = 1/sqrt(exp(0.5*beta_d*t^2 + beta_min*t))
        VE (Variance Exploding): sigma(t) = sqrt(t)
            s(t) = 1
        iDDPM + DDIM:
            sigma(t) = t
            s(t) = 1
        edm (this):
            same as iDDPM + DDIM
    NOTE: The SDE of Song et al can be generalized as a sum of probability flow of ODE in eq.1 of the paper
     and a time varying Langevin diffusion SDE
    The algorithm for sampling is as follows:
        Given sample x_i at noise level sigma(t_i), there are two substeps
        1, Add noise to x_i based on a factor gamma_i >= 0 to increase the noise level
        where t_i^ = t_i + gamma_i*t_i . From this higher noise level, sample x^_i.
        2. Solve the ODE backwards from t^i to t_i+1 in a single step
        The output is a sample x_i+1 at noise level t_i+1
    
    In table 5.0, the authors list the sampling parameters for the stochastic sampler
    Which are listed below:
    =====================================================
    Param           CIFAR-10        ImageNet
    =====================================================
    |               VP  |  VE      Pre-trained | This
    |====================================================
    | S_churn       30     80       80            40    |
    | S_tmin        0.01    0.05    0.05          0.05  |
    | S_tmax        1       1       50            50    |
    | S_nosie       1.007   1.007   1.003         1.003 |
    =====================================================
    
        """
    def __init__(self, dims, max_N):
        self.dims = dims
        self.max_N = max_N
        self.distributions = dict()
        for i in range(1, max_N):
            self.distributions[i] = torch.distributions.Normal(torch.zeros(dims, dims), torch.ones(dims, dims)*i**2 )
    
    def sample(self):
        '''Performs stochastic sammpling at time t'''
        x_0 = torch.zeros(self.dims, self.dims)
        for i in range(1, self.max_N):
            