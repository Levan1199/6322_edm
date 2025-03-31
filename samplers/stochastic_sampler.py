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
        VP: sigma(t) = sqrt(exp(0.5*beta_d*t^2 + beta_min*t) -1), where beta_d = 19.9, and beta_min = 0.1
            s(t) = 1/sqrt(exp(0.5*beta_d*t^2 + beta_min*t))
        VE: sigma(t) = sqrt(t)
            s(t) = 1
        iDDPM + DDIM:
            sigma(t) = t
            s(t) = 1
        edm (this):
            same as iDDPM + DDIM"""
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
            