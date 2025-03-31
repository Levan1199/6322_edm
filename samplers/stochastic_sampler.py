# This file implements the stochastic sampler introduced in 
# Elucidating the Design Space of Diffusion-Based
# Generative Models
# The algorithm can be found on Page 2
import torch
import numpy as np

class StochasticSampler:
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
            