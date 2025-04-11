# This file implements the stochastic sampler introduced in 
# Elucidating the Design Space of Diffusion-Based
# Generative Models
# The algorithm can be found on Page 2
import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal

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
    | S_noise       1.007   1.007   1.003         1.003 |
    =====================================================
    
        """
    # A global config to get the hyperparams
    hyperparam_config_ = {
        "CIFAR-10_VP": dict(S_churn=30, S_tmin=0.01, S_tmax=1.0, S_noise=1.007),
        "CIFAR-10_VE": dict(S_churn=80, S_tmin=0.05, S_tmax=1.0, S_noise=1.007),
        "ImageNet-pre": dict(S_churn=80, S_tmin=0.05, S_tmax=50.0, S_noise=1.003),
        "ImageNet-this": dict(S_churn=40, S_tmin=0.05, S_tmax=50.0, S_noise=1.003),
    }
    schedule_config_ = {
        "VE": dict(sigma_t=lambda t: t**0.5, s_t=lambda t: 1),
        "edm": dict(sigma_t=lambda t: t, s_t=lambda t: 1)
    }
    def __init__(self, dims, max_N, config, schedule_config):
        """
        Parameters:
        ----------
        max_N: The maximum number of ODE steps, this is required to compute scheduling params
        config: The config to load, options are: see hyperparam_config attr
        schedule_config: config for schedule params, see schedule_config attr
        
        
        Parameters_misc:
        --------------
        sigma(t) = t
        and s(t) = 1"""
        if config not in self.hyperparam_config_:
            raise RuntimeError(f"config provided {config} is not valid for this sampler, please choose from {list(self.hyperparam_config_.keys())}")
        if schedule_config not in self.schedule_config_:
            raise RuntimeError(f"schedule config provided {schedule_config} is not valid for this sampler, please choose from {list(self.schedule_config_.keys())}")
        self.__dict__.update(self.hyperparam_config_.get(config))
        self.__dict__.update(self.schedule_config_[schedule_config])
        # self.dims = dims
        self.sample_size = torch.prod(torch.Tensor(dims))
        self.max_N = max_N
        self.set_normal_distribution()
        
        
    def set_normal_distribution(self):
        """Sets the normal distribution, based on S_noise param. 
        This distribution is an indepenedent (diagonal covariance matrix) distribution, from which we 
        sample epsilon_t"""
        self.distribution = Normal(loc=torch.zeros((self.sample_size)), scale=torch.ones((self.sample_size))*self.S_noise**2)
    
    def sample_eps(self):
        """Returns a sample from the distribution"""
        return self.distribution.sample()
    
    def get_scheduled_variance(self, t_i):
        """ returns sigma(t_i)"""
        return self.sigma_t(t_i)
    
    def get_scheduled_gamma(self, t_i):
        """Returns the scheduled gamma according to min(S_churn/N, sqrt(2) -1) if t_i E (S_tmin, S_tmax)
        for the edm case, t_i = sigma_t, however, here we implement a generalized case
        else 0.
        t_i  E {0, ..., N}"""
        if ( t_i <= self.S_tmax and t_i >= self.S_tmin):
            return min( self.S_churn/self.max_N, 2**0.5 - 1)
        return 0.0

    def step_forward(self, x_t, t_i):
        """steps forward according to algorithm 2
        The t_i is the time, to obtain the values sigma(t_i), and s(t_i)
        x_t is the current sample (from last step, or initial value (x _t at t=0))"""   
        sigma_t = self.sigma_t(t_i)
        s_t = self.s_t(t_i)
        # line 5 of algo
        sigma_t_hat = sigma_t + self.get_scheduled_gamma(t_i) 
        # line 6 of algo
        x_t = x_t + self.sample_eps()*(sigma_t_hat**2 - sigma_t**2)**0.5