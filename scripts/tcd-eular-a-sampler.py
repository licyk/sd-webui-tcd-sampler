import torch
import tqdm
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from tqdm.auto import trange, tqdm
from k_diffusion import utils
from k_diffusion.sampling import to_d
import math
from importlib import import_module

sampling = import_module("k_diffusion.sampling")
NAME = 'TCD_Eular_A'
ALIAS = 'tcd_eular_a'



def default_noise_sampler(x):
    return lambda sigma, sigma_next: k_diffusion.sampling.torch.randn_like(x)

def sample_tcd_euler_a(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    # TCD sampling using modified Euler Ancestral sampler. by @laksjdjf
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        #d = to_d(x, sigmas[i], denoised)		
        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]

        t = model.inner_model.sigma_to_t(sigma_from)
        down_t = (1 - gamma) * t
        sigma_down = model.inner_model.t_to_sigma(down_t)

        if sigma_down > sigma_to:
            sigma_down = sigma_to
        sigma_up = (sigma_to ** 2 - sigma_down ** 2) ** 0.5     

        # same as euler ancestral
        d = to_d(x, sigma_from, denoised)
        dt = sigma_down - sigma_from
        x += d * dt

        if sigma_to > 0 and gamma > 0:
            x =  x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigma_up
    return x


# add sampler
if not NAME in [x.name for x in sd_samplers.all_samplers]:
    euler_smea_samplers = [(NAME, sample_tcd_euler_a, [ALIAS], {})]
    samplers_data_euler_smea_samplers = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in euler_smea_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += samplers_data_euler_smea_samplers
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
