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
NAME = 'TCD'
ALIAS = 'tcd'



def default_noise_sampler(x):
    return lambda sigma, sigma_next: k_diffusion.sampling.torch.randn_like(x)

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    # TCD sampling using modified DDPM.
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        sigma_from, sigma_to = sigmas[i], sigmas[i+1]

        # TCD offset, based on gamma, and conversion between sigma and timestep
        t = model.inner_model.sigma_to_t(sigma_from)
        t_s = (1 - gamma) * t
        sigma_to_s = model.inner_model.t_to_sigma(t_s)

        # if sigma_to_s > sigma_to:
        #     sigma_to_s = sigma_to
        # if sigma_to_s < 0:
        #     sigma_to_s = torch.tensor(1.0)
        #print(f"sigma_from: {sigma_from}, sigma_to: {sigma_to}, sigma_to_s: {sigma_to_s}")


        # The following is equivalent to the comfy DDPM implementation
        # x = DDPMSampler_step(x / torch.sqrt(1.0 + sigma_from ** 2.0), sigma_from, sigma_to, (x - denoised) / sigma_from, noise_sampler)

        noise_est = (x - denoised) / sigma_from
        x /= torch.sqrt(1.0 + sigma_from ** 2.0)

        alpha_cumprod = 1 / ((sigma_from * sigma_from) + 1)  # _t
        alpha_cumprod_prev = 1 / ((sigma_to * sigma_to) + 1) # _t_prev
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        ## These values should approach 1.0?
        # print(f"alpha_cumprod: {alpha_cumprod}")
        # print(f"alpha_cumprod_prev: {alpha_cumprod_prev}")
        # print(f"alpha: {alpha}")


        # alpha_cumprod_down = 1 / ((sigma_to_s * sigma_to_s) + 1)  # _s
        # alpha_d = (alpha_cumprod_prev / alpha_cumprod_down)
        # alpha2 = (alpha_cumprod / alpha_cumprod_down)
        # print(f"** alpha_cumprod_down: {alpha_cumprod_down}")
        # print(f"** alpha_d: {alpha_d}, alpha2: #{alpha2}")

        # epsilon noise prediction from comfy DDPM implementation
        x = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())
        # x = (1.0 / alpha_d).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())

        first_step = sigma_to == 0
        last_step = i == len(sigmas) - 2

        if not first_step:
            if gamma > 0 and not last_step:
                noise = noise_sampler(sigma_from, sigma_to)

                # x += ((1 - alpha_d) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise
                variance = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * (1 - alpha_cumprod / alpha_cumprod_prev)
                x += variance.sqrt() * noise # scale noise by std deviation

                # relevant diffusers code from scheduling_tcd.py
                # prev_sample = (alpha_prod_t_prev / alpha_prod_s).sqrt() * pred_noised_sample + (
                #     1 - alpha_prod_t_prev / alpha_prod_s
                # ).sqrt() * noise

            x *= torch.sqrt(1.0 + sigma_to ** 2.0)

        # beta_cumprod_t = 1 - alpha_cumprod
        # beta_cumprod_s = 1 - alpha_cumprod_down

 
    return x



# add sampler
if not NAME in [x.name for x in sd_samplers.all_samplers]:
    euler_smea_samplers = [(NAME, sample_tcd, [ALIAS], {})]
    samplers_data_euler_smea_samplers = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in euler_smea_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += samplers_data_euler_smea_samplers
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
