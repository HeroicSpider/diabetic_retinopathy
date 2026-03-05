import torch
import torch.nn as nn
import numpy as np
import math

class CosineNoiseScheduler(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 1. Cosine schedule mathematics
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        # 2. Register buffer so it moves to GPU automatically
        self.register_buffer("alphas_cumprod", alphas_cumprod[1:]) 

    def add_noise(self, original_samples, noise, timesteps):
        """Adds noise for the forward diffusion training process."""
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)[timesteps]
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1, 1)
        
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def get_ddim_timesteps(self, num_inference_steps):
        """Calculates the specific timesteps to jump to for fast DDIM sampling."""
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        return timesteps

    def ddim_reverse_step(self, sample, noise_pred, t, t_prev):
        """
        The DDIM reverse step with PROPER overflow clamping and noise re-derivation.
        """
        device = sample.device
        
        alpha_prod_t = torch.clamp(self.alphas_cumprod[t], min=1e-8)
        alpha_prod_t_prev = torch.clamp(self.alphas_cumprod[t_prev], min=1e-8) if t_prev >= 0 else torch.tensor(1.0, device=device)
        beta_prod_t = 1 - alpha_prod_t
        
        # 1. Predict the original clean image (x0)
        pred_original_sample = (sample - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        
        # 2. Clamp x0 to prevent neon color explosions
        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)
        
        # 3. CRITICAL FIX: Re-derive the implied noise based on the CLAMPED image!
        # This ensures the noise perfectly aligns with the clamp, preventing static accumulation.
        noise_implied = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        
        # 4. Point towards the next cleaner step (t_prev) using the IMPLIED noise
        dir_xt = (1 - alpha_prod_t_prev) ** 0.5 * noise_implied
        
        # 5. Combine
        x_prev = alpha_prod_t_prev ** 0.5 * pred_original_sample + dir_xt
        return x_prev
