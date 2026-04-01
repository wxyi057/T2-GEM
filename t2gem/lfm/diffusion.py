from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _cosine_log_snr(t, eps: float = 1e-5):
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def _shifted_cosine_log_snr(t, im_size: int, ref_size: int = 64):
    return _cosine_log_snr(t) + 2 * np.log(ref_size / im_size)


def _shifted_cosine_alpha_bar(t, im_size: int, ref_size: int = 64):
    return _sigmoid(_shifted_cosine_log_snr(t, im_size, ref_size))


class ForwardDiffusion(nn.Module):
    def __init__(self, im_size: int = 64, n_diffusion_timesteps: int = 1000) -> None:
        super().__init__()
        self.n_diffusion_timesteps = int(n_diffusion_timesteps)
        cos_alpha_bar_t = _shifted_cosine_alpha_bar(
            np.linspace(0, 1, self.n_diffusion_timesteps),
            im_size=im_size,
        ).astype(np.float32)
        self.register_buffer("alpha_bar_t", torch.from_numpy(cos_alpha_bar_t))

    def q_sample(self, x_start: torch.Tensor, t: int, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        if t < 0 or t >= self.alpha_bar_t.shape[0]:
            raise ValueError(f"Invalid diffusion step t={t}; expected in [0, {self.alpha_bar_t.shape[0]-1}]")

        alpha_bar_t = self.alpha_bar_t[t]
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise
