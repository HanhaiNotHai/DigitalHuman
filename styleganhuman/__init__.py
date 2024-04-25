from typing import Literal

import numpy as np
import torch

import dnnlib

from . import legacy


class StyleGANHuman:

    def __init__(self) -> None:
        with dnnlib.util.open_url(
            'styleganhuman/pretrained_models/stylegan_human_v3_512.pkl'
        ) as f:
            self.model: torch.nn.Module = legacy.load_network_pkl(f)['G_ema']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to(self, device: Literal['cpu', 'cuda']) -> None:
        if device == 'cpu':
            self.model.cpu()
            torch.cuda.empty_cache()
        else:
            self.model.to(self.device)

    def generate_z(sef, z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
        return (
            torch.from_numpy(np.random.RandomState(seed).randn(1, z_dim))
            .to(device)
            .float()
        )

    @torch.inference_mode()
    def generate_image(self, seed: int, truncation_psi: float) -> np.ndarray:
        self.to('cuda')

        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

        z = self.generate_z(self.model.z_dim, seed, self.device)
        label = torch.zeros([1, self.model.c_dim], device=self.device)

        out: torch.Tensor = self.model(
            z, label, truncation_psi=truncation_psi, force_fp32=True
        )
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        self.to('cpu')
        return out[0].cpu().numpy()
