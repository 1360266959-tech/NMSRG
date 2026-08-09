# dalib/modules/masking.py
# -*- coding: utf-8 -*-
"""
Simple complementary binary masking with optional image augmentations.

Usage:
    masking = Masking(block_size=32, ratio=0.5,
                      color_jitter_s=0.2, color_jitter_p=0.2,
                      blur=True, mean=(0.485,0.456,0.406),
                      std=(0.229,0.224,0.225))

    # 生成互补二值遮罩（不回传梯度）
    mask1, mask2 = masking.generate_binary_mask(x)  # shapes: (B,1,H,W)

    # 可选的强增强（按 aug 字符串开关：'j' 'b' 'e' 'g'）
    x_aug = masking(x, aug='jbe')
"""

import math
import random
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kornia
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False
    warnings.warn("[Masking] kornia not found; augmentations will be disabled.", RuntimeWarning)


__all__ = ["Masking"]


# ---------- helpers ----------
def _denorm(img: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean = torch.as_tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, device=img.device).view(1, -1, 1, 1)
    return img.mul(std).add(mean)


def _renorm(img: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean = torch.as_tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, device=img.device).view(1, -1, 1, 1)
    return img.sub(mean).div(std)


def _color_jitter(prob_scalar: float, mean, std, data: torch.Tensor, s: float = 0.25, p: float = 0.2):
    if not _HAS_KORNIA:
        return data
    if prob_scalar > p:
        seq = nn.Sequential(kornia.augmentation.ColorJitter(
            brightness=s, contrast=s, saturation=s, hue=s
        ))
        data = _denorm(data, mean, std)
        data = seq(data)
        data = _renorm(data, mean, std)
    return data


def _gaussian_blur(prob_scalar: float, data: torch.Tensor):
    if not _HAS_KORNIA:
        return data
    if prob_scalar > 0.5:
        sigma = float(np.random.uniform(0.15, 1.15))
        h, w = int(data.shape[2]), int(data.shape[3])
        # 至少 3x3，且奇数核
        kh = max(3, (int(0.1 * h) | 1))
        kw = max(3, (int(0.1 * w) | 1))
        seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=(kh, kw), sigma=(sigma, sigma)))
        data = seq(data)
    return data


def _grey_scale(prob_scalar: float, data: torch.Tensor):
    if not _HAS_KORNIA:
        return data
    if prob_scalar > 0.5:
        seq = nn.Sequential(kornia.augmentation.RandomGrayscale(p=1.0))
        data = seq(data)
    return data


def _random_erasing(prob_scalar: float, data: torch.Tensor):
    if not _HAS_KORNIA:
        return data
    if prob_scalar > 0.5:
        seq = nn.Sequential(
            kornia.augmentation.RandomErasing(scale=(.3, .5), ratio=(.3, 1/.3), p=0.5)
        )
        data = seq(data)
    return data


def _strong_transform(param: dict, data: torch.Tensor, aug: str):
    # 'j' -> jitter, 'b' -> blur, 'e' -> erasing, 'g' -> grayscale
    if 'j' in aug:
        data = _color_jitter(
            prob_scalar=param['color_jitter'],
            s=param['color_jitter_s'],
            p=param['color_jitter_p'],
            mean=param['mean'],
            std=param['std'],
            data=data
        )
    if 'b' in aug:
        data = _gaussian_blur(param['blur'], data)
    if 'e' in aug:
        data = _random_erasing(param.get('erasing', 0.0), data)
    if 'g' in aug:
        data = _grey_scale(param.get('gray', 0.0), data)
    return data


# ---------- main class ----------
class Masking(nn.Module):
    """
    Complementary binary masking on image grids:
        - sample Bernoulli D in patch grid (ph x pw)
        - view-1 mask = D_up; view-2 mask = 1 - D_up  (nearest upsample)

    Args:
        block_size: patch边长（像素），建议 16/32/64。
        ratio: 遮挡比例 r ∈ [0,1]（view-1保留概率=1-r，view-2相反）。
        color_jitter_s/p: 颜色抖动强度与触发概率。
        blur: 是否启用模糊（内部会随机触发）。
        mean/std: 与训练归一化一致。
    """
    def __init__(
        self,
        block_size: int = 64,
        ratio: float = 0.6,
        color_jitter_s: float = 0.2,
        color_jitter_p: float = 0.2,
        blur: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.block = int(block_size)
        self.ratio = float(ratio)
        assert self.block >= 1, "mask_block_size must be >= 1"
        assert 0.0 <= self.ratio <= 1.0, "mask_ratio must be in [0,1]"
        self.mean, self.std = mean, std

        self.augmentation_params = {
            'color_jitter_s': float(color_jitter_s),
            'color_jitter_p': float(color_jitter_p),
            'blur': float(blur),
        }
        self.use_aug = _HAS_KORNIA and ((color_jitter_s > 0 and color_jitter_p > 0) or bool(blur))

    @torch.no_grad()
    def generate_binary_mask(self, imgs: torch.Tensor):
        """
        Return:
            mask1, mask2: (B,1,H,W) in {0,1}, complementary
        """
        if imgs.ndim != 4:
            raise ValueError(f"Expect imgs as (B,C,H,W), got shape {tuple(imgs.shape)}")
        B, _, H, W = imgs.shape

        # 用向上取整的网格，边界覆盖更均匀
        ph = max(1, math.ceil(H / self.block))
        pw = max(1, math.ceil(W / self.block))

        keep_prob = float(max(0.0, min(1.0, 1.0 - self.ratio)))  # 1=保留, 0=遮挡
        D = torch.full((B, 1, ph, pw), keep_prob, device=imgs.device)
        D = torch.bernoulli(D)  # {0,1}

        # 上采样到 HxW（nearest 保持二值）
        D_up = F.interpolate(D, size=(H, W), mode='nearest')
        mask1 = D_up.to(imgs.dtype).to(imgs.device).contiguous().detach()
        mask2 = (1.0 - D_up).to(imgs.dtype).to(imgs.device).contiguous().detach()
        mask1.clamp_(0.0, 1.0)
        mask2.clamp_(0.0, 1.0)
        return mask1, mask2

    @torch.no_grad()
    def forward(self, img: torch.Tensor, aug: str = "") -> torch.Tensor:
        """
        可选增强：根据 aug 字符串决定使用哪些增强（'j','b','e','g'）。
        仅做数据增强，不生成遮罩。
        """
        out = img.clone()
        if self.use_aug and len(aug) > 0:
            param = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.augmentation_params['color_jitter_s'],
                'color_jitter_p': self.augmentation_params['color_jitter_p'],
                'blur': random.uniform(0, 1) if self.augmentation_params['blur'] else 0.0,
                'erasing': random.uniform(0, 1),
                'gray': random.uniform(0, 1),
                'mean': self.mean,
                'std': self.std
            }
            out = _strong_transform(param, data=out, aug=aug)
        return out
