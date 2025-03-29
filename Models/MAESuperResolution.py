import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from Models.MaskedAutoEncoders import MaskedAutoEncoder


class MAEforSuperResolution(nn.Module):
    def __init__(self, LR_shape, MAE_ViT: MaskedAutoEncoder, HR_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LR_shape = LR_shape
        self.HR_shape = HR_shape
        self.MAE_ViT = MAE_ViT
        self.conv1 = nn.Conv2d(
            LR_shape[0],
            MAE_ViT.C,
            kernel_size=1)

        self.conv2 = nn.Conv2d(
            MAE_ViT.C,
            HR_shape[0],
            kernel_size=1)


    def reshape_tensor(self, input_tensor, direction="up", target_shape = None, mode='bilinear'):
        """
        Convert tensor to target shape while preserving batch dimension
        Args:
            input_tensor: Input tensor (B, C, H, W)
            target_shape: Tuple of (B', C', H', W')
            mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        Returns:
            Tensor of shape target_shape
        """
        assert direction in ["up", "down"], "Direction must be 'up' or 'down'"
        if target_shape is None:
            target_shape = self.HR_shape

        if input_tensor.shape[0] != target_shape[0]:
            raise ValueError(f"Batch size mismatch: {input_tensor.shape[0]} vs {target_shape[0]}")

        if input_tensor.shape[1] != target_shape[1]:
            if direction == "up":
                conv = self.conv2
            elif direction == "down":
                conv = self.conv1
            x = conv(input_tensor)
        else:
            x = input_tensor

        print(f"Reshaping tensor from {x.shape} to {target_shape}")
        if x.shape[2:] != target_shape[2:]:
            x = F.interpolate(x, size=target_shape[2:], mode=mode)

        return x



    def forward(self, x):
        """
        Forward pass of the MAE for super-resolution model.
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H', W') where H' and W' are the target height and width
        """
        x = self.reshape_tensor(
            x,
            direction="down",
            target_shape=(
                x.shape[0],
                self.MAE_ViT.C,
                self.MAE_ViT.H,
                self.MAE_ViT.W),
            mode='bilinear')
        print(f"Input tensor shape after downsampling: {x.shape}")
        _, x, _ = self.MAE_ViT(x, mask_ratio=0.0)
        x = x.reshape(x.shape[0], self.HR_shape[0], x.shape[1], x.shape[2])
        print(f"Input tensor shape after MAE: {x.shape}")
        x = self.reshape_tensor(
            x,
            direction="up",
            target_shape=(
                x.shape[0],
                self.HR_shape[0],
                self.HR_shape[1],
                self.HR_shape[2]),
            mode='bilinear')
        return x