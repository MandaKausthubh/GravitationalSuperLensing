import numpy as np
import torch

def PosGrid(shape, dim, cls_token=False):
    # Clarify shape as (height, width)
    h, w = shape
    gridX = np.arange(w, dtype=np.float32) * 4.0  # width (x-axis)
    gridY = np.arange(h, dtype=np.float32) * 4.0 # height (y-axis)
    # Use indexing='ij' for row-major flattening
    x, y = np.meshgrid(gridX, gridY, indexing='ij')  # shape (w, h)
    grid = np.stack([x, y], axis=0).reshape(2, 1, -1)  # (2, 1, h*w)
    
    # Pass total_dim=dim to ensure correct frequencies
    pos_embeddings = np.concatenate([
        AddingSinCosEmbeddings(grid[0], total_dim=dim),  # x-axis
        AddingSinCosEmbeddings(grid[1], total_dim=dim)   # y-axis
    ], axis=1)

    if cls_token:
        # Consider a learnable parameter instead of zeros
        cls_token = np.zeros((1, dim))
        pos_embeddings = np.concatenate([cls_token, pos_embeddings], axis=0)
    
    return torch.tensor(pos_embeddings, dtype=torch.float32)


def AddingSinCosEmbeddings(grid, total_dim):
    # Calculate frequencies based on total_dim
    half_dim = total_dim // 4  # Each axis has total_dim//2 features, half for sin/cos
    omega = np.arange(half_dim, dtype=np.float32)
    omega = 1 / (10000 ** (2 * omega / total_dim))  # Correct scaling
    
    grid = grid.reshape(-1)
    out = np.einsum('i,j->ij', grid, omega)
    out = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return out