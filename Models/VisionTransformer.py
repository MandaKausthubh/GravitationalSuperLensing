import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange

class Embedding(nn.Module):
    
    def __init__(self, input_shape, patch_size, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        self.num_patches = (self.input_height // self.patch_size) * (self.input_width // self.patch_size)
        self.patch_dim = self.input_channels * self.patch_size * self.patch_size
        self.embedding_size = embedding_size
        self.CLSToken = nn.Parameter(torch.zeros(1, 1, self.embedding_size))
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.embedding_size)
        )

    def forward(self, x):
        x = self.projection(x)
        x = torch.cat((self.CLSToken.repeat(x.shape[0], 1, 1), x), dim=1)
        return x

class PositionalEmbedding(nn.Module):
    
    def __init__(self, num_patches, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.embedding_size))
        
    def forward(self, x):
        return x + self.positional_embedding


class VisionTransformer(nn.Module):
    
    def __init__(self, input_shape, patch_size, embeddings_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = input_shape[1]
        self.W = input_shape[2]
        self.C = input_shape[0]
        self.patch_size = patch_size
        self.Embedding = Embedding(input_shape=input_shape, patch_size=patch_size, embedding_size=embeddings_size)
        self.PositionalEmbedding = PositionalEmbedding(num_patches=self.Embedding.num_patches, embedding_size=embeddings_size)
        self.Transformer = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embeddings_size,
                nhead=8,
                dim_feedforward = 1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6,
            norm = nn.LayerNorm(embeddings_size)
        )

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PositionalEmbedding(x)
        x = self.Transformer(x)
        return x

class VisionTransformerModel(nn.Module):
    
    def __init__(self, input_shape, patch_size, embeddings_size, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.VisionTransformer = VisionTransformer(input_shape=input_shape,
                                                patch_size=patch_size,
                                                embeddings_size=embeddings_size
                                                )
        self.classifier = nn.Linear(embeddings_size, num_classes)

    def forward(self, x):
        x = self.VisionTransformer(x)
        x = self.classifier(x[:, 0])
        return x