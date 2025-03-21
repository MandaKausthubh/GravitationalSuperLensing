import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from Models.ResNet import ConvBlock
from Models.VisionTransformer import VisionTransformerModel


class ViTResNet(nn.Module):
    def __init__(self, input_shape, patch_size, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ConvBlock = ConvBlock(input_shape[0], 32, 3, 2, 1)
        self.VisionTransformer = VisionTransformerModel(input_shape=(32, input_shape[1]//2, input_shape[2]//2),
                                                        patch_size=patch_size,
                                                        embeddings_size=embedding_size,
                                                        num_classes=3
                                                    )
        self.Linear1 = nn.Linear(3, 3)
        self.Gelu = nn.GELU()
        self.Linear2 = nn.Linear(3, 3)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.ConvBlock(input)
        x = self.VisionTransformer(x)
        x = self.Gelu(self.Linear1(x))
        x = self.Linear2(x)
        x = self.Sigmoid(x)
        return x