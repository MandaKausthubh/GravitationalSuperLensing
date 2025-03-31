import numpy as np 
import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from Models.MaskedAutoEncoders import MaskedAutoEncoder

class MAEViTClassifier(nn.Module):
    
    def __init__(self, Encoder : MaskedAutoEncoder, num_classes : int, hidden_size=512):
        super(MAEViTClassifier, self).__init__()
        self.Encoder = Encoder
        self.classifier = nn.Sequential(
            nn.Linear(Encoder.encoder_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # print(x.shape)
        x, _, _ = self.Encoder.forward_encoder(x, masking_ratio=0.0, mode="classification")
        # print(x.shape)
        x = self.classifier(x[:, :1, :])
        x = x.squeeze(1)
        # print(x.shape)
        return self.Sigmoid(x)

