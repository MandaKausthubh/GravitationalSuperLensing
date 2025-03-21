import torch
import torch.nn as nn

from MaskedAutoEncoders import MaskedAutoEncoder

class MAEClassifier(nn.Module):
    
    def __init__(self, Encoder:MaskedAutoEncoder, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Encoder = Encoder
        self.num_classes = Encoder.num_classes
        self.LinearLayer1 = nn.Linear(Encoder.encoder_embed_dim, hidden_size)
        self.GeLu = nn.GELU()
        self.LinearLayer2 = nn.Linear(hidden_size, self.num_classes)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        self.Encoder.eval()
        with torch.no_grad():
            embed = self.Encoder.forward_encoder(x, masking_ratio=0, mode="classification")
            CLS = embed[:, 0, :]
            y = self.LinearLayer1(CLS)
            y = self.GeLu(y)
            y = self.LinearLayer2(y)
            y = self.Sigmoid(y)
        return y