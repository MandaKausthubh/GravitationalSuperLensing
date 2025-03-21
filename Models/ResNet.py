from sklearn.metrics import roc_auc_score
import torch.nn as nn
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.projection = None
        if in_channels != out_channels or stride != 1:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )

        
    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = self.relu1(x)
        x = self.bn2(self.conv2(x))
        if self.projection is not None:
            if(x.shape != self.projection(input).shape):
                print(x.shape, input.shape, self.projection(input).shape)
            x = self.relu2(x + self.projection(input))
        return x

class ResNet18(nn.Module):
    
    def __init__(self, input_size = 150, in_channels = 1, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.BN1 = nn.BatchNorm2d(64)
        self.MaxPool = nn.MaxPool2d(3, 2, 1)
        self.ConvBlock1 = ConvBlock(64, 64, 3, 1, 1)
        self.ConvBlock2 = ConvBlock(64, 128, 3, 2, 1)
        self.ConvBlock3 = ConvBlock(128, 128, 3, 1, 1)
        self.ConvBlock4 = ConvBlock(128, 256, 3, 2, 1)
        self.ConvBlock5 = ConvBlock(256, 256, 3, 1, 1)
        self.ConvBlock6 = ConvBlock(256, 512, 3, 2, 1)
        self.ConvBlock7 = ConvBlock(512, 512, 3, 1, 1)
        self.ConvBlock8 = ConvBlock(512, 1024, 3, 2, 1)
        self.AdaPool = nn.AdaptiveAvgPool2d(1)
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(1024, 32)
        self.GeLU = nn.GELU()
        self.Linear1 = nn.Linear(32, num_classes)
        self.Sigmoid = nn.Sigmoid()

        # self.initialize_weights()

            
    def initialize_weights(self):
        """ 
            This applies the Xavier initialization to every layer in the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, input):
        x = self.BN1(self.Conv1(input))
        x = self.MaxPool(x)
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.ConvBlock8(x)
        x = self.AdaPool(x)
        x = self.Flatten(x)
        x = self.Linear(x)
        x = self.GeLU(x)
        x = self.Linear1(x)
        x = self.Sigmoid(x)
        return x



class MiniResNet(nn.Module):
    
    def __init__(self, input_size = 150, in_channels = 1, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.BN1 = nn.BatchNorm2d(64)
        self.MaxPool = nn.MaxPool2d(3, 2, 1)
        self.ConvBlock1 = ConvBlock(64, 64, 3, 1, 1)
        self.ConvBlock2 = ConvBlock(64, 128, 3, 2, 1)
        self.ConvBlock3 = ConvBlock(128, 128, 3, 1, 1)
        self.ConvBlock4 = ConvBlock(128, 256, 3, 2, 1)
        self.AdaPool = nn.AdaptiveAvgPool2d(1)
        self.Flatten = nn.Flatten()
        self.Linear = nn.Linear(256, 32)
        self.GeLU = nn.GELU()
        self.Linear1 = nn.Linear(32, num_classes)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.BN1(self.Conv1(input))
        x = self.MaxPool(x)
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.AdaPool(x)
        x = self.Flatten(x)
        x = self.Linear(x)
        x = self.GeLU(x)
        x = self.Linear1(x)
        x = self.Sigmoid(x)
        return x
