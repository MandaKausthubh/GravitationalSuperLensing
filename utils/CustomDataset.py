import torch
from torch.utils.data import Dataset 

class CustomDataset(Dataset):

    def __init__(self, data:torch.Tensor, labels:torch.Tensor, transforms=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        label = torch.nn.functional.one_hot(self.labels[idx].long(), num_classes=3).float()
        return data, label

        
        
class CustomDatasetSelfSupervised(Dataset):

    def __init__(self, data:torch.Tensor, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data