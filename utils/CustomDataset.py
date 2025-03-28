import torch
from torch.utils.data import Dataset 
import numpy as np
        
class CustomDatasetSelfSupervised(Dataset):
    def __init__(self, root, data, transform=None):
        self.data = data
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_path = self.root + self.data[idx]
        data = np.load( data_path , allow_pickle=True)
        if self.transform is not None:
            data = self.transform(data)
        return data



    
def CustomDataset(Dataset):
    
    def __init__(self, root, data, label, label_to_id, transforms=None, label_transform=None):
        self.data, self.label = data, label
        self.transform = transform
        self.label_transform = label_transform
        self.root = root
        self.label_to_id = label_to_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_path = self.root + self.data[idx]  
        label = self.label[idx]
        data = np.load( data_path , allow_pickle=True)
        if label == "":
            pass
        if self.transform is not None:
            data = self.transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return data, self.label_to_id[label]







        