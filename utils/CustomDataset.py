import torch
from torch.utils.data import Dataset 
import numpy as np
import os, sys




class CustomDatasetSelfSupervised(Dataset):
    def __init__(self, root, data, transform=None):
        self.data = data
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.root, self.data[idx])
        data = np.load( data_path , allow_pickle=True)
        if self.transform is not None:
            data = self.transform(data)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)




class CustomDataset(Dataset):
    def __init__(self, root, data, label, label_to_id, transforms=None, label_transform=None):
        self.data, self.label = data, label
        self.transform = transforms
        self.label_transform = label_transform
        self.root = os.path.abspath(root)
        self.label_to_id = label_to_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.root, self.label[idx], self.data[idx])
        if self.label[idx] == "axion":
            data = torch.tensor((np.load( data_path , allow_pickle=True)[0]), dtype=torch.float32)
        else:
            data = torch.tensor(np.load( data_path , allow_pickle=True), dtype=torch.float32)
        label = self.label_to_id[self.label[idx]]
        if self.transform is not None:
            data = self.transform(data)
        return data.unsqueeze(0), torch.nn.functional.one_hot(torch.tensor(label), num_classes=len(self.label_to_id)).float()






class CustomSuperResolutionDataset(Dataset):

    def __init__(self, root, data, transform_LR, transform_HR, *args, **kwargs):
        super(CustomSuperResolutionDataset, self).__init__(*args, **kwargs)
        self.data = data
        self.root = root
        self.transform_LR = transform_LR
        self.transform_HR = transform_HR


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data[idx]
        data_HR = np.load( os.path.join(self.root, "HR", file_name) , allow_pickle=True)
        data_LR = np.load( os.path.join(self.root, "LR", file_name) , allow_pickle=True)
        if self.transform_LR is not None:
            data_LR = self.transform_LR(data_LR)
        if self.transform_HR is not None:
            data_HR = self.transform_HR(data_HR)
        return torch.tensor(data_LR, dtype=torch.float32), torch.tensor(data_HR, dtype=torch.float32)



