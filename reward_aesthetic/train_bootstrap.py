import os, pdb
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import wandb
import random
import numpy as np
import argparse
import datetime

class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class BootstrappedNetwork(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(BootstrappedNetwork, self).__init__()
        self.models = nn.ModuleList([BaseNetwork(input_dim) for _ in range(num_heads)])
    
    def forward(self, inputs, head_idx=None):
        if head_idx is None: # return all heads
            return [model(inputs) for model in self.models]
        else:  # return a specific head
            assert isinstance(head_idx, int)
            return self.models[head_idx](inputs[:,head_idx,:])

def bootstrapping(dataset, n_datasets=10):
    bootstrapped_data = []
    for _ in range(n_datasets):
        # Resample the dataset with replacement
        sampled_indices = [random.randint(0, len(dataset) - 1) for _ in range(len(dataset))]
        sampled_dataset = [dataset[i] for i in sampled_indices]
        bootstrapped_data.append(sampled_dataset)
    return bootstrapped_data

class BootstrappedDataset(Dataset):
    def __init__(self, bootstrapped_data):
        self.bootstrapped_data = bootstrapped_data

    def __len__(self):
        return len(self.bootstrapped_data[0])  # Assuming all datasets are of the same size

    def __getitem__(self, idx):
        # Retrieve the corresponding item from each dataset
        batch = [dataset[idx] for dataset in self.bootstrapped_data]
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)