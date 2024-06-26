import os, pdb
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
from tqdm import tqdm

import sys
import os
from accelerate.utils import broadcast
import copy
cwd = os.getcwd()
sys.path.append(cwd)

import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np
import argparse
from aesthetic_scorer import MLPDiff
import datetime
from reward_aesthetic.train_bootstrap import BootstrappedNetwork
import ml_collections

from importlib import resources
ASSETS_PATH = resources.files("assets")


class D_explored(torch.nn.Module):
    def __init__(self, config=None, device=None):
        super().__init__()
        
        self.device = device
        self.x = torch.empty(0, device=self.device)
        self.y = torch.empty(0, device=self.device)
        
        self.noise = 0.1
        
        if config.train.optimism in ['none', 'UCB']:
            self.model = MLPDiff()
        elif config.train.optimism == 'bootstrap':
            self.model = BootstrappedNetwork(input_dim=768, num_heads=4)
        else:
            raise NotImplementedError
        
        self.labeler = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.labeler.load_state_dict(state_dict)
        self.labeler.eval()
        self.labeler.requires_grad_(False)
    
    def update(self, new_x, new_y=None):
        if self.x.numel() == 0:
            self.x = new_x
        else:
            self.x = torch.cat((self.x, new_x), dim=0)
        
        if new_y is None:
            with torch.no_grad():
                self.labeler.to(new_x.device)
                y_real = self.labeler(new_x) 
                new_y = y_real + torch.randn_like(y_real) * self.noise
                
        if self.y.numel() == 0:
            self.y = new_y
        else:
            self.y = torch.cat((self.y, new_y), dim=0)

    def cov(self, config=None): # used if we have non-optimism or UCB optimism
        with torch.no_grad():
            if config is not None and config.train.num_gpus > 1:
                features = self.model.module.forward_up_to_second_last(self.x)
            else:  
                features = self.model.forward_up_to_second_last(self.x)
        return torch.cov(features.t())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    def train_MLP(self, accelerator, config):
        
        assert self.x.requires_grad == False
        assert self.y.requires_grad == False
        
        args = ml_collections.ConfigDict()

        # Arguments
        args.num_epochs = 300
        args.train_bs = 512
        args.val_bs = 512
        args.lr = 0.001
        
        if 'SGLD' in config.train.keys():
            args.SGLD_base_noise = config.train.SGLD
            assert config.train.optimism == 'none', "SGLD only works with non-optimism"
        else:
            args.SGLD_base_noise = 0
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        self.model.requires_grad_(True)
        self.model.train()
        
        val_percentage = 0.05 # 5% of the trainingdata will be used for validation
        train_border = int(self.x.shape[0] * (1 - val_percentage) )
        
        train_dataset = TensorDataset(self.x[:train_border],self.y[:train_border])
        train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True) # create your dataloader

        val_dataset = TensorDataset(self.x[train_border:],self.y[train_border:])
        val_loader = DataLoader(val_dataset, batch_size=args.val_bs) # create your dataloader
        
        best_loss = 999
        best_model = {k: torch.empty_like(v) for k, v in self.model.state_dict().items()}
            
        def adjust_noise(learning_rate, batch_size):
            return args.SGLD_base_noise * (learning_rate ** 0.5) / (batch_size ** 0.5)   
    
        with torch.enable_grad():
            for epoch in range(args.num_epochs):
                
                noise_level = adjust_noise(args.lr, args.train_bs)
                
                losses = []
                for batch_num, (x,y) in enumerate(train_loader):
                    optimizer.zero_grad()

                    output = self.model(x)
                    
                    loss = criterion(output, y.detach())
                    accelerator.backward(loss)
                    losses.append(loss.item())
                    
                    # add Gaussian noise to gradients
                    if config.train.num_gpus > 1:
                        for param in self.model.module.parameters():
                            if param.grad is not None:
                                param.grad += noise_level * torch.randn_like(param.grad)
                    else:
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad += noise_level * torch.randn_like(param.grad)
                    
                    # for param in self.model.parameters():
                    #     if param.grad is not None:
                    #         param.grad += noise_level * torch.randn_like(param.grad)


                    optimizer.step()
                
                if accelerator.is_main_process:
                    losses_val = []
                    
                    for _, (x,y) in enumerate(val_loader):
                        self.model.eval()
                        optimizer.zero_grad()
                        output = self.model(x)
                        loss = criterion2(output, y.detach())

                        losses_val.append(loss.item())

                    print('Epoch %d | Loss %6.4f | val-loss %6.4f' % (epoch, (sum(losses)/len(losses)), sum(losses_val)/len(losses_val)))

                    if sum(losses_val)/len(losses_val) < best_loss:
                        best_loss = sum(losses_val)/len(losses_val)
                        print("Best MAE val loss so far: %6.4f" % (best_loss))
                        best_model = self.model.state_dict()
        
        best_model = broadcast(best_model)
        self.model.load_state_dict(best_model)
             
        self.model.requires_grad_(False)
        self.model.eval()
            
        del optimizer, criterion, criterion2, train_dataset, train_loader, val_dataset, val_loader
  
    def train_bootstrap(self,accelerator,config):
        from reward_aesthetic.train_bootstrap import bootstrapping, BootstrappedDataset
        
        assert self.x.requires_grad == False
        assert self.y.requires_grad == False
        
        args = ml_collections.ConfigDict()

        # # Add arguments
        args.num_epochs = 300
        args.train_bs = 512
        args.val_bs = 512
        args.lr = 0.001
        args.num_heads = 4

        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        self.model.requires_grad_(True)
        self.model.train()
        
        val_percentage = 0.05 # 5% of the trainingdata will be used for validation
        train_border = int(self.x.shape[0] * (1 - val_percentage))

        train_dataset = TensorDataset(self.x[:train_border],self.y[:train_border])
        bootstrapped_traindata = bootstrapping(train_dataset, n_datasets=args.num_heads)
        bootstrapped_trainset = BootstrappedDataset(bootstrapped_traindata)
        train_loader = DataLoader(bootstrapped_trainset, batch_size=args.train_bs, shuffle=True)  
        
        val_dataset = TensorDataset(self.x[train_border:],self.y[train_border:])
        bootstrapped_valdata = bootstrapping(val_dataset, n_datasets=args.num_heads)
        bootstrapped_valset = BootstrappedDataset(bootstrapped_valdata)
        val_loader = DataLoader(bootstrapped_valset, batch_size=args.val_bs,shuffle=False)
        
        best_loss = 999
        best_model = {k: torch.empty_like(v) for k, v in self.model.state_dict().items()}
        
        with torch.enable_grad():
            for epoch in range(args.num_epochs):
                
                losses = []
                for _, (inputs,targets) in enumerate(train_loader):
                    
                    optimizer.zero_grad()
                    loss = 0
                    for i in range(args.num_heads):
                        output = self.model(inputs, head_idx=i) #inputs shape: [128,10,768] and output shape: [128,10,1]
                        loss += criterion(output, targets[:,i,:].detach())          
                    
                    loss /= args.num_heads
                    accelerator.backward(loss)
                    losses.append(loss.item())
                    
                    optimizer.step()
                
                if accelerator.is_main_process:
                    losses_val = []    
                    for _, (inputs,targets) in enumerate(val_loader):
                        self.model.eval()
                        optimizer.zero_grad()
                        
                        loss = 0
                        for i in range(args.num_heads):
                            output = self.model(inputs, head_idx=i) #inputs shape: [128,10,768] and output shape: [128,10,1]
                            loss += criterion2(output, targets[:,i,:].detach()) 
                        loss /= args.num_heads
                        losses_val.append(loss.item())
                                       

                    print('Epoch %d | Loss %6.4f | val-loss %6.4f' % (epoch, (sum(losses)/len(losses)), sum(losses_val)/len(losses_val)))
                    
                    if sum(losses_val)/len(losses_val) < best_loss:
                        best_loss = sum(losses_val)/len(losses_val)
                        print("Best MAE val loss so far: %6.4f" % (best_loss))
                        best_model = self.model.state_dict()

        best_model = broadcast(best_model)
        self.model.load_state_dict(best_model)
             
        self.model.requires_grad_(False)
        self.model.eval()
            
        del optimizer, criterion, criterion2, train_dataset, train_loader, bootstrapped_traindata, bootstrapped_trainset,
        val_dataset, val_loader, bootstrapped_valdata, bootstrapped_valset 

if __name__ == "__main__":
    from accelerate import Accelerator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = ml_collections.ConfigDict()
    config.train = train = ml_collections.ConfigDict()
    config.train.optimism = 'bootstrap'
    
    dataset = D_explored(config).to(device, dtype=torch.float32)
    
    new_data_x = torch.from_numpy(np.load("./reward_aesthetic/data/ava_x_openclip_l14.npy"))[:200000,:].to(device)
    dataset.update(new_data_x)
    assert len(dataset.x) == len(dataset.y)
    
    accelerator = Accelerator()
    dataset.model = accelerator.prepare(dataset.model)
    
    for name, param in dataset.model.named_parameters():
        print(name)
    
    # print(dataset.cov().shape)
    # print(dataset.cov().min())
    # print(dataset.cov().max())
    # dataset.train_MLP(accelerator,config)
    
    # dataset.train_bootstrap(accelerator,config)
        
