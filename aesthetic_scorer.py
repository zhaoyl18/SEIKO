# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import CLIPModel
from PIL import Image
from torch.utils.checkpoint import checkpoint
ASSETS_PATH = resources.files("assets")

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)
    
    def forward_up_to_second_last(self, embed):
        # Process the input through all layers except the last one
        for layer in list(self.layers)[:-1]:
            embed = layer(embed)
        return embed

class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1), embed
    
    def generate_feats(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed

class online_AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.dtype = dtype
        
        self.optimism = config.train.optimism
        assert self.optimism in ['none', 'UCB', 'bootstrap'], "optimism must be one of ['none', 'LCB', 'bootstrap']"
        
        self.eval()

    def __call__(self, images, config, D_exp=None):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        if self.optimism == 'none':
            return D_exp.model(embed).squeeze(1)

        if self.optimism == 'UCB':
            if config.train.num_gpus > 1:
                feats = D_exp.model.module.forward_up_to_second_last(embed)
            else:
                feats = D_exp.model.forward_up_to_second_last(embed)  #(B,16)
                
            raw_rewards = D_exp.model(embed)
            
            bonuses = torch.zeros_like(raw_rewards).to(feats.device)
            
            cov_mat = D_exp.cov(config)

            for idx in range(raw_rewards.shape[0]):
                feat = feats[[idx,],].t()
                invertible_cov_mat = cov_mat.to(feat.device) + config.train.osm_lambda * torch.eye(feat.shape[0]).to(feat.device)
                bonus = config.train.osm_alpha*torch.sqrt(torch.mm(feat.t(), torch.mm(torch.linalg.inv(invertible_cov_mat), feat)))
                bonuses[idx,] = bonus.squeeze(1)
            
            optimistic_rewards = raw_rewards + torch.clamp(bonuses,\
                            max=config.train.osm_clipping)
            return optimistic_rewards.squeeze(1)
        
        elif self.optimism == 'bootstrap': 
            outputs = D_exp.model(embed)
            # rand_idx = random.randint(0, len(outputs) - 1) # use a random head for output
            # return outputs[rand_idx].squeeze(1)
            
            stacked_outputs = torch.cat(outputs, dim=1) #(B, N=10)
            
            # Scale the outputs by the temperature parameter. Compute the softmin along the second dimension (dim=1) to get a tensor shaped (B, N)
            # tau = config.train.temp
            # scaled_outputs = stacked_outputs / tau
            # weights = torch.nn.functional.softmax(scaled_outputs, dim=1)
            # weighted_sum = (weights * stacked_outputs).sum(dim=1)
            
            ##### Bootstrapping version 3: use the minimum value of the outputs
            optimistic_rewards, _ = torch.max(stacked_outputs, dim=1, keepdim=True)
            
            return optimistic_rewards.squeeze(1)
        
        else:
            raise NotImplementedError


if __name__ == "__main__":
    scorer = AestheticScorerDiff(dtype=torch.float32).cuda()
    scorer.requires_grad_(False)
    for param in scorer.clip.parameters():
        assert not param.requires_grad
    for param in scorer.mlp.parameters():
        assert not param.requires_grad