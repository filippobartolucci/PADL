import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from .TransformerBlock import *

class DecodingModule(nn.Module):
    def __init__(self, img_channels=3, n_heads=8, d_head=64, S_depth=1, M_depth = 1 , dropout=0. ,img_size=128, patch_size=8):
        super(DecodingModule, self).__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.inner_dim = n_heads * d_head
        self.patch_dim = img_channels * patch_size ** 2
        self.pos_embedding = nn.Parameter(torch.rand(1, (img_size // patch_size) ** 2, self.inner_dim))
        self.class_token = nn.Parameter(torch.rand(1, 1, self.inner_dim))
        
        self.Proj_In = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.D_out = nn.Sequential(
            nn.Linear(self.inner_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        self.SR_Block = nn.ModuleList(
            [BasicTransformerBlock(self.inner_dim, n_heads, d_head, dropout=dropout)
            for _ in range(S_depth)])
    
        self.M_Block = nn.ModuleList(
            [BasicTransformerBlock(self.inner_dim, n_heads, d_head, dropout=dropout)
            for _ in range(M_depth)])
        
        
        self.S_out = nn.Sequential(
            nn.LayerNorm(self.inner_dim),
            nn.Linear(self.inner_dim, self.patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h = img_size // patch_size, 
                        w = img_size // patch_size, 
                        p1 = patch_size, 
                        p2 = patch_size),
        )

        
        self.M_out = nn.Sequential(
            nn.LayerNorm(self.inner_dim),
            nn.Linear(self.inner_dim, self.patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h = img_size // patch_size, 
                        w = img_size // patch_size, 
                        p1 = patch_size, 
                        p2 = patch_size),
            
        )

        self.M_out2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
        )


        self.D_out = nn.Sequential(
            nn.Linear(self.inner_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        

    def forward(self, img):
        b = img.shape[0]

        o_img = img.clone()
        img = self.Proj_In(img)


        img += self.pos_embedding

        signal = img.clone()
        for block in self.SR_Block:
            signal = block(signal, context=signal)

        class_token = repeat(self.class_token, '() n d -> b n d', b = b)
        map = torch.cat((class_token, img), dim=1)

        signal = torch.tanh(signal)
        for block in self.M_Block:
            map = block(map, context=signal)

        detection_token = map[:, 0, :]
        map = map[:, 1:, :]

        map = self.M_out(map)
        signal = self.S_out(signal)
        detection = self.D_out(detection_token)

        map = torch.cat((map, o_img), dim=1)
        map = self.M_out2(map)
        map = torch.sigmoid(map)

        return map, signal, detection