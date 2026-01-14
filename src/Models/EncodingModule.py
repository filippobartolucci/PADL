import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from .TransformerBlock import *

import torch
import torch.nn as nn

class EncodingModule(nn.Module):
    def __init__(self, in_channels=3, n_heads=8, d_head=64, depth=1, dropout=0., 
                 context_dim=None, img_size=128, patch_size=8, scale = 1):
        super(EncodingModule, self).__init__()

        self.original_img_size = img_size
        img_size = img_size // scale
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        patch_dim = in_channels * patch_size ** 2

        self.template = nn.Parameter(torch.randn(1, 3, img_size, img_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, inner_dim))
        
        # Image Embedder: B x C x H x W -> B x N_Token x Token_dim
        self.I_proj_in = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )

        # Template Embedder: B x C x H x W -> B x N_Token x Token_dim
        self.T_proj_in = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )
        
        self.T_Blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)])
        
        self.T_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                           h = img_size // patch_size, 
                           w = img_size // patch_size, 
                           p1 = patch_size, 
                           p2 = patch_size),
        )


    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.I_proj_in(x) # (B, 3, 128, 128) -> (B, 256, 512)
        
        template = self.T_proj_in(self.template)
        template = template.repeat(b, 1, 1)
        template += self.pos_embedding
        
        out = self.T_Blocks[0](template, context=x)
        
        for block in self.T_Blocks[1:]:
            out = block(out, context=x)

        out = self.T_out(out) # (B, 256, 512) -> (B, 3, 64, 64)
        out = torch.tanh(out) # Force the output to be in the range [-1, 1]
        return out
        
    
class ViT_Extractor(nn.Module):
    def __init__(self, in_channels=3, n_heads=8, d_head=64, depth=1, dropout=0., 
                 context_dim=None, img_size=128, patch_size=8, scale = 1):
        super(ViT_Extractor, self).__init__()

        self.original_img_size = img_size
        img_size = img_size // scale
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        patch_dim = in_channels * patch_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, inner_dim))
        
        # Image Embedder: B x C x H x W -> B x N_Token x Token_dim
        self.I_proj_in = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )

        
        self.T_Blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)])
        
        self.T_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                           h = img_size // patch_size, 
                           w = img_size // patch_size, 
                           p1 = patch_size, 
                           p2 = patch_size),
        )


    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.I_proj_in(x) # (B, 3, 128, 128) -> (B, 256, 512)
        
        x += self.pos_embedding
        
        out = self.T_Blocks[0](x)
        
        for block in self.T_Blocks[1:]:
            out = block(out)

        out = self.T_out(out) # (B, 256, 512) -> (B, 3, 64, 64)
        out = torch.tanh(out) # Force the output to be in the range [-1, 1]
        return out
        
    