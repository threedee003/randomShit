
from optparse import Option
import torch
import torch.nn as nn
import math
from typing import Optional
from torchsummary import summary

class PolicyNet(nn.Module):
    def __init__(self,
                num_cameras: Optional[int] = 2,
                camera_resolution: Optional[int] = 128,
                embed_dim: Optional[int] = 44,
                no_of_attn_blocks: Optional[int] = 6,
                num_heads: Optional[int] = 4
                ):
                super(PolicyNet, self).__init__()
                self.camera_res = camera_resolution
                self.embed_dim = embed_dim
                self.no_of_attn_blocks = no_of_attn_blocks
                self.num_heads = num_heads


                self.image_downsampler1 = ImageDownSampler()
                self.image_downsampler2 = ImageDownSampler()
                self.attn_blocks = nn.ModuleList([
                    AttentionBlock(embed_dim=self.embed_dim, num_heads=self.num_heads) for _ in range(self.no_of_attn_blocks)
                ])

    def forward(self, x1, x2):
        x1 = self.image_downsampler1(x1)
        x2 = self.image_downsampler2(x2)
        x = torch.concat((x1,x2), dim=1)
        print(x.shape)
        batch_size, channels, height, width = x.size() 
        x = x.view(batch_size, channels, height * width) 
        x = x.permute(0, 2, 1)
        for blocks in self.attn_blocks:
            x = blocks(x)
        return x

        
        


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(self.embed_dim)


    def _compute_output_dim(self, x):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.layernorm(attn_output)
        return attn_output



class ImageDownSampler(nn.Module):
    def __init__(self, camera_resolution: Optional[int] = 128):
        super(ImageDownSampler, self).__init__()
        
        self.camera_res = camera_resolution
        self.in_channels = 3
        
        l1 = self._compute_output_dim(self.camera_res, kernel_size=5, stride=2, padding=0)
        l2 = self._compute_output_dim(l1, kernel_size = 5, stride=2, padding=0)
        self.conv_downsampling = nn.Sequential(
            nn.Conv2d(self.in_channels, 15, kernel_size=5, stride=2),
            nn.LayerNorm(l1),
            nn.Mish(),
            nn.Conv2d(15, 22, kernel_size=5, stride=2),
            nn.LayerNorm(l2),
            nn.Mish()
        )
    
    def _compute_output_dim(self, dim: int, kernel_size: int, stride: int, padding: int) -> int:
        """Compute the output dimension after applying the convolution."""
        return math.floor((dim + 2 * padding - kernel_size) / stride) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_downsampling(x)
        return x




if __name__ == "__main__":
    img = ImageDownSampler()
    x = torch.randn(1,3, 128, 128)
    # y = img(x)
    # print(y.shape)
    # x = torch.randn(1,44,29,29)
    # attn = AttentionBlock(embed_dim=44, num_heads=4)
    # y = attn(x)
    # print(y.shape)
    model = PolicyNet()
    summary(model, input_size=x.shape)
    y = model(x,x)
    print(y.shape)
    summary(model)


    