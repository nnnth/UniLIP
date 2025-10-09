"""This file contains perceptual loss module using LPIPS and ConvNeXt-S.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

class DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True)
        self.ref_vit = model.vision_model
        self.ref_mlp1 = model.mlp1

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1], need imgnet normalize
        # imgnetnorm
        std = torch.tensor([0.229,0.224,0.225]).to(x.device)
        mean = torch.tensor([0.485,0.456,0.406]).to(x.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        x = (x - mean) / std
        # Always in eval mode.
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.ref_vit.embeddings(x)
            for idx, encoder_layer in enumerate(self.ref_vit.encoder.layers):
                vit_embeds = encoder_layer(vit_embeds)
            vit_embeds = vit_embeds[:,1:,:].contiguous().float()
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = pixel_shuffle(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.ref_mlp1(vit_embeds)
        target_feat = vit_embeds

        distill_loss = F.mse_loss(
                out_feat,
                target_feat,
                reduction="mean")
        loss += distill_loss

        return loss
