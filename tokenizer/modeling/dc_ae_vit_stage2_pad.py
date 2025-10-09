"""This file contains the model definition of TiTok.

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
import torch.nn as nn
from einops import rearrange

from modeling.modules.base_model import BaseModel
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel, AutoConfig
import math 
import torch.nn.functional as F
from copy import deepcopy 
import numpy as np
from diffusers.models import AutoencoderDC
import cv2

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

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.mlp = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x):
        return x + self.mlp(x)

class DC_AE_ViT_Stage2_Pad(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        embed_dim = config.model.embed_dim 
        self.patch_size = config.model.patch_size

        # 加载配置
        path = self.config.model.mllm_path
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True)
        self.encoder = model.vision_model
        vit_dim = model.vision_model.embeddings.patch_embedding.weight.shape[0]
        llm_hidden_size = model.config.llm_config.hidden_size

        # from ViT to vae decoder
        down_blocks = []
        for i in range(3):
            down_blocks.append(ResBlock(
                llm_hidden_size,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.down_mlp = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        self.apply(self._init_weights)
        # reload pretrain as they are reinitialized above
        path = self.config.model.mllm_path
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True)
        self.encoder = model.vision_model
        self.mlp1 = model.mlp1

        for layer in self.encoder.encoder.layers:
            try:
                layer.drop_path1.drop_prob = 0.0
                layer.drop_path2.drop_prob = 0.0
            except:
                continue
        print("should no drop out", self.encoder)

        dc_ae = AutoencoderDC.from_pretrained(self.config.model.dc_ae_path, torch_dtype=torch.float32)
        self.decoder = dc_ae.decoder
        for name, param in self.decoder.named_parameters():
            if len(param.data.shape) == 4:
                param.data = param.data.to(memory_format=torch.channels_last)
        if self.config.model.stage1_ckpt != '':
            msg = self.load_state_dict(torch.load(self.config.model.stage1_ckpt),strict=False)
            print(f"load {self.config.model.stage1_ckpt}")
            print("Missing keys:")
            print(msg.missing_keys)
            print("Unexpected keys:")
            print(msg.unexpected_keys)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def padding(self, tensor):
        # renormalize to [0,1]
        mean, std = self.config.dataset.preprocessing.normalize_mean, self.config.dataset.preprocessing.normalize_std
        std = torch.tensor(std).to(tensor.device)
        mean = torch.tensor(mean).to(tensor.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        # inputs is normalized by imagenet, while reconstruction predict [-1,1]
        # here align both to [0, 1]
        original_images = tensor * std + mean
        b, c, h, w = original_images.shape
    
        # 计算目标高度和宽度（28的最小倍数，大于等于当前尺寸）
        new_h = math.ceil(h / 28) * 28
        new_w = math.ceil(w / 28) * 28
        
        # 计算需要填充的像素数
        pad_h = new_h - h
        pad_w = new_w - w
        
        # 计算顶部/底部和左侧/右侧的填充量
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # 应用对称填充（避免边缘突变）
        padded_images = F.pad(original_images, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        import cv2 
        np_image = padded_images.clamp(0,1)[0].permute(1,2,0).cpu().numpy()[:,:,[2,1,0]]*255
        cv2.imwrite("pad_image.jpg",np_image.astype(np.uint8))

        # normalize 
        normalized_images = (padded_images - mean) / std
        self.pad_info = (pad_top, pad_bottom, pad_left, pad_right)
        return normalized_images
    
    def remove_padding(self, tensor):
        """
        从填充的张量中移除填充部分
        参数:
            tensor: 填充后的张量，形状为(b, c, h_padded, w_padded)
            padding_info: 来自pad_to_multiple_of_28的填充信息 (pad_top, pad_bottom, pad_left, pad_right)
        返回:
            unpadded_tensor: 移除填充后的原始图像区域
        """
        pad_top, pad_bottom, pad_left, pad_right = self.pad_info
        
        # 计算原始图像区域
        _, _, h_padded, w_padded = tensor.shape
        unpadded_tensor = tensor[
            :, :,
            pad_top : h_padded - pad_bottom,
            pad_left : w_padded - pad_right
        ]
        np_image = unpadded_tensor.clamp(-1,1)[0].permute(1,2,0).float().detach().cpu().numpy()[:,:,[2,1,0]]
        np_image = (np_image + 1)/2*255
        cv2.imwrite("unpad_image.jpg",np_image.astype(np.uint8))
        # exit()
        return unpadded_tensor
    
    def resize_down(self, tensor):
        """缩小到28/32比例"""
        _, _, h, w = tensor.shape
        
        # 计算目标尺寸（四舍五入）
        target_h = round(h * 28 / 32)
        target_w = round(w * 28 / 32)
        
        return F.interpolate(
            tensor, 
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=True
        )
    def encode(self, x):
        x = self.padding(x)
        vit_embeds = self.encoder.embeddings(x)
        for idx, encoder_layer in enumerate(self.encoder.encoder.layers):
            vit_embeds = encoder_layer(vit_embeds)
        vit_embeds = vit_embeds[:,1:,:].contiguous().float()
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        distill_output = vit_embeds.clone()

        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)

        vit_embeds = vit_embeds.permute(0,2,1).contiguous()
        b, c, hw = vit_embeds.shape
        vit_embeds = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))

        z = vit_embeds

        return z.float(), {'distill_feat': distill_output}
    
    def decode(self, z_quantized):
        dec = self.decoder(z_quantized)
        dec = self.resize_down(dec)
        dec = self.remove_padding(dec)
        return dec
    
    def forward(self, x):
        z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, result_dict