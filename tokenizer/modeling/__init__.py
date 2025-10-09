"""
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
from .dc_ae_vit import DC_AE_ViT
from .dc_ae_vit_stage2 import DC_AE_ViT_Stage2
from .dc_ae_vit_stage2_pad import DC_AE_ViT_Stage2_Pad
from .dc_ae_vit_stage2_448 import DC_AE_ViT_Stage2_448
from .dc_ae_vit_stage2_448_checkpoint import DC_AE_ViT_Stage2_448_Checkpoint

model_map = {
    'DC_AE_ViT': DC_AE_ViT,
    'DC_AE_ViT_Stage2': DC_AE_ViT_Stage2,
    'DC_AE_ViT_Stage2_Pad': DC_AE_ViT_Stage2_Pad,
    'DC_AE_ViT_Stage2_448': DC_AE_ViT_Stage2_448,
    'DC_AE_ViT_Stage2_448_Checkpoint': DC_AE_ViT_Stage2_448_Checkpoint,
}