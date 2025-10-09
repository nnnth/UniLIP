from diffusers import SanaTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
import torch 

def build_sana(sana_path):
    dit = SanaTransformer2DModel.from_pretrained(sana_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sana_path, subfolder="scheduler")
    return dit, None, noise_scheduler