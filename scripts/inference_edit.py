from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
import torch
import sys
import os
from tqdm import tqdm
from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.mm_utils import get_model_name_from_path
from unilip.pipeline_edit import CustomEditPipeline
import random

model_path = sys.argv[1]
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, multi_model, context_len = load_pretrained_model_general('UniLIP_InternVLForCausalLM', model_path, None, model_name)

from transformers import AutoProcessor
image_processor = AutoProcessor.from_pretrained(multi_model.config.mllm_hf_path).image_processor

pipe = CustomEditPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer, image_processor=image_processor)

def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image

def add_template(prompt):
    instruction = ('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n<img>')
    pos_prompt = instruction.format(input=prompt[0])

    cfg_prompt = instruction.format(input=prompt[1])
    return [pos_prompt, cfg_prompt]

def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

generator = torch.Generator(device=multi_model.device).manual_seed(42)
prompt = "Replace the camper van in the image with a hot air balloon."
input_image_path = "../demo/edit_input.jpg"
input_image = Image.open(input_image_path)
set_global_seed(seed=42)
gen_images = []
for i in range(1):
    multimodal_prompts = add_template([f"Edit the image: {prompt}\n<image>", "Edit the image.\n<image>"])
    multimodal_prompts.append(input_image)
    gen_img = pipe(multimodal_prompts, guidance_scale=4.5, generator=generator)
    gen_images.append(gen_img)
print(f"finish {prompt}")

grid_image = create_image_grid(gen_images, 1, 1)
grid_image.save(f"{prompt[:100]}.png")



