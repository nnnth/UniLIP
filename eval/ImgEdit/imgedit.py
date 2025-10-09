import argparse
import json
import os
import numpy as np
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
import cv2
from unilip.constants import *
from unilip.model.builder import load_pretrained_model_general
from unilip.utils import disable_torch_init
from unilip.pipeline_edit import CustomEditPipeline
import math
import requests
import random
from transformers import AutoProcessor

def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_template(prompt):
    instruction = ('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n<img>')
    pos_prompt = instruction.format(input=prompt[0])

    cfg_prompt = instruction.format(input=prompt[1])
    return [pos_prompt, cfg_prompt]

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cls",
        type=str,
        default="",
        help="CLASS NAME"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="qwen",
        help="Template format"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=4.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    opt = parser.parse_args()
    return opt

def main(opt):
    model_name = opt.model

    outdir = f"{model_name}/imgedit_{opt.prompt_template}_10.3_debug"
    visdir = f"{model_name}/imgedit_vis_{opt.prompt_template}_10.3_debug"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(visdir, exist_ok=True)
    prompt_template = opt.prompt_template
    disable_torch_init()
    tokenizer, multi_model, context_len = load_pretrained_model_general(opt.cls, model_name)
    image_processor = AutoProcessor.from_pretrained(multi_model.config.mllm_hf_path).image_processor

    pipe = CustomEditPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer, image_processor=image_processor)

    # Load all prompts
    json_dir = '../../data/ImgEdit/Benchmark/singleturn'
    json_name = 'singleturn.json'
    json_abs_path = os.path.join(json_dir, json_name)
    with open(json_abs_path) as fp:
        metadatas = json.load(fp)
    metadata_keys = list(metadatas.keys())
    metadata_values = list(metadatas.values())
    print(metadata_keys)
    print(len(metadata_keys))

    # Split the data into chunks: each instance will process every n_chunks-th entry
    metadata_keys = metadata_keys[opt.index::opt.n_chunks]
    metadata_values = metadata_values[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index} out of {opt.n_chunks} total chunks, {len(metadata_values)} samples assigned.")
    os.makedirs(outdir, exist_ok=True)
    index = 0
    generator = torch.Generator(device=multi_model.device).manual_seed(42)
    for key, value in zip(metadata_keys, metadata_values):
        set_global_seed(seed=42)
        outpath = os.path.join(outdir, f"{key}.png")
        
        prompt = value['prompt']

        prompt =[f"Edit the image: {prompt}\n<image>", "Edit the image.\n<image>"]
        if "qwen" in prompt_template:
            multimodal_prompts = add_template(prompt)
        print(f"Prompt ({index: >3}/{len(metadata_values)}): '{multimodal_prompts}'")

        input_image_path = os.path.join(json_dir, value['id'])
        input_image = Image.open(input_image_path)
        multimodal_prompts.append(input_image)
        with torch.no_grad():
            gen_img = pipe(multimodal_prompts, guidance_scale=opt.scale, generator=generator)
            gen_img.save(outpath)

            width, height = gen_img.size
            input_image = input_image.resize((width, height))
            input_array = np.array(input_image)
            gen_array = np.array(gen_img)
            diff_array = np.abs(input_array - gen_array)
            diff_array = np.clip(diff_array, 0, 255).astype(np.uint8)
            diff_image = Image.fromarray(diff_array)

            result = Image.new('RGB', (width * 3, height))
            result.paste(input_image, (0, 0))
            result.paste(gen_img, (width, 0))
            result.paste(diff_image, (2 * width, 0))

            outpath = os.path.join(visdir, f"{value['prompt'][:128]}.png")
            result.save(outpath)

        index += 1

    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)






