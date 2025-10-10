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
from unilip.pipeline_gen import CustomGenPipeline
import math
import requests
import random

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

    outdir = f"{model_name}/wise_{opt.prompt_template}"
    os.makedirs(outdir, exist_ok=True)
    prompt_template = opt.prompt_template
    disable_torch_init()
    tokenizer, multi_model, context_len = load_pretrained_model_general(opt.cls, model_name)

    pipe = CustomGenPipeline(multimodal_encoder=multi_model, tokenizer=tokenizer)
    # Load all prompts
    metadatas = []
    json_dir = '../../data/WISE/data/'
    json_paths = ['cultural_common_sense.json', 'natural_science.json', 'spatio-temporal_reasoning.json']
    for json_path in json_paths:
        json_abs_path = os.path.join(json_dir, json_path)
        with open(json_abs_path) as fp:
            content = json.load(fp)
        metadatas.extend(content)

    # Split the data into chunks: each instance will process every n_chunks-th entry
    metadatas = metadatas[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index} out of {opt.n_chunks} total chunks, {len(metadatas)} samples assigned.")
    os.makedirs(outdir, exist_ok=True)
    for index, metadata in enumerate(metadatas):
        set_global_seed(seed=42)
        outpath = os.path.join(outdir, f"{metadata['prompt_id']}.png")
        
        prompt = metadata['Prompt']

        prompt = [f"Generate an image: {prompt}", "Generate an image."]
        if "qwen" in prompt_template:
            prompt = add_template(prompt)
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")
        with torch.no_grad():
            gen_img = pipe(prompt, guidance_scale=opt.scale)
            gen_img.save(outpath)

    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)






