import argparse
import os
import sys
import time

import webdataset as wds
from datasets import load_dataset

from PIL import Image
from tqdm import tqdm
import json
Image.MAX_IMAGE_PIXELS = 10000000000

save_dir = './edit_sft'
data_root = './ShareGPT-4o-Image'
json_path = './ShareGPT-4o-Image/text_and_image_to_image.json'
os.makedirs(save_dir, exist_ok=True)
list_data_dict = []
with open(json_path) as f:
    content = json.load(f)
for single_data in content:
    input_image_paths = single_data['input_image']
    output_image = single_data['output_image']
    for k in range(len(input_image_paths)):
        input_image_paths[k] = os.path.join(data_root, input_image_paths[k])
    output_image = os.path.join(data_root, output_image)
    single_data['input_image'] = input_image_paths
    single_data['output_image'] = output_image
    list_data_dict.append(single_data)
print(len(list_data_dict))
opat = os.path.join(save_dir, "%06d.tar")
output = wds.ShardWriter(opat, maxcount=10000)
now = time.time()
for i, single_data in tqdm(enumerate(list_data_dict)):
    assert len(single_data['input_image']) == 1
    input_image_path = single_data['input_image'][0]
    output_image_path = single_data['output_image']
    input_img = Image.open(input_image_path)
    output_img = Image.open(output_image_path)
    prompt = single_data['input_prompt']
    output.write({"__key__": f"{i:08d}", "input.jpg": input_img.convert("RGB"), "output.jpg": output_img.convert("RGB"), "txt": prompt})
output.close()