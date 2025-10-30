import json
import os
import shutil
import tarfile
from tqdm import tqdm  # 用于显示进度条
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_json(json_file_path, output_folder='t2i_files', compress=False):
    # 1. 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 创建目标文件夹
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 3. 处理每个条目（添加进度条）
    total_items = len(data)
    for item in tqdm(data, desc="处理条目", total=total_items, unit="条目"):
        image_path = item['output_image']
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 复制图片
        src_image_path = os.path.join("./ShareGPT-4o-Image", image_path)
        dst_image_path = os.path.join(output_folder, os.path.basename(image_path).replace('.png', '.jpg'))
        shutil.copy2(src_image_path, dst_image_path)

        # 生成对应的 txt 文件
        txt_content = item['input_prompt']
        txt_path = os.path.join(output_folder, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(txt_content)



def pack_file_pairs(file_pairs, folder_path, tar_name):
    """将一组文件对打包为一个 .tar 文件"""
    with tarfile.open(tar_name, "w") as tar:
        for files in tqdm(file_pairs, desc=f"打包 {tar_name}", leave=False, unit="对"):
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                tar.add(file_path, arcname=file_name)  # 不保留目录结构

def split_and_compress(folder_path, num_parts=5):
    """将文件夹内容按文件对分组，并行打包为多个 .tar 文件"""
    files = os.listdir(folder_path)

    # 步骤1: 按文件名前缀分组
    prefix_dict = {}
    for file in files:
        prefix = os.path.splitext(file)[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(file)

    # 步骤2: 构建文件对列表（每个前缀对应两个文件）
    file_pairs = []
    for prefix, filenames in prefix_dict.items():
        if len(filenames) == 2:  # 只处理成对存在的文件
            file_pairs.append(filenames)

    print(f"找到 {len(file_pairs)} 对文件")

    # 步骤3: 将文件对均分为 num_parts 份
    chunk_size = (len(file_pairs) + num_parts - 1) // num_parts
    chunks = [file_pairs[i * chunk_size:(i + 1) * chunk_size] for i in range(num_parts)]

    # 步骤4: 并行打包
    with ThreadPoolExecutor(max_workers=num_parts) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            tar_name = f"share4o_gen_part_{i + 1}.tar"
            futures.append(executor.submit(pack_file_pairs, chunk, folder_path, tar_name))
        
        for _ in tqdm(as_completed(futures), total=num_parts, desc="整体进度", unit="任务"):
            pass

    print("所有 tar 包生成完成！")


# 使用示例
json_file_path = './ShareGPT-4o-Image/text_to_image.json'
output_folder = './'
process_json(json_file_path, output_folder=output_folder)
split_and_compress(output_folder, num_parts=5)