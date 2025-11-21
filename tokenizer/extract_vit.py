import torch
import argparse

def process_checkpoint(ckpt_path, output_path):
    """
    Loads a checkpoint, filters and renames keys, and saves the new state dictionary.

    Args:
        ckpt_path (str): Path to the input checkpoint file.
        output_path (str): Path to save the processed state dictionary.
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    # 使用 map_location='cpu' 可以在没有GPU的环境下安全加载
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print("Original keys found:", len(ckpt.keys()))

    # 过滤掉不需要的键
    keys_to_delete = [key for key in ckpt.keys() if 'encoder' not in key and 'mlp1' not in key]
    for key in keys_to_delete:
        del ckpt[key]
    
    print("Keys after filtering:", list(ckpt.keys()))

    # 重命名键
    new_state = {}
    for key, value in ckpt.items():
        if 'encoder' in key:
            new_key = key.replace('encoder.', '', 1) # 使用 replace 更安全
        else: # 'mlp1' in key
            new_key = key.replace('mlp1.', '', 1)
        new_state[new_key] = value
    
    print("New renamed keys:", list(new_state.keys()))

    # 保存处理后的权重
    print(f"Saving new state dictionary to: {output_path}")
    torch.save(new_state, output_path)
    print("Done.")

if __name__ == '__main__':
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Extract and rename 'encoder' and 'mlp1' layers from a PyTorch checkpoint."
    )

    # 2. 添加命令行参数
    parser.add_argument(
        '--ckpt_path', 
        type=str, 
        required=True, 
        help='Path to the input checkpoint file'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help='Path to save the new processed state dictionary'
    )

    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 调用主函数
    process_checkpoint(args.ckpt_path, args.output_path)
