#!/bin/bash

# 脚本：untar.sh
# 目的：只解压当前目录下的 *.tar 文件
cd ShareGPT-4o-Image
echo "--- 准备开始解压 *.tar 文件 ---"

# 使用 find 命令查找当前目录 (深度为1) 的所有 .tar 文件
# -maxdepth 1: 只在当前目录查找，不包括子目录
# -type f:     只查找文件
# -name "*.tar": 只查找后缀为 .tar 的文件
find . -maxdepth 1 -type f -name "*.tar" | while read -r file; do
  
  # 检查找到的文件是否真的存在 (防止空循环)
  if [ -n "$file" ]; then
    echo " "
    echo "▶️ 正在解压: $file"
    
    # -x: 提取 (extract)
    # -v: 详细模式 (verbose)，显示解压过程
    # -f: 指定文件名
    tar -xvf "$file"
    
    # 检查上一条命令 (tar) 是否成功执行
    if [ $? -eq 0 ]; then
      echo "✅ 成功解压: $file"
    else
      echo "❌ 解压失败: $file (请检查文件是否损坏)"
    fi
  fi
done

echo " "
echo "--- 所有 .tar 文件处理完毕 ---"