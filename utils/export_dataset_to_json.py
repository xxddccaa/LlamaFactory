#!/usr/bin/env python3
"""
将 HuggingFace 数据集导出为 JSON/JSONL 格式
用于本地数据集加载
"""
import json
import argparse
from datasets import load_from_disk, load_dataset
from pathlib import Path


def export_dataset(input_path: str, output_path: str, format: str = "jsonl"):
    """
    导出数据集为 JSON 或 JSONL 格式
    
    Args:
        input_path: 输入路径（可以是 save_to_disk 保存的目录，或 HuggingFace Hub 数据集名）
        output_path: 输出文件路径
        format: 输出格式，'json' 或 'jsonl'
    """
    # 尝试加载数据集
    if Path(input_path).exists() and Path(input_path).is_dir():
        # 本地目录，尝试用 load_from_disk
        try:
            dataset = load_from_disk(input_path)
            print(f"从本地目录加载数据集: {input_path}")
        except:
            # 如果不是 save_to_disk 格式，尝试用 load_dataset
            dataset = load_dataset("arrow", data_dir=input_path, split="train")
            print(f"从本地目录加载数据集（Arrow格式）: {input_path}")
    else:
        # 可能是 HuggingFace Hub 数据集
        if "/" in input_path:
            parts = input_path.split("/")
            if len(parts) == 2:
                dataset = load_dataset(parts[0], parts[1], split="train")
            else:
                dataset = load_dataset(input_path, split="train")
        else:
            dataset = load_dataset(input_path, split="train")
        print(f"从 HuggingFace Hub 加载数据集: {input_path}")
    
    # 转换为列表
    data_list = [item for item in dataset]
    
    # 保存为 JSON 或 JSONL
    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:  # json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已导出到: {output_path}")
    print(f"共 {len(data_list)} 条数据")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 HuggingFace 数据集为 JSON/JSONL 格式")
    parser.add_argument("--input", "-i", required=True, help="输入路径（本地目录或 HuggingFace Hub 数据集名）")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径（.json 或 .jsonl）")
    parser.add_argument("--format", "-f", choices=["json", "jsonl"], default="jsonl", 
                       help="输出格式（默认: jsonl）")
    
    args = parser.parse_args()
    
    export_dataset(args.input, args.output, args.format)

