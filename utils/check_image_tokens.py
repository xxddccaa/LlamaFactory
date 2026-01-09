#!/usr/bin/env python3
"""
检查数据集中 <image> 标记数量是否与实际图片数量匹配
"""
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def count_image_tokens(text: str) -> int:
    """统计文本中 <image> 标记的数量"""
    # 匹配 <image> 标记（不区分大小写，允许前后有空格）
    pattern = r'<image>'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


def check_dataset(json_file: str, image_key: str = "image", conversations_key: str = "conversations"):
    """
    检查数据集中的图片标记数量
    
    Args:
        json_file: JSON 文件路径
        image_key: 图片字段的键名（默认: "image"）
        conversations_key: 对话字段的键名（默认: "conversations"）
    """
    print(f"正在读取文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"错误: 数据格式不正确，期望是列表，实际是 {type(data)}")
        return
    
    print(f"总共 {len(data)} 个样本\n")
    
    mismatches = []
    stats = defaultdict(int)
    
    for idx, sample in enumerate(data):
        # 获取图片列表
        images = sample.get(image_key, [])
        if not isinstance(images, list):
            images = []
        
        image_count = len(images)
        
        # 统计所有对话中的 <image> 标记
        conversations = sample.get(conversations_key, [])
        if not isinstance(conversations, list):
            conversations = []
        
        total_image_tokens = 0
        token_locations = []
        
        for conv_idx, conv in enumerate(conversations):
            if isinstance(conv, dict):
                content = conv.get("value", "") or conv.get("content", "")
                if content:
                    tokens = count_image_tokens(content)
                    total_image_tokens += tokens
                    if tokens > 0:
                        token_locations.append(f"conversation[{conv_idx}]: {tokens}个标记")
        
        # 检查是否匹配
        if total_image_tokens != image_count:
            mismatches.append({
                "index": idx,
                "sample_id": sample.get("id", idx),
                "image_tokens": total_image_tokens,
                "image_count": image_count,
                "difference": total_image_tokens - image_count,
                "token_locations": token_locations,
                "images": images[:3] if len(images) > 3 else images,  # 只显示前3个
                "first_conversation": conversations[0].get("value", "")[:100] if conversations else ""
            })
            stats["mismatch"] += 1
        else:
            stats["match"] += 1
        
        # 统计图片数量分布
        stats[f"images_{image_count}"] += 1
        stats[f"tokens_{total_image_tokens}"] += 1
    
    # 打印统计信息
    print("=" * 80)
    print("统计信息:")
    print(f"  匹配的样本: {stats['match']}")
    print(f"  不匹配的样本: {stats['mismatch']}")
    print(f"  总样本数: {len(data)}")
    print()
    
    # 打印图片数量分布
    print("图片数量分布:")
    for key in sorted(stats.keys()):
        if key.startswith("images_"):
            count = key.replace("images_", "")
            print(f"  {count} 张图片: {stats[key]} 个样本")
    print()
    
    # 打印标记数量分布
    print("标记数量分布:")
    for key in sorted(stats.keys()):
        if key.startswith("tokens_"):
            count = key.replace("tokens_", "")
            print(f"  {count} 个标记: {stats[key]} 个样本")
    print()
    
    # 打印不匹配的详细信息
    if mismatches:
        print("=" * 80)
        print(f"发现 {len(mismatches)} 个不匹配的样本:\n")
        
        for i, mismatch in enumerate(mismatches[:20], 1):  # 只显示前20个
            print(f"[样本 {mismatch['index']}] ID: {mismatch['sample_id']}")
            print(f"  <image> 标记数量: {mismatch['image_tokens']}")
            print(f"  实际图片数量: {mismatch['image_count']}")
            print(f"  差异: {mismatch['difference']} ({'+' if mismatch['difference'] > 0 else ''}{mismatch['difference']})")
            if mismatch['token_locations']:
                print(f"  标记位置: {', '.join(mismatch['token_locations'])}")
            if mismatch['images']:
                print(f"  图片路径示例: {mismatch['images']}")
            if mismatch['first_conversation']:
                print(f"  第一条对话预览: {mismatch['first_conversation']}...")
            print()
        
        if len(mismatches) > 20:
            print(f"... 还有 {len(mismatches) - 20} 个不匹配的样本未显示\n")
        
        print("=" * 80)
        print("建议:")
        print("  1. 检查不匹配的样本，确保 <image> 标记数量与图片数量一致")
        print("  2. 如果图片数量多于标记，可能需要添加更多 <image> 标记")
        print("  3. 如果标记数量多于图片，可能需要添加更多图片路径或移除多余的标记")
    else:
        print("=" * 80)
        print("✓ 所有样本的 <image> 标记数量都与图片数量匹配！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查数据集中 <image> 标记数量是否与实际图片数量匹配")
    parser.add_argument("json_file", help="JSON 数据集文件路径")
    parser.add_argument("--image-key", default="image", help="图片字段的键名（默认: image）")
    parser.add_argument("--conversations-key", default="conversations", 
                       help="对话字段的键名（默认: conversations）")
    
    args = parser.parse_args()
    
    if not Path(args.json_file).exists():
        print(f"错误: 文件不存在: {args.json_file}")
        exit(1)
    
    check_dataset(args.json_file, args.image_key, args.conversations_key)

