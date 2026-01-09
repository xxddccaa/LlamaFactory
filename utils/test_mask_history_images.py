#!/usr/bin/env python3
"""
测试 mask_history_sample 时图片列表的拆分是否正确
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llamafactory.data.data_utils import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments

def test_mask_history_sample_with_images():
    """测试 mask_history_sample 时图片列表的拆分"""
    dataset_attr = DatasetAttr("file", "test_dataset")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = "conversations"
    dataset_attr.role_tag = "from"
    dataset_attr.content_tag = "value"
    dataset_attr.user_tag = "user"
    dataset_attr.assistant_tag = "assistant"
    dataset_attr.mask_history_sample = True
    dataset_attr.max_human_steps = 2
    dataset_attr.images = "image"

    data_args = DataArguments()

    # 创建一个包含图片的多轮对话示例
    example = {
        "conversations": [
            {"from": "user", "value": "Image1: <image>. Image2: <image>. Image3: <image>. Describe Image3."},
            {"from": "assistant", "value": "Image3 shows..."},
            {"from": "user", "value": "Image4: <image>. What about Image4?"},
            {"from": "assistant", "value": "Image4 shows..."},
            {"from": "user", "value": "Image5: <image>. Describe Image5."},
            {"from": "assistant", "value": "Image5 shows..."},
        ],
        "image": [
            "img1.jpg",
            "img2.jpg",
            "img3.jpg",
            "img4.jpg",
            "img5.jpg",
        ]
    }

    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    outputs = dataset_converter(example)

    print(f"原始样本有 {len(example['image'])} 张图片")
    print(f"拆分成 {len(outputs)} 个子样本\n")

    for i, output in enumerate(outputs):
        prompt = output["_prompt"]
        response = output["_response"]
        images = output.get("_images", [])
        
        # 统计标记数量
        total_image_tokens = 0
        for msg in prompt + response:
            total_image_tokens += msg.get("content", "").count("<image>")
        
        print(f"样本 {i+1}:")
        print(f"  Prompt 消息数: {len(prompt)}")
        print(f"  Response 消息数: {len(response)}")
        print(f"  <image> 标记数: {total_image_tokens}")
        print(f"  图片数量: {len(images) if images else 0}")
        print(f"  图片列表: {images if images else 'None'}")
        
        if total_image_tokens != (len(images) if images else 0):
            print(f"  ❌ 错误: 标记数量 ({total_image_tokens}) 与图片数量 ({len(images) if images else 0}) 不匹配!")
        else:
            print(f"  ✓ 正确: 标记数量与图片数量匹配")
        print()

if __name__ == "__main__":
    test_mask_history_sample_with_images()

