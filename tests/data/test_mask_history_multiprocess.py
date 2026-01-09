# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
测试 mask_history_sample 功能在多进程环境下的兼容性
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import Dataset
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.converter import align_dataset
from llamafactory.hparams import DataArguments
from transformers import Seq2SeqTrainingArguments


def test_mask_history_sample_multiprocess():
    """
    测试 mask_history_sample 数据处理在多进程下不会出错
    """
    print("\n" + "=" * 80)
    print("测试: mask_history_sample 多进程兼容性")
    print("=" * 80)
    
    # 创建测试数据
    test_data = [
        {
            "conversations": [
                {"from": "user", "value": "Question 1?"},
                {"from": "assistant", "value": "Answer 1."},
                {"from": "user", "value": "Question 2?"},
                {"from": "assistant", "value": "Answer 2."},
                {"from": "user", "value": "Question 3?"},
                {"from": "assistant", "value": "Answer 3."},
            ],
            "image": []
        },
        {
            "conversations": [
                {"from": "user", "value": "Hello"},
                {"from": "assistant", "value": "Hi there!"},
                {"from": "user", "value": "How are you?"},
                {"from": "assistant", "value": "I'm doing well!"},
            ],
            "image": []
        },
    ]
    
    # 创建 Dataset
    dataset = Dataset.from_list(test_data)
    print(f"原始数据集大小: {len(dataset)}")
    
    # 创建 DatasetAttr
    dataset_attr = DatasetAttr(
        load_from="file",
        dataset_name="test_dataset",
        formatting="sharegpt",
        messages="conversations",
        images="image",
        role_tag="from",
        content_tag="value",
        user_tag="user",
        assistant_tag="assistant",
        mask_history_sample=True,
        max_human_steps=2
    )
    
    # 创建 DataArguments
    data_args = DataArguments(
        preprocessing_num_workers=2,  # 测试多进程
        overwrite_cache=True,
    )
    
    # 创建 TrainingArguments
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir,
        )
        
        # 执行数据对齐（会触发拆分）
        print("\n开始数据对齐和拆分...")
        aligned_dataset = align_dataset(dataset, dataset_attr, data_args, training_args)
        
        print(f"拆分后数据集大小: {len(aligned_dataset)}")
        print(f"拆分比例: {len(aligned_dataset) / len(dataset):.2f}x")
        
        # 验证字段存在
        first_sample = aligned_dataset[0]
        print("\n第一个样本的字段:")
        for key in first_sample.keys():
            value = first_sample[key]
            if isinstance(value, list):
                print(f"  {key}: list[{len(value)}]")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # 验证 _mask_history_sample 字段
        assert "_mask_history_sample" in first_sample, "缺少 _mask_history_sample 字段"
        assert first_sample["_mask_history_sample"] == True, "_mask_history_sample 应该为 True"
        
        # 验证拆分逻辑
        # 第一个样本有 3 个 assistant 回复，应该拆分为 3 个样本
        # 第二个样本有 2 个 assistant 回复，应该拆分为 2 个样本
        # 总共应该是 3 + 2 = 5 个样本
        expected_samples = 3 + 2
        assert len(aligned_dataset) == expected_samples, f"期望 {expected_samples} 个样本，实际 {len(aligned_dataset)}"
        
        # 验证 prompt 的最大 human 数量
        max_human_count = 0
        for i in range(len(aligned_dataset)):
            sample = aligned_dataset[i]
            prompt = sample["_prompt"]
            human_count = sum(1 for msg in prompt if msg["role"] == "user")
            max_human_count = max(max_human_count, human_count)
            
            # 每个样本的 response 应该只有一个
            assert len(sample["_response"]) == 1, f"样本 {i} 的 response 数量不为 1"
            assert sample["_response"][0]["role"] == "assistant", f"样本 {i} 的 response 角色不是 assistant"
        
        # max_human_steps=2，所以 prompt 中最多应该有 2 个 user 消息
        assert max_human_count <= 2, f"发现 prompt 中有 {max_human_count} 个 user 消息，超过 max_human_steps=2"
        
        print("\n✅ 所有验证通过!")
        print("=" * 80)


def test_mask_history_sample_with_images():
    """
    测试 mask_history_sample 数据处理对图像的处理
    """
    print("\n" + "=" * 80)
    print("测试: mask_history_sample 图像处理")
    print("=" * 80)
    
    # 创建带图像的测试数据
    test_data = [
        {
            "conversations": [
                {"from": "user", "value": "Image: <image>. What's this?"},
                {"from": "assistant", "value": "It's a cat."},
                {"from": "user", "value": "Image: <image>. And this?"},
                {"from": "assistant", "value": "It's a dog."},
            ],
            "image": ["image1.jpg", "image2.jpg"]
        },
    ]
    
    # 创建 Dataset
    dataset = Dataset.from_list(test_data)
    print(f"原始数据集大小: {len(dataset)}")
    print(f"图像数量: {len(test_data[0]['image'])}")
    
    # 创建 DatasetAttr
    dataset_attr = DatasetAttr(
        load_from="file",
        dataset_name="test_dataset",
        formatting="sharegpt",
        messages="conversations",
        images="image",
        role_tag="from",
        content_tag="value",
        user_tag="user",
        assistant_tag="assistant",
        mask_history_sample=True,
        max_human_steps=2
    )
    
    # 创建 DataArguments
    data_args = DataArguments(
        preprocessing_num_workers=2,
        overwrite_cache=True,
    )
    
    # 创建 TrainingArguments
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir,
        )
        
        # 执行数据对齐
        print("\n开始数据对齐和拆分...")
        aligned_dataset = align_dataset(dataset, dataset_attr, data_args, training_args)
        
        print(f"拆分后数据集大小: {len(aligned_dataset)}")
        
        # 验证图像分配
        for i in range(len(aligned_dataset)):
            sample = aligned_dataset[i]
            images = sample["_images"]
            
            # 统计 prompt 和 response 中的 <image> token 数量
            image_token_count = 0
            for msg in sample["_prompt"] + sample["_response"]:
                content = msg.get("content", "")
                image_token_count += content.count("<image>")
            
            print(f"\n样本 {i}:")
            print(f"  图像数量: {len(images)}")
            print(f"  <image> token 数量: {image_token_count}")
            
            # 图像数量应该匹配 token 数量（或者为 0）
            if image_token_count > 0:
                assert len(images) == image_token_count or len(images) == 0, \
                    f"样本 {i}: 图像数量 ({len(images)}) 与 token 数量 ({image_token_count}) 不匹配"
        
        print("\n✅ 图像处理验证通过!")
        print("=" * 80)


def test_single_vs_multiprocess_consistency():
    """
    测试单进程和多进程的结果一致性（在 tokenization 之前的阶段）
    """
    print("\n" + "=" * 80)
    print("测试: 单进程 vs 多进程一致性")
    print("=" * 80)
    
    # 创建测试数据 - 每个样本有 5 轮对话
    test_data = []
    for _ in range(10):  # 10 个样本
        conversations = []
        for i in range(1, 6):  # 5 turns
            conversations.append({"from": "user", "value": f"Question {i}?"})
            conversations.append({"from": "assistant", "value": f"Answer {i}."})
        test_data.append({
            "conversations": conversations,
            "image": []
        })
    
    dataset = Dataset.from_list(test_data)
    print(f"测试数据集大小: {len(dataset)}")
    
    # 创建 DatasetAttr
    dataset_attr = DatasetAttr(
        load_from="file",
        dataset_name="test_dataset",
        formatting="sharegpt",
        messages="conversations",
        images="image",
        role_tag="from",
        content_tag="value",
        user_tag="user",
        assistant_tag="assistant",
        mask_history_sample=True,
        max_human_steps=2
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_dir,
        )
        
        # 测试单进程
        print("\n使用单进程处理...")
        data_args_single = DataArguments(
            preprocessing_num_workers=1,
            overwrite_cache=True,
        )
        aligned_single = align_dataset(dataset, dataset_attr, data_args_single, training_args)
        print(f"单进程结果: {len(aligned_single)} 个样本")
        
        # 测试多进程
        print("\n使用多进程处理...")
        data_args_multi = DataArguments(
            preprocessing_num_workers=4,
            overwrite_cache=True,
        )
        aligned_multi = align_dataset(dataset, dataset_attr, data_args_multi, training_args)
        print(f"多进程结果: {len(aligned_multi)} 个样本")
        
        # 验证数量一致
        assert len(aligned_single) == len(aligned_multi), \
            f"单进程 ({len(aligned_single)}) 和多进程 ({len(aligned_multi)}) 的样本数量不一致"
        
        # 验证字段一致性
        for i in range(len(aligned_single)):
            sample_single = aligned_single[i]
            sample_multi = aligned_multi[i]
            
            # 检查关键字段
            assert len(sample_single["_prompt"]) == len(sample_multi["_prompt"]), \
                f"样本 {i}: prompt 长度不一致"
            assert len(sample_single["_response"]) == len(sample_multi["_response"]), \
                f"样本 {i}: response 长度不一致"
            assert sample_single["_mask_history_sample"] == sample_multi["_mask_history_sample"], \
                f"样本 {i}: _mask_history_sample 不一致"
        
        print("\n✅ 单进程和多进程结果一致!")
        print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Mask History Sample 多进程兼容性测试套件")
    print("=" * 80)
    
    try:
        # 运行测试
        test_mask_history_sample_multiprocess()
        test_mask_history_sample_with_images()
        test_single_vs_multiprocess_consistency()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
