#!/usr/bin/env python3
"""
测试 dataset 路径中的内联配置语法
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llamafactory.data.parser import _parse_dataset_path_with_config, get_dataset_list

def test_parse_dataset_path_with_config():
    """测试解析 dataset 路径配置"""
    print("=" * 80)
    print("测试 _parse_dataset_path_with_config 函数")
    print("=" * 80)
    
    test_cases = [
        # (input, expected_path, expected_config)
        (
            "/mnt/data.json",
            "/mnt/data.json",
            {}
        ),
        (
            "/mnt/data.json[media_dir=/mnt/images]",
            "/mnt/data.json",
            {"media_dir": "/mnt/images"}
        ),
        (
            "/mnt/data.json[media_dir=/mnt/images,formatting=alpaca]",
            "/mnt/data.json",
            {"media_dir": "/mnt/images", "formatting": "alpaca"}
        ),
        (
            "s3://bucket/data.json[media_dir=s3://bucket/images]",
            "s3://bucket/data.json",
            {"media_dir": "s3://bucket/images"}
        ),
        (
            "/mnt/data.json[media_dir=/mnt/images,user_tag=user,assistant_tag=assistant]",
            "/mnt/data.json",
            {"media_dir": "/mnt/images", "user_tag": "user", "assistant_tag": "assistant"}
        ),
        (
            "/mnt/data.json[mask_history_sample=true,max_human_steps=2]",
            "/mnt/data.json",
            {"mask_history_sample": True, "max_human_steps": 2}
        ),
    ]
    
    for i, (input_str, expected_path, expected_config) in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"  输入: {input_str}")
        
        path, config = _parse_dataset_path_with_config(input_str)
        
        print(f"  解析结果:")
        print(f"    路径: {path}")
        print(f"    配置: {config}")
        
        assert path == expected_path, f"路径不匹配: 期望 {expected_path}, 得到 {path}"
        assert config == expected_config, f"配置不匹配: 期望 {expected_config}, 得到 {config}"
        print(f"  ✓ 通过")
    
    print("\n" + "=" * 80)
    print("所有解析测试通过！")
    print("=" * 80)


def test_get_dataset_list_with_inline_config():
    """测试 get_dataset_list 支持内联配置"""
    print("\n" + "=" * 80)
    print("测试 get_dataset_list 内联配置功能")
    print("=" * 80)
    
    # 测试1: 基本的 media_dir 配置
    print("\n测试1: 带 media_dir 的文件路径")
    dataset_names = ["/tmp/test_data.json[media_dir=/tmp/test_images]"]
    try:
        dataset_list = get_dataset_list(dataset_names, dataset_dir="data")
        assert len(dataset_list) == 1
        assert dataset_list[0].dataset_name == "/tmp/test_data.json"
        assert dataset_list[0].media_dir == "/tmp/test_images"
        assert dataset_list[0].formatting == "sharegpt"
        print(f"  ✓ 数据集名称: {dataset_list[0].dataset_name}")
        print(f"  ✓ media_dir: {dataset_list[0].media_dir}")
        print(f"  ✓ formatting: {dataset_list[0].formatting}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        raise
    
    # 测试2: 多个配置项
    print("\n测试2: 多个配置项")
    dataset_names = ["/tmp/test.json[media_dir=/tmp/img,formatting=alpaca,user_tag=user]"]
    try:
        dataset_list = get_dataset_list(dataset_names, dataset_dir="data")
        assert len(dataset_list) == 1
        assert dataset_list[0].media_dir == "/tmp/img"
        assert dataset_list[0].formatting == "alpaca"
        assert dataset_list[0].user_tag == "user"
        print(f"  ✓ media_dir: {dataset_list[0].media_dir}")
        print(f"  ✓ formatting: {dataset_list[0].formatting}")
        print(f"  ✓ user_tag: {dataset_list[0].user_tag}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        raise
    
    # 测试3: S3 路径
    print("\n测试3: S3 路径配置")
    dataset_names = ["s3://bucket/data.json[media_dir=s3://bucket/images]"]
    try:
        dataset_list = get_dataset_list(dataset_names, dataset_dir="data")
        assert len(dataset_list) == 1
        assert dataset_list[0].dataset_name == "s3://bucket/data.json"
        assert dataset_list[0].media_dir == "s3://bucket/images"
        print(f"  ✓ 数据集名称: {dataset_list[0].dataset_name}")
        print(f"  ✓ media_dir: {dataset_list[0].media_dir}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        raise
    
    # 测试4: mask_history_sample 配置
    print("\n测试4: mask_history_sample 配置")
    dataset_names = ["/tmp/test.json[mask_history_sample=true,max_human_steps=2]"]
    try:
        dataset_list = get_dataset_list(dataset_names, dataset_dir="data")
        assert len(dataset_list) == 1
        assert dataset_list[0].mask_history_sample == True
        assert dataset_list[0].max_human_steps == 2
        print(f"  ✓ mask_history_sample: {dataset_list[0].mask_history_sample}")
        print(f"  ✓ max_human_steps: {dataset_list[0].max_human_steps}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        raise
    
    # 测试5: 验证错误（只设置 mask_history_sample 不设置 max_human_steps）
    print("\n测试5: 验证错误检测（只设置 mask_history_sample）")
    dataset_names = ["/tmp/test.json[mask_history_sample=true]"]
    try:
        dataset_list = get_dataset_list(dataset_names, dataset_dir="data")
        print(f"  ✗ 应该抛出 ValueError")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        print(f"  ✓ 正确抛出 ValueError: {str(e)[:80]}...")
    
    print("\n" + "=" * 80)
    print("所有 get_dataset_list 测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_parse_dataset_path_with_config()
    test_get_dataset_list_with_inline_config()
    print("\n" + "=" * 80)
    print("🎉 所有测试通过！")
    print("=" * 80)

