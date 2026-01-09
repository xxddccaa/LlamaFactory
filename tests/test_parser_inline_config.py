#!/usr/bin/env python3
"""
测试 dataset 路径解析功能（独立测试，不需要完整环境）
"""

def parse_dataset_path_with_config(path_with_config: str) -> tuple:
    """Parse dataset path with optional configuration in brackets."""
    # Check if there's a config section in brackets
    if '[' not in path_with_config:
        return path_with_config, {}
    
    # Find the last '[' to handle paths that might contain '['
    bracket_start = path_with_config.rfind('[')
    if bracket_start == -1 or not path_with_config.endswith(']'):
        return path_with_config, {}
    
    file_path = path_with_config[:bracket_start]
    config_str = path_with_config[bracket_start+1:-1]  # Remove [ and ]
    
    if not config_str.strip():
        return file_path, {}
    
    # Parse config string: "key1=value1,key2=value2"
    config_dict = {}
    parts = []
    current_part = []
    in_value = False
    
    for char in config_str:
        if char == '=' and not in_value:
            in_value = True
            current_part.append(char)
        elif char == ',' and in_value:
            parts.append(''.join(current_part))
            current_part = []
            in_value = False
        else:
            current_part.append(char)
    
    if current_part:
        parts.append(''.join(current_part))
    
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Convert boolean strings
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        # Convert numeric strings
        elif value.isdigit():
            value = int(value)
        
        config_dict[key] = value
    
    return file_path, config_dict


def test_parse():
    """测试解析功能"""
    print("=" * 80)
    print("测试 Dataset 路径内联配置解析")
    print("=" * 80)
    
    test_cases = [
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
        (
            "/mnt/data.json[mask_history_sample=false,max_human_steps=10]",
            "/mnt/data.json",
            {"mask_history_sample": False, "max_human_steps": 10}
        ),
    ]
    
    all_passed = True
    for i, (input_str, expected_path, expected_config) in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"  输入: {input_str}")
        
        path, config = parse_dataset_path_with_config(input_str)
        
        print(f"  解析结果:")
        print(f"    路径: {path}")
        print(f"    配置: {config}")
        
        if path != expected_path:
            print(f"  ✗ 路径不匹配: 期望 {expected_path}, 得到 {path}")
            all_passed = False
        elif config != expected_config:
            print(f"  ✗ 配置不匹配: 期望 {expected_config}, 得到 {config}")
            all_passed = False
        else:
            print(f"  ✓ 通过")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_parse()
    exit(0 if success else 1)

