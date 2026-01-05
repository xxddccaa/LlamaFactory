#!/usr/bin/env python3
"""
修复 NVIDIA 容器镜像的 torch 版本约束，替换为 2.8.0+cu129
同时注释掉 torchaudio 的精确版本约束，避免版本冲突
"""

import os
import sys
import subprocess
import shutil

TARGET_VERSION = "2.8.0+cu129"

def fix_environment_variables():
    """修复环境变量"""
    print("=" * 60)
    print("1. 修复环境变量")
    print("=" * 60)
    
    vars_to_set = {
        'PYTORCH_VERSION': TARGET_VERSION,
        'PYTORCH_BUILD_VERSION': TARGET_VERSION,
        'PIP_CONSTRAINT': '/etc/pip/constraint.txt'  # 保持约束文件路径，但内容会被修改
    }
    
    # 在当前进程中设置
    for var, value in vars_to_set.items():
        old_value = os.environ.get(var, '未设置')
        os.environ[var] = value
        print(f"  设置: {var} = {value}")
        if old_value != '未设置':
            print(f"    (原值: {old_value})")
    
    # 生成 bash 脚本来持久化
    script_content = f"""#!/bin/bash
# 设置 torch 版本环境变量
export PYTORCH_VERSION={TARGET_VERSION}
export PYTORCH_BUILD_VERSION={TARGET_VERSION}
export PIP_CONSTRAINT=/etc/pip/constraint.txt

echo "已设置环境变量:"
echo "  PYTORCH_VERSION={TARGET_VERSION}"
echo "  PYTORCH_BUILD_VERSION={TARGET_VERSION}"
"""
    
    script_path = "/tmp/set_torch_version.sh"
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        print(f"\n✓ 已创建脚本: {script_path}")
        print("  运行: source /tmp/set_torch_version.sh")
    except Exception as e:
        print(f"✗ 创建脚本失败: {e}")

def fix_constraint_file():
    """修复约束文件"""
    print("\n" + "=" * 60)
    print("2. 修复约束文件")
    print("=" * 60)
    
    constraint_file = "/etc/pip/constraint.txt"
    
    if not os.path.exists(constraint_file):
        print(f"  {constraint_file} 不存在，创建新文件")
        try:
            # 创建新文件
            with open(constraint_file, 'w') as f:
                f.write(f"torch=={TARGET_VERSION}\n")
            print(f"  ✓ 已创建约束文件，设置 torch=={TARGET_VERSION}")
            return
        except PermissionError:
            print(f"  ✗ 权限不足，无法创建 {constraint_file}")
            print(f"  请使用 sudo 运行此脚本")
            return
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
            return
    
    try:
        # 备份
        backup_file = f"{constraint_file}.bak"
        shutil.copy2(constraint_file, backup_file)
        print(f"  ✓ 已备份到: {backup_file}")
        
        # 读取并修改
        with open(constraint_file, 'r') as f:
            content = f.read()
        
        import re
        modified = False
        
        lines = content.split('\n')
        new_lines = []
        # 正则表达式：精确匹配 torch 包（后面必须跟版本操作符，不能是其他字母）
        # 匹配格式：可选的缩进 + 可选的注释符号 + torch + 版本操作符
        torch_pattern = re.compile(r'^(\s*)(#\s*)?torch\s*(==|>=|<=|>|<|~=)')
        # 正则表达式：匹配 torchaudio 包（用于移除精确版本约束）
        torchaudio_pattern = re.compile(r'^(\s*)(#\s*)?torchaudio\s*(==|>=|<=|>|<|~=)\s*(.+)')
        
        for line in lines:
            original_line = line
            
            # 检查是否是 torch 约束行（精确匹配，不匹配 torchprofile、torchvision 等）
            match = torch_pattern.match(line)
            if match:
                # 替换为固定版本
                indent = match.group(1)  # 保留原有缩进
                comment_marker = match.group(2)  # 注释符号（如果有）
                
                if comment_marker:
                    # 如果被注释了，保持注释但更新版本
                    new_line = indent + f"# torch=={TARGET_VERSION}"
                else:
                    # 保持原有缩进
                    new_line = indent + f"torch=={TARGET_VERSION}"
                
                if original_line.rstrip() != new_line.rstrip():
                    print(f"  替换: {original_line.strip()} -> {new_line.strip()}")
                    modified = True
                new_lines.append(new_line)
            else:
                # 检查是否是 torchaudio 约束行（精确版本约束会导致冲突，需要注释掉）
                torchaudio_match = torchaudio_pattern.match(line)
                if torchaudio_match:
                    indent = torchaudio_match.group(1)
                    comment_marker = torchaudio_match.group(2)
                    operator = torchaudio_match.group(3)
                    version = torchaudio_match.group(4).strip()
                    
                    # 如果是精确版本约束（==），注释掉它，因为 torchaudio 需要精确匹配 torch 版本
                    # 但 llamafactory 只需要 torchaudio>=2.4.0，让 pip 自动选择合适版本
                    if operator == "==" and not comment_marker:
                        new_line = indent + f"# torchaudio=={version}  # 已注释：torchaudio 需要精确匹配 torch 版本，让 pip 自动选择"
                        if original_line.rstrip() != new_line.rstrip():
                            print(f"  注释: {original_line.strip()} -> {new_line.strip()}")
                            modified = True
                        new_lines.append(new_line)
                    else:
                        # 其他情况（>=, <= 等）或已注释的行，保持不变
                        new_lines.append(line)
                else:
                    new_lines.append(line)
        
        if modified:
            # 写入修改后的内容
            with open(constraint_file, 'w') as f:
                f.write('\n'.join(new_lines))
                if not content.endswith('\n'):
                    f.write('\n')
            print(f"  ✓ 已修改约束文件，所有 torch 约束已替换为 {TARGET_VERSION}")
        else:
            # 如果没有找到 torch 约束，添加一个
            with open(constraint_file, 'a') as f:
                f.write(f"\ntorch=={TARGET_VERSION}\n")
            print(f"  ✓ 未找到 torch 约束，已添加 torch=={TARGET_VERSION}")
            
    except PermissionError:
        print(f"  ✗ 权限不足，无法修改 {constraint_file}")
        print(f"  请使用 sudo 运行此脚本")
    except Exception as e:
        print(f"  ✗ 修改失败: {e}")

def fix_pip_config():
    """修复 pip 配置"""
    print("\n" + "=" * 60)
    print("3. 修复 pip 配置")
    print("=" * 60)
    
    try:
        # 移除约束配置
        result = subprocess.run(['pip', 'config', 'unset', 'global.constraint'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ 已移除 global.constraint")
        else:
            # 可能不存在，尝试其他方式
            result = subprocess.run(['pip', 'config', 'unset', 'constraint'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  ✓ 已移除 constraint")
            else:
                print("  ℹ 未找到约束配置")
    except Exception as e:
        print(f"  ✗ 修改失败: {e}")

def verify_fix():
    """验证修复"""
    print("\n" + "=" * 60)
    print("4. 验证修复")
    print("=" * 60)
    
    # 检查环境变量
    print("\n环境变量:")
    vars_to_check = {
        'PYTORCH_VERSION': TARGET_VERSION,
        'PYTORCH_BUILD_VERSION': TARGET_VERSION,
        'PIP_CONSTRAINT': '/etc/pip/constraint.txt'
    }
    for var, expected in vars_to_check.items():
        value = os.environ.get(var)
        if value == expected:
            print(f"  {var} = {value} ✓")
        elif value:
            print(f"  {var} = {value} (期望: {expected})")
        else:
            print(f"  {var} = 未设置 (期望: {expected})")
    
    # 检查实际 torch 版本
    print("\n实际 torch 版本:")
    try:
        import torch
        print(f"  Python 导入: {torch.__version__}")
        if TARGET_VERSION in torch.__version__:
            print(f"  ✓ 版本匹配目标 {TARGET_VERSION}")
        else:
            print(f"  ⚠ 版本不匹配，期望 {TARGET_VERSION}")
    except ImportError:
        print("  torch 未安装")
    except Exception as e:
        print(f"  检查失败: {e}")

def main():
    print("=" * 60)
    print("修复 torch 版本约束")
    print("=" * 60)
    print(f"\n目标版本: {TARGET_VERSION}")
    print("将替换 NVIDIA 容器镜像设置的版本约束")
    print("同时注释掉 torchaudio 的精确版本约束，避免版本冲突\n")
    
    if os.geteuid() != 0:
        print("⚠️  警告: 未以 root 运行，某些操作可能失败")
        print("  建议使用: sudo python fix_torch_version.py\n")
    
    fix_environment_variables()
    fix_constraint_file()
    fix_pip_config()
    verify_fix()
    
    print("\n" + "=" * 60)
    print("修复完成")
    print("=" * 60)
    print("\n注意:")
    print("  1. 环境变量的修改仅对当前 Python 进程有效")
    print("  2. 在 shell 中运行: source /tmp/set_torch_version.sh")
    print("  3. 或在 Dockerfile 中添加:")
    print(f"     ENV PYTORCH_VERSION={TARGET_VERSION}")
    print(f"     ENV PYTORCH_BUILD_VERSION={TARGET_VERSION}")
    print("\n现在可以尝试安装包:")
    print("  pip install flash-attn --no-build-isolation --no-deps")

if __name__ == "__main__":
    main()

