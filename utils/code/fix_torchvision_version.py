#!/usr/bin/env python3
"""
修复 NVIDIA 容器镜像的 torchvision 版本约束，将版本约束替换为指定版本
将 torchvision 的任何版本约束替换为 torchvision==0.23.0
"""

import os
import sys
import subprocess
import shutil
import re

TARGET_VERSION = "0.23.0"

def fix_constraint_file():
    """修复约束文件"""
    print("=" * 60)
    print("修复 torchvision 版本约束")
    print("=" * 60)
    print(f"\n目标版本: {TARGET_VERSION}")
    print("将替换开发版本（如 0.22.0a0+95f10a4e）为正式版本\n")
    
    constraint_file = "/etc/pip/constraint.txt"
    
    if not os.path.exists(constraint_file):
        print(f"  {constraint_file} 不存在，创建新文件")
        try:
            # 创建新文件
            with open(constraint_file, 'w') as f:
                f.write(f"torchvision=={TARGET_VERSION}\n")
            print(f"  ✓ 已创建约束文件，设置 torchvision=={TARGET_VERSION}")
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
        if os.path.exists(backup_file):
            # 如果已经有备份，创建带时间戳的备份
            import time
            timestamp = int(time.time())
            backup_file = f"{constraint_file}.bak.{timestamp}"
        shutil.copy2(constraint_file, backup_file)
        print(f"  ✓ 已备份到: {backup_file}")
        
        # 读取并修改
        with open(constraint_file, 'r') as f:
            content = f.read()
        
        modified = False
        
        lines = content.split('\n')
        new_lines = []
        # 正则表达式：精确匹配 torchvision 包（后面必须跟版本操作符，不能是其他字母）
        # 匹配格式：可选的缩进 + 可选的注释符号 + torchvision + 版本操作符 + 版本号
        torchvision_pattern = re.compile(r'^(\s*)(#\s*)?torchvision\s*(==|>=|<=|>|<|~=)\s*(.+)')
        
        for line in lines:
            original_line = line
            
            # 检查是否是 torchvision 约束行
            match = torchvision_pattern.match(line)
            if match:
                # 替换为固定版本
                indent = match.group(1)  # 保留原有缩进
                comment_marker = match.group(2)  # 注释符号（如果有）
                operator = match.group(3)  # 版本操作符
                old_version = match.group(4).strip()  # 旧版本号（去掉尾部空格）
                
                if comment_marker:
                    # 如果被注释了，保持注释但更新版本
                    new_line = indent + f"# torchvision=={TARGET_VERSION}"
                else:
                    # 保持原有缩进，但使用 == 操作符和固定版本
                    new_line = indent + f"torchvision=={TARGET_VERSION}"
                
                if original_line.rstrip() != new_line.rstrip():
                    print(f"  替换: {original_line.strip()}")
                    print(f"      -> {new_line.strip()}")
                    modified = True
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        if modified:
            # 写入修改后的内容
            with open(constraint_file, 'w') as f:
                f.write('\n'.join(new_lines))
                if not content.endswith('\n') and new_lines:
                    f.write('\n')
            print(f"\n  ✓ 已修改约束文件，所有 torchvision 约束已替换为 {TARGET_VERSION}")
        else:
            # 如果没有找到 torchvision 约束，添加一个
            with open(constraint_file, 'a') as f:
                f.write(f"\ntorchvision=={TARGET_VERSION}\n")
            print(f"\n  ✓ 未找到 torchvision 约束，已添加 torchvision=={TARGET_VERSION}")
            
    except PermissionError:
        print(f"  ✗ 权限不足，无法修改 {constraint_file}")
        print(f"  请使用 sudo 运行此脚本")
    except Exception as e:
        print(f"  ✗ 修改失败: {e}")
        import traceback
        traceback.print_exc()

def verify_fix():
    """验证修复"""
    print("\n" + "=" * 60)
    print("验证修复")
    print("=" * 60)
    
    constraint_file = "/etc/pip/constraint.txt"
    
    if not os.path.exists(constraint_file):
        print(f"  {constraint_file} 不存在")
        return
    
    try:
        with open(constraint_file, 'r') as f:
            content = f.read()
        
        # 查找 torchvision 约束
        torchvision_pattern = re.compile(r'^(\s*)(#\s*)?torchvision\s*(==|>=|<=|>|<|~=)\s*(.+)', re.MULTILINE)
        matches = torchvision_pattern.findall(content)
        
        if matches:
            print(f"\n约束文件中的 torchvision 约束:")
            for match in matches:
                indent, comment, operator, version = match
                status = "✓" if (operator == "==" and version.strip() == TARGET_VERSION) else "⚠"
                if comment:
                    print(f"  {status} # torchvision{operator}{version.strip()} (已注释)")
                else:
                    print(f"  {status} torchvision{operator}{version.strip()}")
        else:
            print(f"\n  未找到 torchvision 约束")
        
        # 检查实际 torchvision 版本
        print("\n实际 torchvision 版本:")
        try:
            import torchvision
            print(f"  Python 导入: {torchvision.__version__}")
            if TARGET_VERSION in torchvision.__version__ or torchvision.__version__.startswith(TARGET_VERSION):
                print(f"  ✓ 版本匹配目标 {TARGET_VERSION}")
            else:
                print(f"  ⚠ 版本不匹配，期望 {TARGET_VERSION}")
        except ImportError:
            print("  torchvision 未安装")
        except Exception as e:
            print(f"  检查失败: {e}")
            
    except Exception as e:
        print(f"  验证失败: {e}")

def main():
    print("=" * 60)
    print("修复 torchvision 版本约束")
    print("=" * 60)
    print(f"\n目标版本: {TARGET_VERSION}")
    print("将替换所有 torchvision 版本约束为 {TARGET_VERSION}\n")
    
    if os.geteuid() != 0:
        print("⚠️  警告: 未以 root 运行，某些操作可能失败")
        print("  建议使用: sudo python fix_torchvision_version.py\n")
    
    fix_constraint_file()
    verify_fix()
    
    print("\n" + "=" * 60)
    print("修复完成")
    print("=" * 60)
    print("\n现在可以尝试安装包:")
    print("  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129")

if __name__ == "__main__":
    main()

