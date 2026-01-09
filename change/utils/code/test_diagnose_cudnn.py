#!/usr/bin/env python3
"""
诊断和修复 cuDNN 加载失败问题
用于解决 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED 错误
"""

import sys
import os
import subprocess
import glob
from pathlib import Path

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_ld_library_path():
    """检查 LD_LIBRARY_PATH 配置"""
    print_section("检查 1: LD_LIBRARY_PATH 环境变量")
    
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"当前 LD_LIBRARY_PATH: {ld_path if ld_path else '(未设置)'}")
    
    paths = ld_path.split(':') if ld_path else []
    cudnn_paths = []
    
    for path in paths:
        if path and os.path.exists(path):
            # 检查是否包含 cuDNN 相关库
            cudnn_libs = glob.glob(os.path.join(path, '*cudnn*'))
            if cudnn_libs:
                cudnn_paths.append(path)
                print(f"  ✓ 找到 cuDNN 库路径: {path}")
                for lib in cudnn_libs[:3]:  # 只显示前3个
                    print(f"    - {os.path.basename(lib)}")
    
    if not cudnn_paths:
        print("  ⚠ 在 LD_LIBRARY_PATH 中未找到 cuDNN 库")
    
    return paths, cudnn_paths

def find_cudnn_libraries():
    """查找系统中的 cuDNN 库文件"""
    print_section("检查 2: 查找 cuDNN 库文件")
    
    # 常见的 cuDNN 库路径
    common_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/nvidia/lib64',
        '/usr/local/nvidia/lib',
        '/opt/conda/lib',
        '/opt/conda/envs/*/lib',
    ]
    
    # 从 LD_LIBRARY_PATH 获取路径
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        common_paths.extend(ld_path.split(':'))
    
    found_libs = {}
    search_patterns = ['libcudnn*.so*', 'libcudnn*.so']
    
    for base_path in common_paths:
        # 处理通配符路径
        if '*' in base_path:
            expanded = glob.glob(base_path)
            search_paths = expanded if expanded else []
        else:
            search_paths = [base_path] if os.path.exists(base_path) else []
        
        for search_path in search_paths:
            if not os.path.isdir(search_path):
                continue
                
            for pattern in search_patterns:
                libs = glob.glob(os.path.join(search_path, pattern))
                if libs:
                    if search_path not in found_libs:
                        found_libs[search_path] = []
                    found_libs[search_path].extend(libs)
    
    if found_libs:
        print("✓ 找到 cuDNN 库文件:")
        for path, libs in found_libs.items():
            print(f"  路径: {path}")
            for lib in sorted(set(libs))[:5]:  # 只显示前5个
                lib_name = os.path.basename(lib)
                try:
                    # 尝试获取符号链接的真实路径
                    real_path = os.path.realpath(lib)
                    if real_path != lib:
                        print(f"    - {lib_name} -> {os.path.basename(real_path)}")
                    else:
                        print(f"    - {lib_name}")
                except:
                    print(f"    - {lib_name}")
    else:
        print("✗ 未找到 cuDNN 库文件")
    
    return found_libs

def check_pytorch_cudnn():
    """检查 PyTorch 的 cuDNN 配置"""
    print_section("检查 3: PyTorch cuDNN 配置")
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            
            # 检查 cuDNN 是否可用
            try:
                cudnn_enabled = torch.backends.cudnn.enabled
                cudnn_version = torch.backends.cudnn.version()
                print(f"  cuDNN 已启用: {cudnn_enabled}")
                print(f"  cuDNN 版本: {cudnn_version}")
                
                # 尝试测试 cuDNN 功能
                try:
                    device = torch.device('cuda:0')
                    x = torch.randn(1, 3, 224, 224, device=device)
                    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
                    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False):
                        y = conv(x)
                    print("  ✓ cuDNN 功能测试通过")
                    return True, cudnn_version
                except Exception as e:
                    print(f"  ✗ cuDNN 功能测试失败: {e}")
                    return False, cudnn_version
                    
            except Exception as e:
                print(f"  ✗ 无法获取 cuDNN 信息: {e}")
                return False, None
        else:
            print("  ⚠ CUDA 不可用，无法测试 cuDNN")
            return None, None
            
    except ImportError:
        print("✗ PyTorch 未安装")
        return None, None
    except Exception as e:
        print(f"✗ 检查 PyTorch 时出错: {e}")
        return None, None

def check_cudnn_installation():
    """检查 cuDNN 安装（通过 nvidia-smi 或其他方式）"""
    print_section("检查 4: cuDNN 系统安装")
    
    # 检查是否有 nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip().split('\n')[0]
            print(f"✓ NVIDIA 驱动版本: {driver_version}")
    except:
        print("  ⚠ 无法获取 NVIDIA 驱动信息")
    
    # 检查常见的 cuDNN 头文件
    cudnn_header_paths = [
        '/usr/local/cuda/include/cudnn.h',
        '/usr/include/cudnn.h',
        '/opt/conda/include/cudnn.h',
    ]
    
    found_header = False
    for header_path in cudnn_header_paths:
        if os.path.exists(header_path):
            print(f"✓ 找到 cuDNN 头文件: {header_path}")
            found_header = True
            # 尝试读取版本信息
            try:
                with open(header_path, 'r') as f:
                    content = f.read()
                    # 查找版本定义
                    if 'CUDNN_MAJOR' in content:
                        print("  cuDNN 头文件包含版本信息")
            except:
                pass
            break
    
    if not found_header:
        print("  ⚠ 未找到 cuDNN 头文件（可能不影响运行时）")

def generate_fix_suggestions(ld_paths, cudnn_libs, pytorch_ok):
    """生成修复建议"""
    print_section("修复建议")
    
    suggestions = []
    
    # 如果 PyTorch cuDNN 测试失败
    if pytorch_ok is False:
        suggestions.append({
            'priority': 'HIGH',
            'title': 'PyTorch cuDNN 测试失败',
            'solutions': [
                '1. 检查 LD_LIBRARY_PATH 是否包含 cuDNN 库路径',
                '2. 尝试设置环境变量: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH',
                '3. 如果使用容器，确保 cuDNN 库已正确挂载',
                '4. 尝试禁用 cuDNN: export PYTORCH_CUDNN_DISABLE=1 (不推荐，会影响性能)',
            ]
        })
    
    # 如果找不到 cuDNN 库
    if not cudnn_libs:
        suggestions.append({
            'priority': 'HIGH',
            'title': '未找到 cuDNN 库文件',
            'solutions': [
                '1. 检查容器镜像是否包含 cuDNN',
                '2. 检查 LD_LIBRARY_PATH 配置',
                '3. 如果使用 PyTorch wheel，cuDNN 应该已经包含在 PyTorch 包中',
                '4. 检查 nvidia-cudnn-cu12 包是否正确安装: pip list | grep cudnn',
            ]
        })
    
    # 如果找到库但 PyTorch 无法使用
    if cudnn_libs and pytorch_ok is False:
        suggestions.append({
            'priority': 'MEDIUM',
            'title': 'cuDNN 库存在但无法使用',
            'solutions': [
                '1. 检查 cuDNN 版本是否与 PyTorch 兼容',
                '2. 检查库文件权限: ls -l /path/to/libcudnn*.so*',
                '3. 尝试重新设置 LD_LIBRARY_PATH',
                '4. 检查是否有多个版本的 cuDNN 冲突',
            ]
        })
    
    # 通用建议
    if not suggestions:
        suggestions.append({
            'priority': 'LOW',
            'title': '环境检查通过',
            'solutions': [
                '如果训练时仍然出现 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED 错误:',
                '1. 检查是否是特定操作（如 conv3d）导致的问题',
                '2. 尝试使用 torch.backends.cudnn.benchmark = False',
                '3. 检查是否有内存不足或其他资源限制',
                '4. 查看完整的错误堆栈，定位具体是哪个模块加载失败',
            ]
        })
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n建议 {i} [{suggestion['priority']}]: {suggestion['title']}")
        for solution in suggestion['solutions']:
            print(f"  {solution}")

def generate_fix_script(ld_paths, cudnn_libs):
    """生成修复脚本"""
    print_section("生成修复脚本")
    
    fix_commands = []
    
    # 如果找到了 cuDNN 库但不在 LD_LIBRARY_PATH 中
    if cudnn_libs:
        for lib_path in cudnn_libs.keys():
            if lib_path not in ld_paths:
                fix_commands.append(f"export LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH")
    
    # 如果没找到库，尝试常见路径
    if not cudnn_libs:
        common_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/nvidia/lib64',
        ]
        for path in common_paths:
            if os.path.exists(path):
                fix_commands.append(f"export LD_LIBRARY_PATH={path}:$LD_LIBRARY_PATH")
    
    if fix_commands:
        print("建议在训练脚本中添加以下环境变量设置:")
        print("\n```bash")
        for cmd in fix_commands:
            print(cmd)
        print("```")
    else:
        print("未发现需要修复的配置问题")

def main():
    """主函数"""
    print("=" * 60)
    print("cuDNN 诊断工具")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 执行检查
    ld_paths, cudnn_paths = check_ld_library_path()
    cudnn_libs = find_cudnn_libraries()
    pytorch_ok, cudnn_version = check_pytorch_cudnn()
    check_cudnn_installation()
    
    # 生成建议
    generate_fix_suggestions(ld_paths, cudnn_libs, pytorch_ok)
    generate_fix_script(ld_paths, cudnn_libs)
    
    print_section("诊断完成")
    
    # 返回状态码
    if pytorch_ok is False:
        print("\n⚠ 发现严重问题: PyTorch cuDNN 测试失败")
        return 1
    elif pytorch_ok is None:
        print("\n⚠ 无法完成完整测试（CUDA 不可用）")
        return 2
    else:
        print("\n✓ 基本检查通过")
        return 0

if __name__ == '__main__':
    sys.exit(main())

