#!/usr/bin/env python3
"""
测试 PyTorch 和 Triton 兼容性的脚本
用于验证 triton_key 导入是否正常
"""

import sys
import os
import shutil
import subprocess

def get_cuda_driver_version():
    """获取 CUDA 驱动版本"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Driver Version:' in line:
                    return line.split('Driver Version:')[1].strip().split()[0]
    except:
        pass
    
    return None

def check_cuda_library_paths():
    """检查 CUDA 库路径配置"""
    print("\n" + "-" * 60)
    print("CUDA 库路径检查")
    print("-" * 60)
    
    # 检查 LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"LD_LIBRARY_PATH: {ld_path if ld_path else '(未设置)'}")
    
    # 检查 LD_PRELOAD
    ld_preload = os.environ.get('LD_PRELOAD', '')
    print(f"LD_PRELOAD: {ld_preload if ld_preload else '(未设置)'}")
    
    # 检查常见的 CUDA 库路径
    common_paths = [
        '/usr/local/nvidia/lib64',
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu',
    ]
    
    libcuda_found = False
    for path in common_paths:
        libcuda_path = os.path.join(path, 'libcuda.so')
        if os.path.exists(libcuda_path):
            print(f"✓ 找到 libcuda.so: {libcuda_path}")
            libcuda_found = True
            if '/usr/local/nvidia/lib64' in path and '/usr/local/nvidia/lib64' not in ld_path:
                print(f"  ⚠ 建议: 将 {path} 添加到 LD_LIBRARY_PATH")
            break
    
    if not libcuda_found:
        print("⚠ 未找到 libcuda.so，这可能导致 cuModuleGetFunction 错误")
    
    return libcuda_found

def test_imports():
    """测试基本导入"""
    print("=" * 60)
    print("测试 1: 基本导入")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            
            # 获取 CUDA 驱动版本
            driver_version = get_cuda_driver_version()
            if driver_version:
                print(f"  CUDA 驱动版本: {driver_version}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import triton
        print(f"✓ Triton 版本: {triton.__version__}")
    except ImportError as e:
        print(f"✗ Triton 导入失败: {e}")
        return False
    
    # 检查 CUDA 库路径
    check_cuda_library_paths()
    
    return True

def test_triton_key():
    """测试 triton_key 导入（关键测试）"""
    print("\n" + "=" * 60)
    print("测试 2: triton_key 导入（关键测试）")
    print("=" * 60)
    
    try:
        from triton.compiler.compiler import triton_key
        print("✓ triton_key 导入成功！")
        print(f"  triton_key 类型: {type(triton_key)}")
        return True
    except ImportError as e:
        print(f"✗ triton_key 导入失败: {e}")
        print("\n这是导致训练失败的关键错误！")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_torch_inductor():
    """测试 PyTorch Inductor（使用 triton_key 的地方）"""
    print("\n" + "=" * 60)
    print("测试 3: PyTorch Inductor 缓存系统")
    print("=" * 60)
    
    try:
        import torch
        from torch._inductor.codecache import CacheBase
        
        # 尝试获取系统信息（这会调用 triton_key）
        system_info = CacheBase.get_system()
        print("✓ PyTorch Inductor 缓存系统正常")
        print(f"  系统信息键: {list(system_info.keys())[:3]}...")  # 只显示前3个键
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        print("  这可能是 triton_key 导入问题导致的")
        return False

def get_triton_cache_dir():
    """获取 Triton 缓存目录"""
    cache_dir = os.environ.get('TRITON_CACHE_DIR')
    if cache_dir:
        return cache_dir
    
    # 默认缓存目录
    home = os.path.expanduser('~')
    return os.path.join(home, '.triton', 'cache')

def clear_triton_cache():
    """清除 Triton 缓存"""
    cache_dir = get_triton_cache_dir()
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"✓ 已清除 Triton 缓存: {cache_dir}")
            return True
        except Exception as e:
            print(f"⚠ 清除缓存失败: {e}")
            return False
    else:
        print(f"ℹ Triton 缓存目录不存在: {cache_dir}")
        return True

def test_basic_torch_compile():
    """测试基本的 torch.compile 功能"""
    print("\n" + "=" * 60)
    print("测试 4: torch.compile 基本功能")
    print("=" * 60)
    
    try:
        import torch
        
        @torch.compile
        def simple_add(x, y):
            return x + y
        
        # 创建测试张量
        x = torch.randn(10, 10, device='cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randn(10, 10, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 执行（第一次会触发编译）
        result = simple_add(x, y)
        print("✓ torch.compile 测试通过")
        print(f"  结果形状: {result.shape}")
        return True
    except ImportError as e:
        error_msg = str(e)
        if 'undefined symbol' in error_msg or 'cuModuleGetFunction' in error_msg:
            print(f"✗ torch.compile 测试失败: {e}")
            print("\n" + "=" * 60)
            print("诊断信息")
            print("=" * 60)
            print("这个错误通常由以下原因引起：")
            print("1. CUDA 驱动库路径配置不正确，找不到 libcuda.so")
            print("2. Triton 编译的代码无法链接到正确的 CUDA Driver API")
            print("3. LD_LIBRARY_PATH 或 LD_PRELOAD 未正确设置")
            print("\n建议解决方案：")
            print("1. 设置 CUDA 库路径（推荐）：")
            print("   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH")
            print("   export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so")
            print("\n2. 清除 Triton 缓存后重试：")
            cache_dir = get_triton_cache_dir()
            print(f"   rm -rf {cache_dir}")
            print("\n3. 检查 CUDA 驱动版本：")
            print("   nvidia-smi")
            print("\n4. 如果问题持续，可能需要：")
            print("   - 检查 /usr/local/nvidia/lib64/libcuda.so 是否存在")
            print("   - 尝试更新 Triton 版本（如 3.4.0 或 3.4.1）")
            print("   - 检查 CUDA 驱动版本是否与 PyTorch 兼容")
            return False
        else:
            print(f"✗ torch.compile 测试失败: {e}")
            return False
    except Exception as e:
        print(f"✗ torch.compile 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("PyTorch 和 Triton 兼容性测试")
    print("=" * 60 + "\n")
    
    # 显示 Triton 缓存信息
    cache_dir = get_triton_cache_dir()
    print(f"Triton 缓存目录: {cache_dir}")
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        )
        print(f"  缓存大小: {cache_size / (1024*1024):.2f} MB")
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("基本导入", test_imports()))
    results.append(("triton_key 导入", test_triton_key()))
    results.append(("PyTorch Inductor", test_torch_inductor()))
    
    # 只在有 GPU 时测试 compile
    compile_failed = False
    try:
        import torch
        if torch.cuda.is_available():
            compile_result = test_basic_torch_compile()
            results.append(("torch.compile", compile_result))
            if not compile_result:
                compile_failed = True
        else:
            print("\n" + "=" * 60)
            print("测试 4: torch.compile 基本功能")
            print("=" * 60)
            print("⚠ 跳过：未检测到 GPU")
    except:
        pass
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！PyTorch 和 Triton 兼容性正常。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        
        # 如果 compile 失败，提供快速修复建议
        if compile_failed:
            print("\n" + "=" * 60)
            print("快速修复建议")
            print("=" * 60)
            print("如果 torch.compile 失败（cuModuleGetFunction 错误），按以下步骤操作：")
            print("\n步骤 1: 设置 CUDA 库路径")
            print("  export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH")
            print("  export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so")
            print("\n步骤 2: 清除 Triton 缓存")
            print(f"  rm -rf {cache_dir}")
            print("\n步骤 3: 重新运行测试")
            print("  python '/mnt/s3fs/swifttrain/utils/code/test_triton_compat.py'")
            print("\n或者一键执行：")
            print("  export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH && \\")
            print("  export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so && \\")
            print(f"  rm -rf {cache_dir} && \\")
            print("  python '/mnt/s3fs/swifttrain/utils/code/test_triton_compat.py'")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())

