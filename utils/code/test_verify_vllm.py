#!/usr/bin/env python3
"""
验证 vLLM 是否正确安装和可用
用于检查 vLLM 是否可以正常导入和使用
"""

import sys
import subprocess

def check_vllm_installation():
    """检查 vLLM 安装"""
    print("=" * 60)
    print("测试 1: vLLM 安装检查")
    print("=" * 60)
    
    try:
        import vllm
        version = getattr(vllm, '__version__', '未知')
        print(f"✓ vLLM 版本: {version}")
        return True, version
    except ImportError as e:
        print(f"✗ vLLM 未安装")
        print(f"  错误信息: {e}")
        return False, None
    except Exception as e:
        print(f"✗ vLLM 导入失败")
        print(f"  错误信息: {e}")
        return False, None

def check_pytorch():
    """检查 PyTorch 环境"""
    print("\n" + "=" * 60)
    print("测试 2: PyTorch 环境检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            
            # 获取 CUDA 驱动版本
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    driver_version = result.stdout.strip().split('\n')[0]
                    print(f"  CUDA 驱动版本: {driver_version}")
            except:
                pass
        
        return True
    except ImportError as e:
        print(f"✗ PyTorch 未安装: {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch 检查失败: {e}")
        return False

def test_vllm_critical_modules():
    """测试 vLLM 关键模块（这是会出错的地方）"""
    print("\n" + "=" * 60)
    print("测试 3: vLLM 关键模块导入（关键测试）")
    print("=" * 60)
    
    try:
        # 测试 vllm.platforms
        from vllm.platforms import current_platform
        print("✓ vllm.platforms 导入成功")
        
        # 测试 vllm._C（C++ 扩展，最容易出错）
        import vllm._C
        print("✓ vllm._C 导入成功")
        
        print("\n✓ vLLM 可以正常使用！")
        return True
    except ImportError as e:
        print(f"✗ vLLM 关键模块导入失败: {e}")
        print("\n这是导致训练失败的关键错误！")
        print("\n建议解决方案：")
        print("1. 重新安装 vLLM:")
        print("   pip uninstall vllm -y")
        print("   pip install vllm==0.11.0 --no-deps")
        print("\n2. 或者安装完整版本:")
        print("   pip install vllm==0.11.0")
        print("\n3. 检查 CUDA 和 PyTorch 版本兼容性")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"✗ vLLM 关键模块导入失败: {e}")
        
        # 诊断常见错误
        if 'undefined symbol' in error_msg:
            print("\n" + "-" * 60)
            print("符号错误诊断:")
            print("-" * 60)
            print("检测到符号未定义错误，可能的原因：")
            print("1. vLLM 与当前 PyTorch 版本不兼容")
            print("2. CUDA 库路径配置不正确")
            print("3. vLLM 需要重新编译")
            print("\n建议：")
            print("1. 设置 CUDA 库路径:")
            print("   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH")
            print("   export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so")
            print("\n2. 重新安装 vLLM")
        
        return False

def test_vllm_basic_functionality():
    """测试 vLLM 基本功能"""
    print("\n" + "=" * 60)
    print("测试 4: vLLM 基本功能测试")
    print("=" * 60)
    
    try:
        import vllm
        from vllm import LLM
        
        # 检查 LLM 类是否可用
        print("✓ vLLM.LLM 类可用")
        
        # 检查其他关键组件
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            print("✓ vLLM 引擎参数类可用")
        except:
            pass
        
        try:
            from vllm.worker.worker import Worker
            print("✓ vLLM Worker 类可用")
        except:
            pass
        
        print("✓ vLLM 基本功能正常")
        return True
    except Exception as e:
        print(f"✗ vLLM 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("vLLM 安装和可用性测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    vllm_installed, vllm_version = check_vllm_installation()
    if not vllm_installed:
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print("✗ vLLM 未安装，请先安装 vLLM")
        print("\n安装命令:")
        print("  pip install vllm==0.11.0")
        return 1
    
    results.append(("vLLM 安装", vllm_installed))
    results.append(("PyTorch 环境", check_pytorch()))
    results.append(("vLLM 关键模块", test_vllm_critical_modules()))
    
    # 只在关键模块通过时测试基本功能
    if results[-1][1]:  # 如果关键模块测试通过
        results.append(("vLLM 基本功能", test_vllm_basic_functionality()))
    
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
        print("\n🎉 所有测试通过！vLLM 可以正常使用。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        
        # 如果关键模块失败，提供快速修复建议
        if not results[-1][1] if results else False:
            print("\n" + "=" * 60)
            print("快速修复建议")
            print("=" * 60)
            print("如果 vLLM 关键模块导入失败，可以尝试：")
            print("\n1. 重新安装 vLLM:")
            print("   pip uninstall vllm -y")
            print("   pip install vllm==0.11.0 --no-deps")
            print("\n2. 设置 CUDA 库路径（如果遇到符号错误）:")
            print("   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH")
            print("   export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so")
            print("\n3. 检查 PyTorch 和 CUDA 版本兼容性")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())

