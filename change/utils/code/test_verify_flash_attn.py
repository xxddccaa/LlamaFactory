#!/usr/bin/env python3
"""
验证 Flash Attention 是否正确安装和可用
用于检查 flash-attn 是否可以正常导入和使用
"""

import sys
import subprocess

def check_flash_attn_installation():
    """检查 Flash Attention 安装"""
    print("=" * 60)
    print("测试 1: Flash Attention 安装检查")
    print("=" * 60)
    
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', '未知')
        print(f"✓ flash_attn 版本: {version}")
        return True, version
    except ImportError as e:
        print(f"✗ flash_attn 未安装")
        print(f"  错误信息: {e}")
        return False, None
    except Exception as e:
        print(f"✗ flash_attn 导入失败")
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

def test_flash_attn_critical_modules():
    """测试 Flash Attention 关键模块（这是会出错的地方）"""
    print("\n" + "=" * 60)
    print("测试 3: Flash Attention 关键模块导入（关键测试）")
    print("=" * 60)
    
    try:
        # 测试 flash_attn 主模块
        import flash_attn
        print("✓ flash_attn 主模块导入成功")
        
        # 测试 flash_attn.flash_attn_interface（C++ 扩展，最容易出错）
        try:
            import flash_attn.flash_attn_interface
            print("✓ flash_attn.flash_attn_interface 导入成功")
        except ImportError as e:
            print(f"⚠ flash_attn.flash_attn_interface 导入失败: {e}")
            print("  这可能是编译问题，但可能不影响使用")
        
        # 测试 flash_attn 的关键函数
        try:
            from flash_attn import flash_attn_func
            print("✓ flash_attn_func 导入成功")
        except ImportError as e:
            print(f"✗ flash_attn_func 导入失败: {e}")
            print("\n这是导致训练失败的关键错误！")
            return False
        
        # 测试 flash_attn 的其他关键组件
        try:
            from flash_attn import flash_attn_varlen_func
            print("✓ flash_attn_varlen_func 导入成功")
        except ImportError:
            print("⚠ flash_attn_varlen_func 不可用（可选）")
        
        try:
            from flash_attn import flash_attn_with_kvcache
            print("✓ flash_attn_with_kvcache 导入成功")
        except ImportError:
            print("⚠ flash_attn_with_kvcache 不可用（可选）")
        
        print("\n✓ Flash Attention 关键模块可以正常使用！")
        return True
        
    except ImportError as e:
        print(f"✗ Flash Attention 关键模块导入失败: {e}")
        print("\n这是导致训练失败的关键错误！")
        print("\n建议解决方案：")
        print("1. 重新安装 flash-attn:")
        print("   pip uninstall flash-attn -y")
        print("   pip install flash-attn --no-build-isolation")
        print("\n2. 或者从源码编译（如果预编译版本不兼容）:")
        print("   pip install flash-attn --no-build-isolation")
        print("\n3. 检查 CUDA 和 PyTorch 版本兼容性")
        print("   Flash Attention 需要 CUDA 11.6+ 和兼容的 PyTorch 版本")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Flash Attention 关键模块导入失败: {e}")
        
        # 诊断常见错误
        if 'undefined symbol' in error_msg:
            print("\n" + "-" * 60)
            print("符号错误诊断:")
            print("-" * 60)
            print("检测到符号未定义错误，可能的原因：")
            print("1. Flash Attention 与当前 PyTorch 版本不兼容")
            print("2. CUDA 库路径配置不正确")
            print("3. Flash Attention 需要重新编译")
            print("\n建议：")
            print("1. 设置 CUDA 库路径:")
            print("   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH")
            print("   export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so")
            print("\n2. 重新安装 flash-attn")
            print("   pip uninstall flash-attn -y")
            print("   pip install flash-attn --no-build-isolation")
        
        return False

def test_flash_attn_functionality():
    """测试 Flash Attention 基本功能"""
    print("\n" + "=" * 60)
    print("测试 4: Flash Attention 基本功能测试")
    print("=" * 60)
    
    try:
        import torch
        from flash_attn import flash_attn_func
        
        if not torch.cuda.is_available():
            print("⚠ 跳过功能测试：未检测到 GPU")
            return True
        
        # 创建测试数据
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        device = 'cuda'
        dtype = torch.float16
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        
        # 测试 flash_attn_func
        try:
            output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
            print(f"✓ flash_attn_func 执行成功")
            print(f"  输出形状: {output.shape}")
            
            # 测试 causal attention
            output_causal = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
            print(f"✓ causal attention 执行成功")
            
            print("✓ Flash Attention 基本功能正常")
            return True
        except Exception as e:
            print(f"✗ flash_attn_func 执行失败: {e}")
            print("  这可能是 CUDA 编译问题或版本不兼容")
            return False
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ Flash Attention 功能测试失败: {e}")
        return False

def test_flash_attn_ops():
    """测试 Flash Attention 的其他操作"""
    print("\n" + "=" * 60)
    print("测试 5: Flash Attention 其他操作")
    print("=" * 60)
    
    try:
        # 测试 flash_attn.ops
        try:
            from flash_attn.ops import fused_dense
            print("✓ flash_attn.ops.fused_dense 可用")
        except ImportError:
            print("⚠ flash_attn.ops.fused_dense 不可用（可选）")
        
        try:
            from flash_attn.ops import rms_norm
            print("✓ flash_attn.ops.rms_norm 可用")
        except ImportError:
            print("⚠ flash_attn.ops.rms_norm 不可用（可选）")
        
        try:
            from flash_attn.ops import rotary
            print("✓ flash_attn.ops.rotary 可用")
        except ImportError:
            print("⚠ flash_attn.ops.rotary 不可用（可选）")
        
        return True
    except Exception as e:
        print(f"⚠ 其他操作检查失败: {e}")
        return True  # 这些是可选的，不影响主要功能

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Flash Attention 安装和可用性测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    flash_attn_installed, flash_attn_version = check_flash_attn_installation()
    if not flash_attn_installed:
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print("✗ Flash Attention 未安装，请先安装 flash-attn")
        print("\n安装命令:")
        print("  pip install flash-attn --no-build-isolation")
        print("\n注意: Flash Attention 需要 CUDA 11.6+ 和兼容的 PyTorch 版本")
        return 1
    
    results.append(("Flash Attention 安装", flash_attn_installed))
    results.append(("PyTorch 环境", check_pytorch()))
    results.append(("Flash Attention 关键模块", test_flash_attn_critical_modules()))
    
    # 只在关键模块通过时测试基本功能
    if results[-1][1]:  # 如果关键模块测试通过
        results.append(("Flash Attention 基本功能", test_flash_attn_functionality()))
        results.append(("Flash Attention 其他操作", test_flash_attn_ops()))
    
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
        print("\n🎉 所有测试通过！Flash Attention 可以正常使用。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        
        # 如果关键模块失败，提供快速修复建议
        critical_failed = len(results) >= 3 and not results[2][1]
        if critical_failed:
            print("\n" + "=" * 60)
            print("快速修复建议")
            print("=" * 60)
            print("如果 Flash Attention 关键模块导入失败，可以尝试：")
            print("\n1. 重新安装 flash-attn:")
            print("   pip uninstall flash-attn -y")
            print("   pip install flash-attn --no-build-isolation")
            print("\n2. 设置 CUDA 库路径（如果遇到符号错误）:")
            print("   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH")
            print("   export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so")
            print("\n3. 检查 PyTorch 和 CUDA 版本兼容性")
            print("   Flash Attention 需要 CUDA 11.6+ 和兼容的 PyTorch 版本")
            print("\n4. 如果预编译版本不兼容，可能需要从源码编译")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())

