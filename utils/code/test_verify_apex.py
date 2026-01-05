#!/usr/bin/env python3
"""
验证 APEX 是否正确安装，特别是 CUDA 扩展模块
"""

import sys
import importlib
import os

def check_pytorch_version():
    """检查 PyTorch 版本信息"""
    try:
        import torch
        print(f"  PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  CUDA 可用: True")
        return torch.__version__
    except ImportError:
        print("  ⚠ PyTorch 未安装")
        return None

def diagnose_symbol_error(error_msg):
    """诊断符号未定义错误"""
    if 'undefined symbol' in str(error_msg):
        symbol = str(error_msg)
        print("\n" + "-" * 60)
        print("符号错误诊断:")
        print("-" * 60)
        
        # 检查是否是 PyTorch 相关符号
        if 'c10' in symbol or 'SetDevice' in symbol or '_ZN3c10' in symbol:
            print("⚠ 检测到 PyTorch 符号未定义错误")
            print("  这通常表示 APEX 与当前 PyTorch 版本不兼容")
            print("\n可能的原因:")
            print("1. APEX 是用不同版本的 PyTorch 编译的")
            print("2. PyTorch 版本更新后，APEX 需要重新编译")
            print("3. APEX wheel 文件与当前环境不匹配")
            print("\n建议解决方案:")
            print("1. 重新编译 APEX（推荐）:")
            print("   git clone https://github.com/NVIDIA/apex.git")
            print("   cd apex")
            print("   pip install -v --no-cache-dir --no-build-isolation \\")
            print("     --global-option=\"--cpp_ext\" \\")
            print("     --global-option=\"--cuda_ext\" .")
            print("\n2. 或者使用与当前 PyTorch 版本匹配的 APEX wheel")
            print("\n3. 如果只是 amp_C 失败，可以忽略（AMP 功能可能不可用）")
            print("   其他关键模块（如 fused_weight_gradient_mlp_cuda）正常即可")

def check_module(module_name, description):
    """检查模块是否可以导入"""
    try:
        mod = importlib.import_module(module_name)
        print(f"✓ {description}: {module_name} - 已安装")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} - 未安装或导入失败")
        print(f"  错误信息: {e}")
        diagnose_symbol_error(e)
        return False
    except Exception as e:
        print(f"✗ {description}: {module_name} - 导入时出错")
        print(f"  错误信息: {e}")
        diagnose_symbol_error(e)
        return False

def main():
    print("=" * 60)
    print("APEX 安装验证脚本")
    print("=" * 60)
    print()
    
    # 检查 PyTorch 版本
    print("0. 环境信息:")
    print("-" * 60)
    pytorch_version = check_pytorch_version()
    print()
    
    # 检查基础模块
    print("1. 检查基础 APEX 模块:")
    print("-" * 60)
    apex_basic = check_module("apex", "APEX 基础包")
    if apex_basic:
        try:
            import apex
            print(f"  APEX 版本: {getattr(apex, '__version__', '未知')}")
        except:
            pass
    print()
    
    if not apex_basic:
        print("❌ APEX 基础包未安装，请先安装 APEX")
        return 1
    
    # 检查 C++ 扩展
    print("2. 检查 C++ 扩展 (APEX_CPP_EXT):")
    print("-" * 60)
    apex_c = check_module("apex_C", "APEX C++ 扩展")
    print()
    
    # 检查 CUDA 扩展
    print("3. 检查 CUDA 扩展 (APEX_CUDA_EXT):")
    print("-" * 60)
    cuda_extensions = [
        ("amp_C", "AMP CUDA 扩展"),
        ("syncbn", "SyncBatchNorm CUDA 扩展"),
        ("fused_layer_norm_cuda", "Fused LayerNorm CUDA 扩展"),
        ("mlp_cuda", "MLP CUDA 扩展"),
        ("fused_weight_gradient_mlp_cuda", "Fused Weight Gradient MLP CUDA 扩展 (gradient_accumulation_fusion 需要)"),
        ("scaled_upper_triang_masked_softmax_cuda", "Scaled Upper Triangular Masked Softmax CUDA 扩展"),
        ("generic_scaled_masked_softmax_cuda", "Generic Scaled Masked Softmax CUDA 扩展"),
        ("scaled_masked_softmax_cuda", "Scaled Masked Softmax CUDA 扩展"),
    ]
    
    cuda_results = {}
    for module_name, description in cuda_extensions:
        result = check_module(module_name, description)
        cuda_results[module_name] = result
    
    print()
    
    # 检查关键模块（gradient_accumulation_fusion 需要的）
    print("4. 关键模块检查 (gradient_accumulation_fusion 必需):")
    print("-" * 60)
    critical_module = "fused_weight_gradient_mlp_cuda"
    if cuda_results.get(critical_module, False):
        print(f"✓ {critical_module} 已正确安装")
        print("  → gradient_accumulation_fusion 功能可用")
    else:
        print(f"✗ {critical_module} 未安装或导入失败")
        print("  → gradient_accumulation_fusion 功能不可用")
        print("  → 解决方案:")
        print("     1. 重新安装 APEX，确保设置 APEX_CUDA_EXT=1")
        print("     2. 或者在训练脚本中添加 --no_gradient_accumulation_fusion true")
    print()
    
    # 总结
    print("=" * 60)
    print("验证总结:")
    print("=" * 60)
    
    all_cuda_ok = all(cuda_results.values())
    critical_ok = cuda_results.get(critical_module, False)
    
    if apex_basic and apex_c and all_cuda_ok:
        print("✓ APEX 完整安装成功，所有扩展模块可用")
        return 0
    elif apex_basic and critical_ok:
        print("⚠ APEX 部分安装成功，关键模块可用")
        print("  部分 CUDA 扩展可能缺失，但不影响 gradient_accumulation_fusion")
        
        # 检查是否有符号错误
        amp_failed = not cuda_results.get("amp_C", True)
        if amp_failed:
            print("\n注意: amp_C 模块失败通常不影响训练，只是 AMP 功能可能不可用")
            print("      如果训练中不使用 AMP，可以忽略此错误")
        
        return 0
    elif apex_basic:
        print("⚠ APEX 基础包已安装，但 CUDA 扩展缺失")
        print("  建议重新安装 APEX 并确保设置 APEX_CUDA_EXT=1")
        if not critical_ok:
            print("  ⚠ 关键模块缺失，gradient_accumulation_fusion 不可用")
            print("  → 解决方案: 在训练脚本中添加 --no_gradient_accumulation_fusion true")
        
        # 检查是否是符号错误
        amp_failed = not cuda_results.get("amp_C", True)
        if amp_failed:
            print("\n⚠ amp_C 模块失败可能是 PyTorch 版本不匹配导致的")
            print("  如果训练中不使用 AMP，可以忽略此错误")
        
        return 1
    else:
        print("❌ APEX 未正确安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())

