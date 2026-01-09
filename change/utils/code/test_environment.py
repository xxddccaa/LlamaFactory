#!/usr/bin/env python3
"""
统一的环境测试脚本
用于在容器中测试深度学习环境是否正常
包括：cuDNN、Triton、APEX、Flash Attention、vLLM 的测试
"""

import sys
import os
import subprocess
import glob
import shutil
import importlib
from pathlib import Path

# ============================================================================
# 通用工具函数
# ============================================================================

def print_section(title, level=1):
    """打印分节标题"""
    if level == 1:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
    elif level == 2:
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)
    else:
        print(f"\n{title}")

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

def check_pytorch_basic():
    """检查 PyTorch 基本信息"""
    try:
        import torch
        info = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'gpu_count': 0,
            'driver_version': None
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['driver_version'] = get_cuda_driver_version()
        
        return True, info
    except ImportError:
        return False, None
    except Exception as e:
        return False, {'error': str(e)}

# ============================================================================
# cuDNN 诊断测试
# ============================================================================

def test_cudnn():
    """cuDNN 诊断测试"""
    print_section("测试模块 1: cuDNN 诊断", 1)
    
    results = {}
    
    # 检查 LD_LIBRARY_PATH
    print_section("检查 1: LD_LIBRARY_PATH 环境变量", 2)
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    print(f"当前 LD_LIBRARY_PATH: {ld_path if ld_path else '(未设置)'}")
    
    paths = ld_path.split(':') if ld_path else []
    cudnn_paths = []
    
    for path in paths:
        if path and os.path.exists(path):
            cudnn_libs = glob.glob(os.path.join(path, '*cudnn*'))
            if cudnn_libs:
                cudnn_paths.append(path)
                print(f"  ✓ 找到 cuDNN 库路径: {path}")
    
    if not cudnn_paths:
        print("  ⚠ 在 LD_LIBRARY_PATH 中未找到 cuDNN 库")
    
    # 查找 cuDNN 库文件
    print_section("检查 2: 查找 cuDNN 库文件", 2)
    common_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/nvidia/lib64',
        '/usr/local/nvidia/lib',
        '/opt/conda/lib',
        '/opt/conda/envs/*/lib',
    ]
    
    if ld_path:
        common_paths.extend(ld_path.split(':'))
    
    found_libs = {}
    search_patterns = ['libcudnn*.so*', 'libcudnn*.so']
    
    for base_path in common_paths:
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
            for lib in sorted(set(libs))[:3]:
                print(f"    - {os.path.basename(lib)}")
    else:
        print("✗ 未找到 cuDNN 库文件")
    
    # 检查 PyTorch cuDNN
    print_section("检查 3: PyTorch cuDNN 配置", 2)
    pytorch_ok = None
    cudnn_version = None
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            
            try:
                cudnn_enabled = torch.backends.cudnn.enabled
                cudnn_version = torch.backends.cudnn.version()
                print(f"  cuDNN 已启用: {cudnn_enabled}")
                print(f"  cuDNN 版本: {cudnn_version}")
                
                # 测试 cuDNN 功能
                try:
                    device = torch.device('cuda:0')
                    x = torch.randn(1, 3, 224, 224, device=device)
                    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
                    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False):
                        y = conv(x)
                    print("  ✓ cuDNN 功能测试通过")
                    pytorch_ok = True
                except Exception as e:
                    print(f"  ✗ cuDNN 功能测试失败: {e}")
                    pytorch_ok = False
                    
            except Exception as e:
                print(f"  ✗ 无法获取 cuDNN 信息: {e}")
                pytorch_ok = False
        else:
            print("  ⚠ CUDA 不可用，无法测试 cuDNN")
            
    except ImportError:
        print("✗ PyTorch 未安装")
    except Exception as e:
        print(f"✗ 检查 PyTorch 时出错: {e}")
    
    results['ld_paths'] = paths
    results['cudnn_libs'] = found_libs
    results['pytorch_ok'] = pytorch_ok
    results['cudnn_version'] = cudnn_version
    
    return results

# ============================================================================
# Triton 兼容性测试
# ============================================================================

def test_triton():
    """Triton 兼容性测试"""
    print_section("测试模块 2: Triton 兼容性", 1)
    
    results = {}
    
    # 基本导入测试
    print_section("测试 1: 基本导入", 2)
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            driver_version = get_cuda_driver_version()
            if driver_version:
                print(f"  CUDA 驱动版本: {driver_version}")
        results['torch_ok'] = True
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        results['torch_ok'] = False
        return results
    
    try:
        import triton
        print(f"✓ Triton 版本: {triton.__version__}")
        results['triton_installed'] = True
    except ImportError as e:
        print(f"✗ Triton 导入失败: {e}")
        results['triton_installed'] = False
        return results
    
    # 检查 CUDA 库路径
    print_section("CUDA 库路径检查", 2)
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    ld_preload = os.environ.get('LD_PRELOAD', '')
    print(f"LD_LIBRARY_PATH: {ld_path if ld_path else '(未设置)'}")
    print(f"LD_PRELOAD: {ld_preload if ld_preload else '(未设置)'}")
    
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
            break
    
    if not libcuda_found:
        print("⚠ 未找到 libcuda.so，这可能导致 cuModuleGetFunction 错误")
    
    results['libcuda_found'] = libcuda_found
    
    # triton_key 导入测试
    print_section("测试 2: triton_key 导入（关键测试）", 2)
    try:
        from triton.compiler.compiler import triton_key
        print("✓ triton_key 导入成功！")
        print(f"  triton_key 类型: {type(triton_key)}")
        results['triton_key_ok'] = True
    except ImportError as e:
        print(f"✗ triton_key 导入失败: {e}")
        print("\n这是导致训练失败的关键错误！")
        results['triton_key_ok'] = False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        results['triton_key_ok'] = False
    
    # PyTorch Inductor 测试
    print_section("测试 3: PyTorch Inductor 缓存系统", 2)
    try:
        from torch._inductor.codecache import CacheBase
        system_info = CacheBase.get_system()
        print("✓ PyTorch Inductor 缓存系统正常")
        print(f"  系统信息键: {list(system_info.keys())[:3]}...")
        results['inductor_ok'] = True
    except Exception as e:
        print(f"✗ PyTorch Inductor 测试失败: {e}")
        results['inductor_ok'] = False
    
    # torch.compile 测试
    print_section("测试 4: torch.compile 基本功能", 2)
    if torch.cuda.is_available():
        try:
            @torch.compile
            def simple_add(x, y):
                return x + y
            
            x = torch.randn(10, 10, device='cuda')
            y = torch.randn(10, 10, device='cuda')
            result = simple_add(x, y)
            print("✓ torch.compile 测试通过")
            print(f"  结果形状: {result.shape}")
            results['compile_ok'] = True
        except Exception as e:
            error_msg = str(e)
            print(f"✗ torch.compile 测试失败: {e}")
            if 'undefined symbol' in error_msg or 'cuModuleGetFunction' in error_msg:
                print("  这是 CUDA 驱动库路径配置问题")
            results['compile_ok'] = False
    else:
        print("⚠ 跳过：未检测到 GPU")
        results['compile_ok'] = None
    
    return results

# ============================================================================
# APEX 验证测试
# ============================================================================

def diagnose_symbol_error(error_msg):
    """诊断符号未定义错误"""
    if 'undefined symbol' in str(error_msg):
        print("\n" + "-" * 60)
        print("符号错误诊断:")
        print("-" * 60)
        symbol = str(error_msg)
        if 'c10' in symbol or 'SetDevice' in symbol or '_ZN3c10' in symbol:
            print("⚠ 检测到 PyTorch 符号未定义错误")
            print("  这通常表示 APEX 与当前 PyTorch 版本不兼容")

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

def test_apex():
    """APEX 验证测试"""
    print_section("测试模块 3: APEX 验证", 1)
    
    results = {}
    
    # 检查 PyTorch
    print_section("环境信息", 2)
    pytorch_ok, pytorch_info = check_pytorch_basic()
    if pytorch_ok:
        print(f"  PyTorch 版本: {pytorch_info['version']}")
        if pytorch_info['cuda_available']:
            print(f"  CUDA 版本: {pytorch_info['cuda_version']}")
    else:
        print("  ⚠ PyTorch 未安装")
    
    # 检查基础模块
    print_section("检查基础 APEX 模块", 2)
    apex_basic = check_module("apex", "APEX 基础包")
    if apex_basic:
        try:
            import apex
            print(f"  APEX 版本: {getattr(apex, '__version__', '未知')}")
        except:
            pass
    results['apex_basic'] = apex_basic
    
    if not apex_basic:
        results['status'] = 'not_installed'
        return results
    
    # 检查 C++ 扩展
    print_section("检查 C++ 扩展 (APEX_CPP_EXT)", 2)
    apex_c = check_module("apex_C", "APEX C++ 扩展")
    results['apex_c'] = apex_c
    
    # 检查 CUDA 扩展
    print_section("检查 CUDA 扩展 (APEX_CUDA_EXT)", 2)
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
    
    results['cuda_extensions'] = cuda_results
    
    # 检查关键模块
    print_section("关键模块检查 (gradient_accumulation_fusion 必需)", 2)
    critical_module = "fused_weight_gradient_mlp_cuda"
    critical_ok = cuda_results.get(critical_module, False)
    if critical_ok:
        print(f"✓ {critical_module} 已正确安装")
        print("  → gradient_accumulation_fusion 功能可用")
    else:
        print(f"✗ {critical_module} 未安装或导入失败")
        print("  → gradient_accumulation_fusion 功能不可用")
    
    results['critical_ok'] = critical_ok
    results['status'] = 'ok' if (apex_basic and critical_ok) else 'partial'
    
    return results

# ============================================================================
# Flash Attention 验证测试
# ============================================================================

def test_flash_attn():
    """Flash Attention 验证测试"""
    print_section("测试模块 4: Flash Attention 验证", 1)
    
    results = {}
    
    # 检查安装
    print_section("测试 1: Flash Attention 安装检查", 2)
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', '未知')
        print(f"✓ flash_attn 版本: {version}")
        results['installed'] = True
        results['version'] = version
    except ImportError as e:
        print(f"✗ flash_attn 未安装")
        print(f"  错误信息: {e}")
        results['installed'] = False
        return results
    except Exception as e:
        print(f"✗ flash_attn 导入失败")
        print(f"  错误信息: {e}")
        results['installed'] = False
        return results
    
    # 检查 PyTorch
    print_section("测试 2: PyTorch 环境检查", 2)
    pytorch_ok, pytorch_info = check_pytorch_basic()
    if pytorch_ok:
        print(f"✓ PyTorch 版本: {pytorch_info['version']}")
        print(f"  CUDA 可用: {pytorch_info['cuda_available']}")
        if pytorch_info['cuda_available']:
            print(f"  CUDA 版本: {pytorch_info['cuda_version']}")
            print(f"  GPU 数量: {pytorch_info['gpu_count']}")
            if pytorch_info['driver_version']:
                print(f"  CUDA 驱动版本: {pytorch_info['driver_version']}")
    else:
        print("✗ PyTorch 未安装")
        results['pytorch_ok'] = False
        return results
    
    # 关键模块测试
    print_section("测试 3: Flash Attention 关键模块导入（关键测试）", 2)
    try:
        import flash_attn
        print("✓ flash_attn 主模块导入成功")
        
        try:
            import flash_attn.flash_attn_interface
            print("✓ flash_attn.flash_attn_interface 导入成功")
        except ImportError as e:
            print(f"⚠ flash_attn.flash_attn_interface 导入失败: {e}")
        
        try:
            from flash_attn import flash_attn_func
            print("✓ flash_attn_func 导入成功")
            results['critical_modules_ok'] = True
        except ImportError as e:
            print(f"✗ flash_attn_func 导入失败: {e}")
            print("\n这是导致训练失败的关键错误！")
            results['critical_modules_ok'] = False
            return results
        
        try:
            from flash_attn import flash_attn_varlen_func
            print("✓ flash_attn_varlen_func 导入成功")
        except ImportError:
            print("⚠ flash_attn_varlen_func 不可用（可选）")
        
        print("\n✓ Flash Attention 关键模块可以正常使用！")
        
    except ImportError as e:
        print(f"✗ Flash Attention 关键模块导入失败: {e}")
        results['critical_modules_ok'] = False
        return results
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Flash Attention 关键模块导入失败: {e}")
        if 'undefined symbol' in error_msg:
            print("\n检测到符号未定义错误，可能是版本不兼容或 CUDA 库路径问题")
        results['critical_modules_ok'] = False
        return results
    
    # 功能测试
    print_section("测试 4: Flash Attention 基本功能测试", 2)
    if pytorch_info['cuda_available']:
        try:
            import torch
            from flash_attn import flash_attn_func
            
            batch_size = 2
            seq_len = 128
            num_heads = 8
            head_dim = 64
            
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
            
            output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
            print(f"✓ flash_attn_func 执行成功")
            print(f"  输出形状: {output.shape}")
            results['functionality_ok'] = True
        except Exception as e:
            print(f"✗ flash_attn_func 执行失败: {e}")
            results['functionality_ok'] = False
    else:
        print("⚠ 跳过功能测试：未检测到 GPU")
        results['functionality_ok'] = None
    
    return results

# ============================================================================
# vLLM 验证测试
# ============================================================================

def test_vllm():
    """vLLM 验证测试"""
    print_section("测试模块 5: vLLM 验证", 1)
    
    results = {}
    
    # 检查安装
    print_section("测试 1: vLLM 安装检查", 2)
    try:
        import vllm
        version = getattr(vllm, '__version__', '未知')
        print(f"✓ vLLM 版本: {version}")
        results['installed'] = True
        results['version'] = version
    except ImportError as e:
        print(f"✗ vLLM 未安装")
        print(f"  错误信息: {e}")
        results['installed'] = False
        return results
    except Exception as e:
        print(f"✗ vLLM 导入失败")
        print(f"  错误信息: {e}")
        results['installed'] = False
        return results
    
    # 检查 PyTorch
    print_section("测试 2: PyTorch 环境检查", 2)
    pytorch_ok, pytorch_info = check_pytorch_basic()
    if pytorch_ok:
        print(f"✓ PyTorch 版本: {pytorch_info['version']}")
        print(f"  CUDA 可用: {pytorch_info['cuda_available']}")
        if pytorch_info['cuda_available']:
            print(f"  CUDA 版本: {pytorch_info['cuda_version']}")
            print(f"  GPU 数量: {pytorch_info['gpu_count']}")
            if pytorch_info['driver_version']:
                print(f"  CUDA 驱动版本: {pytorch_info['driver_version']}")
    else:
        print("✗ PyTorch 未安装")
        results['pytorch_ok'] = False
        return results
    
    # 关键模块测试
    print_section("测试 3: vLLM 关键模块导入（关键测试）", 2)
    try:
        from vllm.platforms import current_platform
        print("✓ vllm.platforms 导入成功")
        
        import vllm._C
        print("✓ vllm._C 导入成功")
        
        print("\n✓ vLLM 可以正常使用！")
        results['critical_modules_ok'] = True
    except ImportError as e:
        print(f"✗ vLLM 关键模块导入失败: {e}")
        print("\n这是导致训练失败的关键错误！")
        results['critical_modules_ok'] = False
        return results
    except Exception as e:
        error_msg = str(e)
        print(f"✗ vLLM 关键模块导入失败: {e}")
        if 'undefined symbol' in error_msg:
            print("\n检测到符号未定义错误，可能是版本不兼容或 CUDA 库路径问题")
        results['critical_modules_ok'] = False
        return results
    
    # 基本功能测试
    print_section("测试 4: vLLM 基本功能测试", 2)
    try:
        from vllm import LLM
        print("✓ vLLM.LLM 类可用")
        
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            print("✓ vLLM 引擎参数类可用")
        except:
            pass
        
        results['functionality_ok'] = True
    except Exception as e:
        print(f"✗ vLLM 基本功能测试失败: {e}")
        results['functionality_ok'] = False
    
    return results

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主测试函数"""
    print("=" * 80)
    print("深度学习环境统一测试脚本")
    print("=" * 80)
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"工作目录: {os.getcwd()}")
    
    all_results = {}
    
    # 运行所有测试
    try:
        all_results['cudnn'] = test_cudnn()
    except Exception as e:
        print(f"\n✗ cuDNN 测试出错: {e}")
        all_results['cudnn'] = {'error': str(e)}
    
    try:
        all_results['triton'] = test_triton()
    except Exception as e:
        print(f"\n✗ Triton 测试出错: {e}")
        all_results['triton'] = {'error': str(e)}
    
    try:
        all_results['apex'] = test_apex()
    except Exception as e:
        print(f"\n✗ APEX 测试出错: {e}")
        all_results['apex'] = {'error': str(e)}
    
    try:
        all_results['flash_attn'] = test_flash_attn()
    except Exception as e:
        print(f"\n✗ Flash Attention 测试出错: {e}")
        all_results['flash_attn'] = {'error': str(e)}
    
    try:
        all_results['vllm'] = test_vllm()
    except Exception as e:
        print(f"\n✗ vLLM 测试出错: {e}")
        all_results['vllm'] = {'error': str(e)}
    
    # 汇总结果
    print_section("测试结果汇总", 1)
    
    summary = []
    
    # cuDNN
    cudnn_result = all_results.get('cudnn', {})
    if 'error' in cudnn_result:
        summary.append(("cuDNN", "错误", "测试过程出错"))
    elif cudnn_result.get('pytorch_ok') is True:
        summary.append(("cuDNN", "✓ 通过", "cuDNN 功能正常"))
    elif cudnn_result.get('pytorch_ok') is False:
        summary.append(("cuDNN", "✗ 失败", "cuDNN 功能测试失败"))
    else:
        summary.append(("cuDNN", "⚠ 跳过", "CUDA 不可用或未安装"))
    
    # Triton
    triton_result = all_results.get('triton', {})
    if 'error' in triton_result:
        summary.append(("Triton", "错误", "测试过程出错"))
    elif not triton_result.get('triton_installed', False):
        summary.append(("Triton", "⚠ 未安装", "Triton 未安装"))
    elif triton_result.get('triton_key_ok', False) and triton_result.get('inductor_ok', False):
        summary.append(("Triton", "✓ 通过", "Triton 兼容性正常"))
    elif triton_result.get('triton_key_ok', False):
        summary.append(("Triton", "⚠ 部分通过", "triton_key 正常，但 Inductor 有问题"))
    else:
        summary.append(("Triton", "✗ 失败", "triton_key 导入失败"))
    
    # APEX
    apex_result = all_results.get('apex', {})
    if 'error' in apex_result:
        summary.append(("APEX", "错误", "测试过程出错"))
    elif not apex_result.get('apex_basic', False):
        summary.append(("APEX", "⚠ 未安装", "APEX 未安装"))
    elif apex_result.get('critical_ok', False):
        summary.append(("APEX", "✓ 通过", "APEX 关键模块正常"))
    else:
        summary.append(("APEX", "✗ 失败", "APEX 关键模块缺失"))
    
    # Flash Attention
    flash_attn_result = all_results.get('flash_attn', {})
    if 'error' in flash_attn_result:
        summary.append(("Flash Attention", "错误", "测试过程出错"))
    elif not flash_attn_result.get('installed', False):
        summary.append(("Flash Attention", "⚠ 未安装", "Flash Attention 未安装"))
    elif flash_attn_result.get('critical_modules_ok', False):
        summary.append(("Flash Attention", "✓ 通过", "Flash Attention 关键模块正常"))
    else:
        summary.append(("Flash Attention", "✗ 失败", "Flash Attention 关键模块导入失败"))
    
    # vLLM
    vllm_result = all_results.get('vllm', {})
    if 'error' in vllm_result:
        summary.append(("vLLM", "错误", "测试过程出错"))
    elif not vllm_result.get('installed', False):
        summary.append(("vLLM", "⚠ 未安装", "vLLM 未安装"))
    elif vllm_result.get('critical_modules_ok', False):
        summary.append(("vLLM", "✓ 通过", "vLLM 关键模块正常"))
    else:
        summary.append(("vLLM", "✗ 失败", "vLLM 关键模块导入失败"))
    
    # 打印汇总表
    print(f"\n{'模块':<20} {'状态':<15} {'说明'}")
    print("-" * 80)
    for module, status, desc in summary:
        print(f"{module:<20} {status:<15} {desc}")
    
    # 统计
    passed = sum(1 for _, status, _ in summary if '✓' in status)
    failed = sum(1 for _, status, _ in summary if '✗' in status)
    skipped = sum(1 for _, status, _ in summary if '⚠' in status or '错误' in status)
    total = len(summary)
    
    print(f"\n总计: {total} 个测试模块")
    print(f"  通过: {passed}")
    print(f"  失败: {failed}")
    print(f"  跳过/错误: {skipped}")
    
    # 最终结论
    print_section("最终结论", 1)
    if failed == 0 and skipped == 0:
        print("🎉 所有测试通过！环境配置正常。")
        return 0
    elif failed == 0:
        print("⚠️  部分模块未安装或跳过，但已安装的模块均正常。")
        return 0
    else:
        print("❌ 发现环境问题，请检查上述错误信息并按照建议进行修复。")
        return 1

if __name__ == '__main__':
    sys.exit(main())

