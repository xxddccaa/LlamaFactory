from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/jfs6/model/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True  # 只使用本地文件，不尝试从网络下载
)

def count_parameters(module):
    """统计模块的参数量"""
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())

# 查看总参数量
total_params = count_parameters(model)
print(f"总参数量: {total_params:,}")
print()

# 分别统计各个组件的参数量
print("=" * 70)
print("各组件参数量统计:")
print("=" * 70)

# 1. 语言模型 (LLM) 核心
# Qwen2.5-VL 的结构: model.language_model 或 model (取决于 transformers 版本)
llm_params = 0
if hasattr(model, 'model'):
    if hasattr(model.model, 'language_model'):
        # 新版本结构: model.language_model
        llm_params = count_parameters(model.model.language_model)
    else:
        # 旧版本结构: model 本身是语言模型，但需要排除 visual
        if hasattr(model.model, 'visual'):
            # 计算除了 visual 之外的所有参数
            visual_total = count_parameters(model.model.visual)
            llm_params = count_parameters(model.model) - visual_total
        else:
            llm_params = count_parameters(model.model)

# 加上 lm_head（如果有）
if hasattr(model, 'lm_head'):
    llm_params += count_parameters(model.lm_head)

print(f"1. 语言模型 (LLM) 核心: {llm_params:,} ({llm_params/total_params*100:.2f}%)")

# 2. 视觉编码器 (Vision Transformer) 和 3. 视觉-语言融合模块 (Merger)
visual_module = None
if hasattr(model, 'model') and hasattr(model.model, 'visual'):
    visual_module = model.model.visual
elif hasattr(model, 'visual'):
    visual_module = model.visual

if visual_module is not None:
    visual_total = count_parameters(visual_module)
    
    # 计算 merger 的参数（如果存在）
    merger_params = 0
    if hasattr(visual_module, 'merger'):
        merger_params = count_parameters(visual_module.merger)
        vision_encoder_params = visual_total - merger_params
    else:
        # 如果没有 merger 子模块，尝试查找其他可能的融合模块
        vision_encoder_params = visual_total
        merger_params = 0
    
    print(f"2. 视觉编码器 (Vision Transformer): {vision_encoder_params:,} ({vision_encoder_params/total_params*100:.2f}%)")
    
    if merger_params > 0:
        print(f"3. 视觉-语言融合模块 (Merger): {merger_params:,} ({merger_params/total_params*100:.2f}%)")
    else:
        print(f"3. 视觉-语言融合模块 (Merger): 未找到独立模块")
else:
    print(f"2. 视觉编码器 (Vision Transformer): 未找到")
    print(f"3. 视觉-语言融合模块 (Merger): 未找到")
    vision_encoder_params = 0
    merger_params = 0

# 验证：计算其他参数
calculated_total = llm_params + vision_encoder_params + merger_params
other_params = total_params - calculated_total

if abs(other_params) > 100:  # 允许小的误差
    print(f"4. 其他模块: {other_params:,} ({other_params/total_params*100:.2f}%)")

print("=" * 70)
print(f"总计: {total_params:,} (100.00%)")
print("=" * 70)

# 打印模型结构信息（用于调试）
print("\n模型结构信息:")
if hasattr(model, 'model'):
    print(f"  - model 存在")
    if hasattr(model.model, 'language_model'):
        print(f"  - model.language_model 存在")
    if hasattr(model.model, 'visual'):
        print(f"  - model.visual 存在")
        if hasattr(model.model.visual, 'merger'):
            print(f"  - model.visual.merger 存在")
if hasattr(model, 'lm_head'):
    print(f"  - lm_head 存在")
