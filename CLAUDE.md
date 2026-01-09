# LlamaFactory 训练框架结构

## 命令行入口

```bash
llamafactory-cli train <config.yaml>    # 训练模型
llamafactory-cli api <config.yaml>      # 启动 OpenAI 兼容 API
llamafactory-cli chat <config.yaml>     # CLI 聊天
llamafactory-cli export <config.yaml>   # 导出/合并模型
llamafactory-cli webui                  # 启动 Web UI
llamafactory-cli env                    # 显示环境信息

# 快捷方式
lmf train <config.yaml>
```

## 顶层目录结构

```
LlamaFactory/
├── src/llamafactory/     # 核心源代码
├── examples/             # 配置示例 (train_full/, train_lora/, deepspeed/, megatron/)
├── data/                 # 演示数据集
├── tests/                # 测试套件
├── scripts/              # 辅助脚本 (转换, 推理, 评估)
├── docker/               # Docker 配置 (CUDA/NPU/ROCm)
└── saves/                # 训练输出目录
```

## 核心模块结构 (src/llamafactory/)

```
src/llamafactory/
├── cli.py                # 命令行入口
├── launcher.py           # 启动器 - 命令分发和分布式训练

├── hparams/              # 超参数管理
│   ├── parser.py         # 参数解析器
│   ├── model_args.py     # 模型参数
│   ├── data_args.py      # 数据参数
│   ├── training_args.py  # 训练参数
│   └── finetuning_args.py# 微调参数

├── data/                 # 数据处理
│   ├── loader.py         # 数据加载 (本地文件/HF Hub/ModelScope)
│   ├── parser.py         # 数据集配置解析 (DatasetAttr)
│   ├── converter.py      # 数据格式转换
│   ├── template.py       # 对话模板
│   ├── collator.py       # Batch 整理器
│   ├── mm_plugin.py      # 多模态插件
│   └── processor/        # 数据处理器
│       ├── supervised.py     # SFT 处理器
│       ├── unsupervised.py   # 预训练处理器
│       ├── pairwise.py       # DPO/RM 处理器
│       └── feedback.py       # KTO 处理器

├── model/                # 模型加载和适配
│   ├── loader.py         # 模型加载器
│   ├── adapter.py        # LoRA/OFT/Freeze 适配器
│   ├── patcher.py        # 模型/Tokenizer 修补
│   └── model_utils/      # 模型优化工具
│       ├── quantization.py   # 量化
│       ├── attention.py      # 注意力优化
│       ├── rope.py           # RoPE 配置
│       ├── longlora.py       # 长度扩展
│       └── visual.py         # 视觉模型支持

├── train/                # 训练模块
│   ├── tuner.py          # 主入口 run_exp(), export_model()
│   ├── sft/              # 监督微调
│   ├── pt/               # 预训练
│   ├── rm/               # 奖励模型
│   ├── dpo/              # 直接偏好优化
│   ├── ppo/              # PPO 强化学习
│   ├── kto/              # KTO 优化
│   └── mca/              # Megatron Core Adapter

├── chat/                 # 推理引擎
│   ├── chat_model.py     # 聊天模型
│   ├── hf_engine.py      # HF Transformers 引擎
│   ├── vllm_engine.py    # vLLM 引擎
│   └── sglang_engine.py  # SGLang 引擎

├── api/                  # OpenAI 兼容 API
│   ├── app.py            # FastAPI 应用
│   └── chat.py           # 聊天接口

├── webui/                # Gradio Web UI
│   ├── interface.py      # 界面入口
│   └── components/       # UI 组件

└── extras/               # 辅助工具
    ├── env.py            # 环境管理
    ├── constants.py      # 常量定义
    └── logging.py        # 日志管理
```

## 数据处理流程

```
loader.py: _load_single_dataset()
    ↓
parser.py: DatasetAttr (解析配置)
    ↓
converter.py: align_dataset() (格式对齐)
    ↓
processor/*: 数据处理器 (SFT/PT/DPO/KTO)
    ↓
template.py: Template (应用对话模板)
    ↓
collator.py: DataCollator (batch 整理)
```

## 训练流程

```
tuner.py: run_exp()
    ↓
hparams/parser.py: get_train_args()
    ↓
选择训练阶段:
├── sft/workflow.py: run_sft()   # 监督微调
├── pt/workflow.py: run_pt()     # 预训练
├── rm/workflow.py: run_rm()     # 奖励模型
├── dpo/workflow.py: run_dpo()   # DPO
├── ppo/workflow.py: run_ppo()   # PPO
├── kto/workflow.py: run_kto()   # KTO
└── mca/workflow.py: run_*()     # Megatron
    ↓
*/trainer.py: 自定义训练器
```

## 支持的训练阶段 (stage)

| 阶段 | 模块 | 功能 |
|------|------|------|
| `sft` | train/sft/ | 监督微调 (指令调整) |
| `pt` | train/pt/ | 预训练 |
| `rm` | train/rm/ | 奖励模型 (RLHF) |
| `dpo` | train/dpo/ | 直接偏好优化 |
| `ppo` | train/ppo/ | PPO 强化学习 |
| `kto` | train/kto/ | KTO 优化 |

## 微调类型 (finetuning_type)

- `full` - 全参数微调
- `lora` - LoRA 低秩适配
- `oft` - OFT 正交微调
- `freeze` - 冻结层微调

## 配置文件示例

```yaml
### 模型
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora  # full, lora, oft, freeze
deepspeed: examples/deepspeed/ds_z3_config.json

### 数据集
dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000

### 输出
output_dir: saves/qwen3-4b/lora/sft
logging_steps: 10
save_steps: 500

### 训练
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
```

## 示例配置目录

```
examples/
├── train_full/           # 全参数微调
├── train_lora/           # LoRA 微调
├── train_qlora/          # QLoRA 微调
├── deepspeed/            # DeepSpeed 配置 (ds_z0/z2/z3)
├── accelerate/           # FSDP 配置
├── megatron/             # Megatron 配置
├── inference/            # 推理配置
├── merge_lora/           # LoRA 合并配置
└── extras/               # 高级优化 (galore, badam, fp8, pissa...)
```

## 环境变量

```bash
USE_V1=1                      # 使用 V1 新架构
USE_MCA=1                     # 使用 Megatron Core Adapter
USE_MODELSCOPE_HUB=1          # 使用 ModelScope
USE_OPENMIND_HUB=1            # 使用 OpenMind
DISABLE_VERSION_CHECK=1       # 禁用版本检查
LLAMAFACTORY_VERBOSITY=WARN   # 日志级别
FORCE_TORCHRUN=1              # 强制使用 torchrun
```

## 多模态支持

- 图像占位符: `<image>`
- 音频占位符: `<audio>`
- 关键模块: `data/mm_plugin.py`, `model/model_utils/visual.py`

## 关键文件位置

| 功能 | 文件路径 |
|------|----------|
| 命令行入口 | `src/llamafactory/cli.py` |
| 训练主入口 | `src/llamafactory/train/tuner.py` |
| 数据加载 | `src/llamafactory/data/loader.py` |
| 数据集配置 | `src/llamafactory/data/parser.py` |
| 模型加载 | `src/llamafactory/model/loader.py` |
| 对话模板 | `src/llamafactory/data/template.py` |
| SFT 训练器 | `src/llamafactory/train/sft/trainer.py` |
| DPO 训练器 | `src/llamafactory/train/dpo/trainer.py` |
| API 服务 | `src/llamafactory/api/app.py` |
| Web UI | `src/llamafactory/webui/interface.py` |

## 数据集配置 (data/dataset_info.json)

支持的数据源:
- 本地文件: CSV, JSON, JSONL, Arrow, Parquet, Text
- Hugging Face Hub
- ModelScope Hub
- OpenMind Hub

数据集配置支持:
- `file_name`: 本地文件路径 (支持直接文件路径)
- `media_dir`: 数据集特定的媒体目录
- 远程图片 URL (http/https)

## 运行示例

```bash
# 单卡 LoRA 微调
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 多卡 DeepSpeed 训练
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml

# Megatron 训练
USE_MCA=1 llamafactory-cli train examples/megatron/qwen3_vl_mmdu_full.yaml

# 启动 API 服务
llamafactory-cli api examples/inference/qwen3_vllm.yaml

# 启动 Web UI
llamafactory-cli webui
```
