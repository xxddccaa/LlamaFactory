# LlamaFactory 分支更改说明

**分支名称**: `suport_tars`
**基于分支**: `main`
**更改日期**: 2025年
**提交数**: 3

---

## 核心功能更改

### 1. TARS 格式数据支持 (commit: 8863f81)

支持 WebDataset TARS 格式的数据集加载，适用于大规模多模态数据训练。

**涉及文件**:
- `src/llamafactory/data/converter.py` - 新增 TARS 数据格式转换逻辑
- `src/llamafactory/data/loader.py` - 新增 TARS 数据集加载支持
- `src/llamafactory/data/parser.py` - 新增 TARS 数据集配置解析
- `src/llamafactory/data/processor/supervised.py` - 调整 SFT 处理器以支持 TARS

**使用方式**:
```yaml
# 在 dataset_info.json 中配置 TARS 数据集
"my_tars_dataset": {
  "file_name": "/path/to/data-{000000..000099}.tar",
  "formatting": "tars"
}
```

### 2. 直接文件路径数据集支持 (commit: f578889)

新增多项数据集配置增强功能：

#### 2.1 直接文件路径支持
可以直接使用文件路径作为数据集，无需在 `dataset_info.json` 中预定义。

```yaml
# 直接在训练配置中使用文件路径
dataset: /path/to/your/data.json
```

#### 2.2 数据集特定的 media_dir
每个数据集可以配置独立的媒体文件目录。

```json
"my_dataset": {
  "file_name": "data.json",
  "media_dir": "/path/to/images/"
}
```

#### 2.3 远程图片路径支持
支持 HTTP/HTTPS 远程图片 URL，自动下载并处理。

**涉及文件**:
- `src/llamafactory/data/loader.py` - 直接路径加载逻辑
- `src/llamafactory/data/parser.py` - 配置解析增强
- `src/llamafactory/data/mm_plugin.py` - 远程图片处理
- `src/llamafactory/data/converter.py` - 格式转换增强

---

## 文件清单

### 核心更改文件

| 文件路径 | 说明 |
|----------|------|
| `src/llamafactory/data/converter.py` | 数据格式转换，新增 TARS 支持 |
| `src/llamafactory/data/loader.py` | 数据加载器，新增直接路径和 TARS 支持 |
| `src/llamafactory/data/parser.py` | 数据集配置解析，新增 media_dir 等参数 |
| `src/llamafactory/data/mm_plugin.py` | 多模态插件，新增远程图片支持 |
| `src/llamafactory/data/processor/supervised.py` | SFT 处理器调整 |
| `src/llamafactory/hparams/parser.py` | 参数解析器微调 |

### 配置示例文件

| 文件路径 | 说明 |
|----------|------|
| `examples/megatron/qwen3_full.yaml` | Megatron 训练配置示例 |
| `docker/docker-cuda/Dockerfile2.megatron` | Megatron 环境 Dockerfile |

### 工具脚本

| 文件路径 | 说明 |
|----------|------|
| `utils/code/fix_torch_version.py` | PyTorch 版本修复工具 |
| `utils/code/fix_torchvision_version.py` | TorchVision 版本修复工具 |
| `utils/code/get_model_pata.py` | 模型路径获取工具 |
| `utils/code/test_diagnose_cudnn.py` | cuDNN 诊断测试 |
| `utils/code/test_environment.py` | 环境测试工具 |
| `utils/code/test_triton_compat.py` | Triton 兼容性测试 |
| `utils/code/test_verify_apex.py` | Apex 验证测试 |
| `utils/code/test_verify_flash_attn.py` | Flash Attention 验证测试 |
| `utils/code/test_verify_vllm.py` | vLLM 验证测试 |
| `utils/export_dataset_to_json.py` | 数据集导出工具 |

### 文档文件

| 文件路径 | 说明 |
|----------|------|
| `README.md` | 中文 README (精简版) |
| `README_en.md` | 英文 README |
| `.dockerignore` | Docker 忽略配置 |

---

## 集成指南

### 方法一：合并文件

将以下核心文件复制到 main 分支对应位置：

```bash
# 核心数据处理文件 (必须)
cp change/src/llamafactory/data/converter.py src/llamafactory/data/
cp change/src/llamafactory/data/loader.py src/llamafactory/data/
cp change/src/llamafactory/data/parser.py src/llamafactory/data/
cp change/src/llamafactory/data/mm_plugin.py src/llamafactory/data/
cp change/src/llamafactory/data/processor/supervised.py src/llamafactory/data/processor/
cp change/src/llamafactory/hparams/parser.py src/llamafactory/hparams/
```

### 方法二：Git 合并

```bash
git checkout main
git merge suport_tars
```

---

## 依赖说明

TARS 格式支持需要安装 WebDataset：

```bash
pip install webdataset
```

---

## 统计信息

- 新增代码行数: ~4686 行
- 删除代码行数: ~1041 行
- 净增加: ~3645 行
- 更改文件数: 21 个
