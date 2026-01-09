# Dataset 内联配置功能 - 变更说明

## 功能概述

新增支持在 `--dataset` 参数中通过方括号语法直接配置数据集属性，无需创建 `dataset_info.json` 文件。

## 新增功能

### 1. 内联配置语法
支持在文件路径后使用 `[key=value,...]` 语法配置数据集属性：

```bash
--dataset "/mnt/data.json[media_dir=/mnt/images,formatting=alpaca]"
```

### 2. 支持的配置项
- 基础配置: `formatting`, `messages`, `media_dir`
- 标签配置: `role_tag`, `content_tag`, `user_tag`, `assistant_tag`
- 多模态: `images`, `videos`, `audios`, `system`, `tools`
- 历史掩码: `mask_history_sample`, `max_human_steps`

### 3. 远程路径支持
支持 S3、OSS、GCS 等远程路径：
```bash
--dataset "s3://bucket/data.json[media_dir=s3://bucket/images]"
```

## 代码变更

### 修改文件
1. `src/llamafactory/data/parser.py`
   - 新增 `_parse_dataset_path_with_config()` 函数：解析内联配置
   - 修改 `get_dataset_list()` 函数：应用内联配置
   - 更新 `_is_file_path()` 函数：支持方括号语法

### 新增文件
1. `tests/test_parser_inline_config.py` - 解析功能测试
2. `tests/test_dataset_path_config.py` - 完整功能测试
3. `docs/dataset_inline_config.md` - 完整文档（英文）
4. `docs/dataset_inline_config_zh.md` - 快速参考（中文）

## 核心实现

### 解析函数
```python
def _parse_dataset_path_with_config(path_with_config: str) -> tuple[str, dict[str, Any]]:
    """
    解析格式: /path/to/data.json[key1=value1,key2=value2]
    返回: (file_path, config_dict)
    """
```

### 配置应用
在 `get_dataset_list()` 中，对于文件路径类型的数据集：
1. 解析路径和内联配置
2. 创建 DatasetAttr 时应用内联配置
3. 验证配置有效性（如 mask_history_sample 配置）

## 使用示例

### 基础用法
```bash
llamafactory-cli train \
    --dataset "/mnt/data.json[media_dir=/mnt/images]" \
    --model_name_or_path /path/to/model
```

### 完整示例
```bash
llamafactory-cli train \
    --model_name_or_path /data/models/Qwen3-VL-2B \
    --dataset "s3://bucket/gui_data.json[media_dir=s3://bucket/images,mask_history_sample=true,max_human_steps=2]" \
    --template qwen3_vl \
    --stage sft \
    --do_train \
    --output_dir /data/output \
    --bf16
```

## 向后兼容

- ✅ 完全向后兼容
- ✅ 不影响现有的 dataset_info.json 配置方式
- ✅ 支持混合使用（内联配置 + dataset_info.json）

## 测试验证

### 测试覆盖
- ✅ 基础路径解析
- ✅ 单个配置项
- ✅ 多个配置项
- ✅ 布尔值和数字转换
- ✅ S3/OSS 远程路径
- ✅ mask_history_sample 验证
- ✅ 错误处理

### 测试结果
所有测试通过：
```bash
$ python tests/test_parser_inline_config.py
================================================================================
🎉 所有测试通过！
================================================================================
```

## 优势对比

### 传统方式
```json
// dataset_info.json
{
    "my_dataset": {
        "file_name": "train.json",
        "media_dir": "/mnt/images",
        "formatting": "sharegpt"
    }
}
```
```bash
llamafactory-cli train --dataset my_dataset --dataset_dir /mnt/
```

### 新方式
```bash
llamafactory-cli train --dataset "/mnt/train.json[media_dir=/mnt/images,formatting=sharegpt]"
```

**优势**:
- 🚀 无需额外配置文件
- 🎯 配置直观清晰
- 🔄 适合快速实验
- ☁️ 原生支持远程路径

## 应用场景

### 场景 1: 快速实验
无需创建配置文件，直接在命令行指定所有参数。

### 场景 2: 多数据集训练
每个数据集使用不同的 media_dir：
```bash
--dataset "data1.json[media_dir=/mnt/img1],data2.json[media_dir=/mnt/img2]"
```

### 场景 3: CI/CD 流水线
在脚本中动态生成训练命令，无需管理配置文件。

### 场景 4: 云存储数据
直接使用 S3/OSS 路径，无需下载到本地：
```bash
--dataset "s3://bucket/data.json[media_dir=s3://bucket/images]"
```

## 注意事项

1. **配置优先级**: 内联配置 > dataset_info.json > 全局参数
2. **必需配对**: `mask_history_sample` 和 `max_human_steps` 必须同时设置
3. **路径拼接**: 相对路径会拼接 media_dir，绝对路径和远程路径不拼接
4. **特殊字符**: 避免在值中使用逗号，复杂配置使用 dataset_info.json

## 未来改进

### 可能的扩展
1. 支持更复杂的语法（如引号包裹值）
2. 支持嵌套配置
3. 支持配置文件引用（如 `@config.yaml`）

## 相关 Issue / PR

- Feature Request: 支持 Swift 风格的数据集指定方式
- Implementation: Dataset 内联配置语法

## 文档

- 完整文档: [docs/dataset_inline_config.md](docs/dataset_inline_config.md)
- 快速参考: [docs/dataset_inline_config_zh.md](docs/dataset_inline_config_zh.md)
- 测试代码: [tests/test_parser_inline_config.py](tests/test_parser_inline_config.py)

## 变更日期

2026-01-09

## 作者

Implementation by AI Assistant

