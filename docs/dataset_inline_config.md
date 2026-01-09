# Dataset 内联配置语法

## 功能说明

LlamaFactory 支持在 `--dataset` 参数中直接指定文件路径，并通过方括号语法配置数据集属性。

## 语法格式

```bash
/path/to/data.json[key1=value1,key2=value2,...]
```

## 支持的配置项

### 基础配置
- `media_dir`: 媒体文件（图片、视频、音频）所在目录
- `formatting`: 数据格式，可选 `alpaca`、`sharegpt`、`openai`（默认：`sharegpt`）
- `messages`: 消息字段名（默认：`conversations`）

### 标签配置
- `role_tag`: 角色标签字段名（默认：`from`）
- `content_tag`: 内容标签字段名（默认：`value`）
- `user_tag`: 用户标签（默认：`human`）
- `assistant_tag`: 助手标签（默认：`gpt`）

### 多模态配置
- `images`: 图片字段名
- `videos`: 视频字段名
- `audios`: 音频字段名
- `system`: 系统提示字段名
- `tools`: 工具字段名

### 历史掩码配置
- `mask_history_sample`: 是否启用历史掩码（`true`/`false`）
- `max_human_steps`: 最大保留的 user 消息数量（整数）

**注意**: `mask_history_sample` 和 `max_human_steps` 必须同时设置。

## 使用示例

### 示例 1：基础用法 - 指定 media_dir

```bash
llamafactory-cli train \
    --model_name_or_path /path/to/model \
    --dataset "/mnt/jfs5/data/train.json[media_dir=/mnt/jfs5/images]" \
    --template qwen3_vl \
    --output_dir /path/to/output
```

**数据文件内容** (`/mnt/jfs5/data/train.json`):
```json
[
    {
        "conversations": [
            {"from": "human", "value": "<image>描述这张图片"},
            {"from": "gpt", "value": "这是一张..."}
        ],
        "image_name": ["images/photo1.jpg"]
    }
]
```

**实际图片路径**: `/mnt/jfs5/images/images/photo1.jpg`

### 示例 2：多个配置项

```bash
llamafactory-cli train \
    --dataset "/mnt/data.json[media_dir=/mnt/images,formatting=alpaca,user_tag=user,assistant_tag=assistant]" \
    --model_name_or_path /path/to/model
```

### 示例 3：S3 远程路径

```bash
llamafactory-cli train \
    --dataset "s3://bucket/data.json[media_dir=s3://bucket/images]" \
    --model_name_or_path /path/to/model
```

### 示例 4：多个数据集（混合配置）

```bash
llamafactory-cli train \
    --dataset "/mnt/data1.json[media_dir=/mnt/images1],/mnt/data2.json[media_dir=/mnt/images2],dataset_from_info_json" \
    --dataset_dir /path/to/dataset_dir \
    --model_name_or_path /path/to/model
```

说明：
- `/mnt/data1.json[media_dir=/mnt/images1]` - 直接文件路径 + 内联配置
- `/mnt/data2.json[media_dir=/mnt/images2]` - 直接文件路径 + 内联配置
- `dataset_from_info_json` - 从 dataset_info.json 读取的数据集

### 示例 5：配置历史掩码

```bash
llamafactory-cli train \
    --dataset "/mnt/data.json[media_dir=/mnt/images,mask_history_sample=true,max_human_steps=2]" \
    --model_name_or_path /path/to/model
```

这将启用历史掩码功能，每个训练样本最多保留 2 个 user 消息。

### 示例 6：完整的训练配置

```bash
llamafactory-cli train \
    --model_name_or_path /data/models/Qwen3-VL-2B-Instruct \
    --dataset "s3://mybucket/gui_data/train.json[media_dir=s3://mybucket/gui_data/images,mask_history_sample=true,max_human_steps=2]" \
    --template qwen3_vl \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_target all \
    --output_dir /data/output \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --cutoff_len 8192 \
    --bf16
```

## 配置优先级

1. **内联配置** (最高优先级) - 在文件路径中通过 `[...]` 指定
2. **dataset_info.json 配置** - 在 dataset_info.json 中定义的数据集配置
3. **全局配置** - 命令行参数（如 `--media_dir`）

## 注意事项

### 1. 相对路径处理
- 如果 `media_dir` 是相对路径，会使用绝对路径 `media_dir + 数据中的相对路径`
- 如果 `media_dir` 是绝对路径或远程路径（`s3://`、`oss://`），直接使用

### 2. 数据中的路径
如果数据中的路径已经是绝对路径或远程路径，`media_dir` 不会拼接：

```json
{
    "image_name": ["/absolute/path/to/image.jpg"]  // 已经是绝对路径，不会拼接
}
```

### 3. 远程路径支持
支持的远程协议：
- `s3://` - Amazon S3
- `oss://` - 阿里云 OSS
- `gs://` / `gcs://` - Google Cloud Storage

需要安装 `megfile` 并配置相应的环境变量：
```bash
pip install megfile

# S3 配置
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_ENDPOINT_URL=https://s3.amazonaws.com  # 或自定义 endpoint

# OSS 配置
export OSS_ACCESS_KEY_ID=your_key
export OSS_SECRET_ACCESS_KEY=your_secret
export OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
```

### 4. 布尔值和数字
- 布尔值使用 `true` / `false`（不区分大小写）
- 数字会自动转换为整数

### 5. 特殊字符
如果值中包含逗号，请避免使用，或者使用 dataset_info.json 配置方式。

## 对比：内联配置 vs dataset_info.json

### 内联配置（推荐用于简单场景）
**优点**:
- 简洁，直接在命令行指定
- 适合快速测试
- 支持远程路径（S3/OSS）

**缺点**:
- 配置较多时命令行会很长
- 不支持复杂的嵌套配置

**示例**:
```bash
--dataset "/mnt/data.json[media_dir=/mnt/images]"
```

### dataset_info.json（推荐用于复杂场景）
**优点**:
- 支持复杂配置
- 配置可复用
- 便于版本管理

**缺点**:
- 需要额外的配置文件
- 配置分散在多个地方

**示例**:
```json
{
    "my_dataset": {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "media_dir": "/mnt/images",
        "columns": {
            "messages": "conversations",
            "images": "image_name"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        }
    }
}
```

## 常见问题

### Q: 内联配置和 dataset_info.json 可以混用吗？
A: 可以！可以在同一个 `--dataset` 参数中混用：
```bash
--dataset "direct_file.json[media_dir=/mnt/img],dataset_from_info"
```

### Q: 如何验证配置是否生效？
A: 查看训练日志，LlamaFactory 会在加载数据集时打印配置信息。

### Q: 支持哪些数据格式？
A: 支持 `.json`、`.jsonl`、`.csv`、`.parquet`、`.arrow` 等格式。

### Q: media_dir 支持多个路径吗？
A: 每个数据集只能指定一个 media_dir。如果需要多个路径，请使用多个数据集配置。

## 更新日志

- 2026-01-09: 添加内联配置语法支持

