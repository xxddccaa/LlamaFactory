# Dataset 内联配置语法 - 快速参考

## 快速开始

### 最简单的用法
```bash
llamafactory-cli train \
    --dataset "/mnt/data.json[media_dir=/mnt/images]" \
    --model_name_or_path /path/to/model
```

### 常用配置示例

#### 1. 指定图片目录
```bash
--dataset "/mnt/data.json[media_dir=/mnt/images]"
```

#### 2. S3 远程数据
```bash
--dataset "s3://bucket/data.json[media_dir=s3://bucket/images]"
```

#### 3. 自定义角色标签
```bash
--dataset "/mnt/data.json[media_dir=/mnt/images,user_tag=user,assistant_tag=assistant]"
```

#### 4. 使用 Alpaca 格式
```bash
--dataset "/mnt/data.json[media_dir=/mnt/images,formatting=alpaca]"
```

#### 5. 启用历史掩码（多轮对话训练）
```bash
--dataset "/mnt/data.json[media_dir=/mnt/images,mask_history_sample=true,max_human_steps=2]"
```

#### 6. 多个数据集混合
```bash
--dataset "/mnt/data1.json[media_dir=/mnt/img1],/mnt/data2.json[media_dir=/mnt/img2]"
```

## 支持的配置项

| 配置项 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `media_dir` | 媒体文件目录 | - | `/mnt/images` |
| `formatting` | 数据格式 | `sharegpt` | `alpaca`, `sharegpt`, `openai` |
| `messages` | 消息字段名 | `conversations` | `messages` |
| `role_tag` | 角色字段名 | `from` | `role` |
| `content_tag` | 内容字段名 | `value` | `content` |
| `user_tag` | 用户标签 | `human` | `user` |
| `assistant_tag` | 助手标签 | `gpt` | `assistant` |
| `images` | 图片字段名 | - | `image_name` |
| `videos` | 视频字段名 | - | `video_name` |
| `audios` | 音频字段名 | - | `audio_name` |
| `mask_history_sample` | 历史掩码 | `false` | `true`, `false` |
| `max_human_steps` | 最大 user 消息数 | `-1` | `2`, `3` |

## 实际使用案例

### 案例 1：训练 GUI Agent（带图片）
```bash
llamafactory-cli train \
    --model_name_or_path /data/models/Qwen3-VL-2B \
    --dataset "s3://mybucket/gui_data/train.json[media_dir=s3://mybucket/gui_images]" \
    --template qwen3_vl \
    --stage sft \
    --do_train \
    --output_dir /data/output \
    --per_device_train_batch_size 4 \
    --bf16
```

**数据格式**:
```json
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "用户任务：打开设置\n当前屏幕：<image>\n请输出操作。"
            },
            {
                "from": "gpt",
                "value": "点击设置图标<|call_start|>click(x=100,y=200)<|call_end|>"
            }
        ],
        "images": ["screenshots/screen_001.jpg"]
    }
]
```

### 案例 2：训练多轮对话（历史掩码）
```bash
llamafactory-cli train \
    --model_name_or_path /data/models/Qwen3-VL-2B \
    --dataset "/mnt/mmdu/train.json[media_dir=/mnt/mmdu/images,mask_history_sample=true,max_human_steps=2]" \
    --template qwen3_vl \
    --stage sft \
    --cutoff_len 8192 \
    --output_dir /data/output
```

**说明**: 
- 每个训练样本最多保留 2 个 user 消息
- 历史 assistant 消息作为上下文保留
- 只对最后一个 assistant 回复计算 loss

### 案例 3：混合本地和远程数据
```bash
llamafactory-cli train \
    --model_name_or_path /data/models/Qwen-7B \
    --dataset "/mnt/local_data.json[media_dir=/mnt/images],s3://bucket/remote_data.json[media_dir=s3://bucket/images]" \
    --output_dir /data/output
```

## 数据路径拼接规则

### 规则 1：相对路径拼接
```bash
media_dir = "/mnt/images"
数据中的路径 = "screenshots/img1.jpg"
实际路径 = "/mnt/images/screenshots/img1.jpg"  # 拼接
```

### 规则 2：绝对路径不拼接
```bash
media_dir = "/mnt/images"
数据中的路径 = "/absolute/path/img1.jpg"
实际路径 = "/absolute/path/img1.jpg"  # 不拼接
```

### 规则 3：远程路径不拼接
```bash
media_dir = "/mnt/images"
数据中的路径 = "s3://bucket/img1.jpg"
实际路径 = "s3://bucket/img1.jpg"  # 不拼接
```

## 对比传统方式

### 传统方式（使用 dataset_info.json）

**步骤 1**: 创建 `/mnt/dataset_dir/dataset_info.json`
```json
{
    "my_dataset": {
        "file_name": "train.json",
        "media_dir": "/mnt/images",
        "formatting": "sharegpt"
    }
}
```

**步骤 2**: 运行训练
```bash
llamafactory-cli train \
    --dataset my_dataset \
    --dataset_dir /mnt/dataset_dir \
    --model_name_or_path /path/to/model
```

### 新方式（内联配置）

**一步搞定**:
```bash
llamafactory-cli train \
    --dataset "/mnt/dataset_dir/train.json[media_dir=/mnt/images,formatting=sharegpt]" \
    --model_name_or_path /path/to/model
```

**优势**:
- ✅ 无需创建 dataset_info.json
- ✅ 命令行直观清晰
- ✅ 适合快速实验
- ✅ 支持远程路径（S3/OSS）

## 常见错误及解决

### 错误 1: mask_history_sample 配置不完整
```bash
# ❌ 错误
--dataset "/mnt/data.json[mask_history_sample=true]"

# ✅ 正确
--dataset "/mnt/data.json[mask_history_sample=true,max_human_steps=2]"
```

**原因**: `mask_history_sample` 和 `max_human_steps` 必须同时设置。

### 错误 2: 图片路径找不到
```bash
# 检查 media_dir 是否正确
media_dir = "/mnt/images"
数据中 = ["img.jpg"]
实际查找 = "/mnt/images/img.jpg"  # 确保此文件存在
```

### 错误 3: S3 访问失败
```bash
# 配置环境变量
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_ENDPOINT_URL=https://your-s3-endpoint.com

# 安装依赖
pip install megfile
```

## 进阶技巧

### 技巧 1: 使用环境变量
```bash
export MEDIA_DIR="/mnt/shared/images"
llamafactory-cli train \
    --dataset "/mnt/data.json[media_dir=$MEDIA_DIR]" \
    --model_name_or_path /path/to/model
```

### 技巧 2: 脚本批量处理
```bash
#!/bin/bash
for data_file in /mnt/datasets/*.json; do
    llamafactory-cli train \
        --dataset "${data_file}[media_dir=/mnt/images]" \
        --model_name_or_path /path/to/model \
        --output_dir /data/output_$(basename $data_file .json)
done
```

### 技巧 3: 验证配置
```bash
# 使用 --overwrite_cache true 强制重新加载数据
llamafactory-cli train \
    --dataset "/mnt/data.json[media_dir=/mnt/images]" \
    --overwrite_cache true \
    --model_name_or_path /path/to/model
```

## 相关链接

- 完整文档: [dataset_inline_config.md](dataset_inline_config.md)
- 数据格式说明: [data/README.md](../data/README.md)
- 模板说明: [templates/README.md](../templates/README.md)

