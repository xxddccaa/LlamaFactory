

联网机器：
```
NAME=kevinchina/deeplearning:llamafactory0-9-4-base-1-megatron-1-ok
docker build -f ./docker/docker-cuda/Dockerfile2.megatron \
    -t ${NAME} .
docker push ${NAME}
```

内网机器：
```
NAME=kevinchina/deeplearning:llamafactory0-9-4-base-1-megatron-1-ok
NAME2=hub.i.x.com/g-xiedong-fine/llamafactory0-9-4-base-1-megatron-1-ok:v1
docker pull ${NAME}
docker tag ${NAME} ${NAME2}
docker push ${NAME2}
```

sanlab key：
```
export SWANLAB_API_KEY=pM7Xvs5OS2EeXPO5gKXfJ   # 设置在线跟踪模式API，这里我随便填的
export SWANLAB_LOG_DIR=/swanlab_log    # 设置本地日志存储路径
export SWANLAB_MODE=cloud     # 包含四种模式：cloud云端跟踪模式（默认）、cloud-only仅云端跟踪本地不保存文件、local本地跟踪模式、disabled完全不记录用于debug
```

训练：
```bash
USE_MCA=1 llamafactory-cli train /mnt/s3fs/train-LlamaFactory/examples/megatron/qwen3_vl_2b_full.yaml
```

导出权重：
```
python scripts/megatron_merge.py \
    --checkpoint_path saves/mca/qwen3_vl_test \
    --output_path ./qwen3_vl_hf_model \
    --bf16 true
```
