


开发：
```
NAME=kevinchina/deeplearning:llamafactory0-9-4-base-1-megatron-1-ok-dev
docker build -f ./docker/docker-cuda/Dockerfile2.megatron -t ${NAME} .
docker run -it --gpus all \
  -v /data/xiedong/:/data/xiedong/ \
  --shm-size=16g \
  ${NAME} bash
cd /data/xiedong/LlamaFactory
pip install -e .
```
这意味着代码本身在容器里，修改本工程后，执行类似的指令docker exec xxxx就可以用容器环境执行代码。

数据集：
```bash
./hfd.sh laolao77/MMDU --dataset --local-dir /data/xiedong/mmdu/
```

```bash
USE_MCA=1 llamafactory-cli train examples/megatron/qwen3_vl_mmdu_full.yaml
```

```bash
llamafactory-cli train examples/megatron/qwen3_vl_mmdu_full.yaml
```



```
NAME=kevinchina/deeplearning:llamafactory0-9-4-base-1-megatron-1-ok-tars2
docker build -f ./docker/docker-cuda/Dockerfile2.megatron -t ${NAME} .
docker push ${NAME} 
```

