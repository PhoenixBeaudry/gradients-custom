FROM --platform=linux/amd64 diagonalge/kohya_latest:latest

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images && \
    chmod -R 777 /dataset

ENV CONFIG_DIR="/dataset/configs"
ENV OUTPUT_DIR="/dataset/outputs"
ENV DATASET_DIR="/dataset/images"

RUN export NCCL_NVLS_ENABLE=1          # turn on the fast collective on DGX‑H100
RUN export NCCL_P2P_DISABLE=0          # make sure P2P over NVLink is allowed
RUN export NCCL_LAUNCH_MODE=GROUP      # fewer CUDA context switches


CMD accelerate launch --dynamo_backend inductor --dynamo_mode reduce-overhead --mixed_precision bf16 /app/sd-scripts/${BASE_MODEL}_train_network.py --config_file ${CONFIG_DIR}/${JOB_ID}.toml
