FROM diagonalge/kohya_latest:latest

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images && \
    chmod -R 777 /dataset

ENV CONFIG_DIR="/dataset/configs"
ENV OUTPUT_DIR="/dataset/outputs"
ENV DATASET_DIR="/dataset/images"

RUN pip install flash-attn --no-build-isolation

CMD accelerate launch --dynamo_backend inductor --dynamo_mode reduce-overhead --mixed_precision bf16 --multi_gpu /app/sd-scripts/${BASE_MODEL}_train_network.py --config_file ${CONFIG_DIR}/${JOB_ID}.toml