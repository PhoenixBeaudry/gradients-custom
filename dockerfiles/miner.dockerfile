FROM --platform=linux/amd64 axolotlai/axolotl:main-20241128-py3.11-cu124-2.5.1

# Install dependencies and Unsloth
RUN pip install --upgrade pip setuptools wheel && \
    pip install mlflow protobuf huggingface_hub wandb && \
    pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/outputs /workspace/data /workspace/input_data

ENV CONFIG_DIR="/workspace/configs"
ENV OUTPUT_DIR="/workspace/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233bcef0e60addba6

RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

# Copy the Unsloth launcher
COPY dockerfiles/unsloth_train.py /workspace/unsloth_train.py

CMD echo 'Preparing data...' && \
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \
    echo "Attempting to log in to Hugging Face" && \
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \
    else \
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."; \
    fi && \
    if [ -n "$WANDB_TOKEN" ]; then \
    echo "Attempting to log in to W&B" && \
    wandb login "$WANDB_TOKEN"; \
    else \
    echo "WANDB_TOKEN is not set. Skipping W&B login."; \
    fi && \
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/data/${DATASET_FILENAME}; \
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \
    fi && \
    echo 'Starting training command' && \
    python3 /workspace/unsloth_train.py --config ${CONFIG_DIR}/${JOB_ID}.yml
