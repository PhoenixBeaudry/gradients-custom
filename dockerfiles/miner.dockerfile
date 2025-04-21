FROM --platform=linux/amd64 axolotlai/axolotl:main-latest

# --- install additional deps ---
RUN pip install mlflow huggingface_hub wandb protobuf pyyaml

# --- directory setup ---
WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=d49fdd0cc9750a097b58ba35b2d9fbed
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=02e398474b783af6ded4c4638b5388ceb8079c83bb2f8233d5bcef0e60addba6
RUN export NCCL_NVLS_ENABLE=1          # turn on the fast collective on DGX‑H100
RUN export NCCL_P2P_DISABLE=0          # make sure P2P over NVLink is allowed
RUN export NCCL_LAUNCH_MODE=GROUP      # fewer CUDA context switches


# dummy AWS config for libraries that expect it
RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

# --- entrypoint ---
CMD echo '🔧 Preparing environment...' && \
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \
        echo "🔐 Logging into Hugging Face" && \
        huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \
    else \
        echo "⚠️ HUGGINGFACE_TOKEN is not set. Skipping login."; \
    fi && \
    if [ -n "$WANDB_TOKEN" ]; then \
        echo "🔐 Logging into W&B" && \
        wandb login "$WANDB_TOKEN"; \
    else \
        echo "⚠️ WANDB_TOKEN is not set. Skipping login."; \
    fi && \
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \
        cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/data/${DATASET_FILENAME}; \
        cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \
    fi && \
    echo '🚀 Starting training (with push_to_hub: false)' && \
    accelerate launch --multi_gpu -m axolotl.cli.train ${CONFIG_DIR}/${JOB_ID}.yml && \
    echo '🧼 Cleaning README.md metadata...' && \
    sed -i '/^datasets:/,+1d' ${OUTPUT_DIR}/*/README.md && \
    echo '📦 Extracting hub_model_id from config...' && \
    export HF_REPO_ID=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_DIR}/${JOB_ID}.yml'))['hub_model_id'])") && \
    echo "🌍 Target HF model repo: $HF_REPO_ID" && \
    huggingface-cli repo create "$HF_REPO_ID" --type model --private -y && \
    echo '⬆️ Uploading to Hugging Face Hub...' && \
    huggingface-cli upload ${OUTPUT_DIR}/* "$HF_REPO_ID" --include '*.bin' '*.json' '*.md' '*.txt' && \
    echo '✅ Upload complete.'
