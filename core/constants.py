import os

from dotenv import load_dotenv


load_dotenv()

VERSION_KEY = 61_000
# Default NETUID if not set in environment
DEFAULT_NETUID = 56

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

MINER_DOCKER_IMAGE = "phoenixbeaudry/god-text-miner:custom"
MINER_DOCKER_IMAGE_DIFFUSION = "phoenixbeaudry/god-text-miner-diffusion:custom"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"
VALIDATOR_DOCKER_IMAGE_DPO = "weightswandering/tuning_vali_dpo:latest"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "diagonalge/tuning_validator_diffusion:latest"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "core/config/"
OUTPUT_DIR = "core/outputs/"
CACHE_DIR = "~/.cache/huggingface"
CACHE_DIR_HUB = os.path.expanduser("~/.cache/huggingface/hub")
DIFFUSION_DATASET_DIR = "core/dataset/images"
CONTAINER_FLUX_PATH = "/app/flux/unet"
JOB_BACKUP_DIR = "core/job_backups/"

DIFFUSION_SDXL_REPEATS = 10
DIFFUSION_FLUX_REPEATS = 1
DIFFUSION_DEFAULT_INSTANCE_PROMPT = "lora"
DIFFUSION_DEFAULT_CLASS_PROMPT = "style"

MIN_IMAGE_TEXT_PAIRS = 10
MAX_IMAGE_TEXT_PAIRS = 50

CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL = CONFIG_DIR + "base_diffusion_sdxl.toml"
CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX = CONFIG_DIR + "base_diffusion_flux.toml"


CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") # Add Redis password

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
CUSTOM_DATASET_TYPE = "custom"

# DPO default dataset type
DPO_DEFAULT_DATASET_TYPE = "chatml.intel"
DPO_DEFAULT_FIELD_PROMPT = "question"
DPO_DEFAULT_FIELD_SYSTEM = "system"
DPO_DEFAULT_FIELD_CHOSEN = "chosen"
DPO_DEFAULT_FIELD_REJECTED = "rejected"
