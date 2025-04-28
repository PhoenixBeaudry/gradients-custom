import asyncio # Import asyncio
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from transformers import AutoConfig
import docker
import pandas as pd
import toml
import yaml
from docker.errors import DockerException
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi
from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import save_config_toml
from core.config.config_handler import update_flash_attention
from core.config.config_handler import update_model_info
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.docker_utils import stream_logs
from core.utils import download_s3_file # Import the download utility
from core.models.utility_models import DiffusionJob
from core.models.utility_models import DPODatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import InstructDatasetType
from core.models.utility_models import TextJob
from core.models.utility_models import ImageModelType
from miner.utils import download_flux_unet


logger = get_logger(__name__)


@dataclass
class DockerEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str
    base_model: str

    def to_dict(self) -> dict[str, str]:
        return {"HUGGINGFACE_TOKEN": self.huggingface_token, "WANDB_TOKEN": self.wandb_token, "JOB_ID": self.job_id, "BASE_MODEL": self.base_model}


@dataclass
class DockerEnvironment:
    huggingface_token: str
    wandb_token: str
    job_id: str
    dataset_type: str
    dataset_filename: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }


def _load_and_modify_config(
    dataset: str,
    model: str,
    hours_to_complete: int,
    dataset_type: InstructDatasetType | DPODatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    if isinstance(dataset_type, DPODatasetType):
        config["rl"] = "dpo"

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name)
    hf_cfg = AutoConfig.from_pretrained(model)
 
    max_pos = getattr(hf_cfg, "max_position_embeddings", None) or getattr(hf_cfg, "n_ctx", None)

    # clamp sequence_len to the model’s max
    desired_len = 8196
    if max_pos is not None and desired_len > max_pos:
        logger.warning(f"Requested seq_len={desired_len} > model max {max_pos}; falling back to {max_pos}")
        config["sequence_len"] = max_pos
        logger.info(f"Sequence Length set to: {max_pos}")
    else:
        config["sequence_len"] = desired_len

    config["mlflow_experiment_name"] = dataset
    config["hours_to_complete"] = hours_to_complete

    return config


def _load_and_modify_config_diffusion(job: DiffusionJob) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    if job.model_type == ImageModelType.SDXL:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = job.model
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    elif job.model_type == ImageModelType.FLUX:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = f"{cst.CONTAINER_FLUX_PATH}/flux_unet_{job.model.replace('/', '_')}.safetensors"
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    else:
        logger.error(f"Unknown model type: {job.model_type}")
    return config


def create_job_diffusion(
    job_id: str,
    model: str,
    dataset_zip: str,
    model_type: ImageModelType,
    expected_repo_name: str | None
):
    return DiffusionJob(job_id=job_id, model=model, dataset_zip=dataset_zip, model_type=model_type, expected_repo_name=expected_repo_name)


def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: InstructDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )


def start_tuning_container_diffusion(job: DiffusionJob, hours_to_complete: int):
    logger.info("=" * 80)
    logger.info("STARTING THE DIFFUSION TUNING CONTAINER")
    logger.info("=" * 80)

    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(job)
    save_config_toml(config, config_path)

    logger.info(config)
    if job.model_type == ImageModelType.FLUX:
        logger.info(f"Downloading flux unet from {job.model}")
        flux_unet_path = download_flux_unet(job.model)

    # Download the dataset zip file using the URI stored in the job object
    logger.info(f"Downloading dataset zip from URI: {job.dataset_zip}")
    try:
        # Define a local path for the downloaded zip
        local_zip_path = os.path.join(cst.DIFFUSION_DATASET_DIR, f"{job.job_id}_downloaded.zip")
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)
        
        # Perform the download (assuming download_s3_file is async, but job handler is sync)
        # NOTE: If download_s3_file is async, this needs adjustment (e.g., run in event loop or make job handler async)
        # For now, assuming it can be called synchronously or we adapt it.
        # If it MUST be async, we'd need asyncio.run() or similar here.
        # Let's assume for now it's adapted or can work synchronously for simplicity.
        # If download_s3_file is strictly async, this will need `asyncio.run(download_s3_file(...))` # Confirmed async needed
        # Run the async download function synchronously using asyncio.run()
        downloaded_local_zip_path = asyncio.run(download_s3_file(job.dataset_zip, local_zip_path))
        logger.info(f"Dataset zip downloaded to: {downloaded_local_zip_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset zip from {job.dataset_zip}: {e}")
        raise # Re-raise the exception to fail the job

    prepare_dataset(
        training_images_zip_path=downloaded_local_zip_path, # Use the downloaded path
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if job.model_type == ImageModelType.SDXL else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    docker_env = DockerEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id, base_model=job.model_type.value
    ).to_dict()

    # Get assigned GPUs from worker environment
    assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned_gpus:
        logger.info(f"Worker assigned GPUs: {assigned_gpus}")
        # Pass GPU assignment into the container environment
        docker_env["CUDA_VISIBLE_DEVICES"] = assigned_gpus
        # Revert device_requests to allow container to see all assigned GPUs, rely on env var inside.
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
    else:
        logger.warning("CUDA_VISIBLE_DEVICES not set for worker, container will see all GPUs.")
        # Default: request all GPUs if not specified
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    logger.info(f"Docker environment: {docker_env}")
    logger.info(f"Docker device requests: {device_requests}")


    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/dataset/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/dataset/outputs",
                "mode": "rw",
            },
            os.path.abspath(cst.DIFFUSION_DATASET_DIR): {
                "bind": "/dataset/images",
                "mode": "rw",
            },
        }

        if job.model_type == ImageModelType.FLUX:
            volume_bindings[os.path.dirname(flux_unet_path)] =  {
                "bind": cst.CONTAINER_FLUX_PATH,
                "mode": "rw",
            }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE_DIFFUSION,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            ulimits=[
                docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
                docker.types.Ulimit(name="stack",  soft=67108864, hard=67108864),
            ],
            shm_size="32g",
            device_requests=device_requests, # Use specific GPUs if assigned
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function
        stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if "container" in locals():
            try:
                container.remove(force=True)
                logger.info(f"Removed container for job {job.job_id}")
            except Exception as e:
                 logger.warning(f"Failed to remove container for job {job.job_id}: {e}")


        # Clean up the specific downloaded zip file if it exists
        if 'downloaded_local_zip_path' in locals() and os.path.exists(downloaded_local_zip_path):
             try:
                 os.remove(downloaded_local_zip_path)
                 logger.info(f"Removed downloaded zip: {downloaded_local_zip_path}")
             except Exception as e:
                 logger.warning(f"Failed to remove downloaded zip {downloaded_local_zip_path}: {e}")

        # Clean up the extracted training data directory
        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"
        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)


def _dpo_format_prompt(row, format_str):
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DPODatasetType, apply_formatting: bool = False):
    """
    Transform a DPO JSON dataset file to match axolotl's `chatml.argilla` expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DPODatasetType with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    column_mapping = {
        dataset_type.field_prompt: cst.DPO_DEFAULT_FIELD_PROMPT,
        dataset_type.field_system: cst.DPO_DEFAULT_FIELD_SYSTEM,
        dataset_type.field_chosen: cst.DPO_DEFAULT_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.DPO_DEFAULT_FIELD_REJECTED
    }
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
            format_str = dataset_type.prompt_format
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(lambda row: _dpo_format_prompt(row, format_str), axis=1)
        if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
            format_str = dataset_type.chosen_format
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(lambda row: _dpo_format_chosen(row, format_str), axis=1)
        if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
            format_str = dataset_type.rejected_format
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(lambda row: _dpo_format_rejected(row, format_str), axis=1)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def start_tuning_container(job: TextJob, hours_to_complete: int):
    logger.info("=" * 80)
    logger.info("STARTING THE TUNING CONTAINER")
    logger.info("=" * 80)

    # Prepare config file
    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)
    config = _load_and_modify_config(
        job.dataset,
        job.model,
        hours_to_complete,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )
    save_config(config, config_path)

    logger.info(config)
    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    # Build environment dict (HuggingFace, W&B, etc.)
    docker_env = DockerEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=cst.CUSTOM_DATASET_TYPE,
        dataset_filename=os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    ).to_dict()

    # GPU assignment
    assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned_gpus:
        logger.info(f"Worker assigned GPUs: {assigned_gpus}")
        docker_env["CUDA_VISIBLE_DEVICES"] = assigned_gpus
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
    else:
        logger.warning("CUDA_VISIBLE_DEVICES not set; container will see all GPUs.")
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    logger.info(f"Docker environment: {docker_env}")
    logger.info(f"Docker device requests: {device_requests}")

    try:
        docker_client = docker.from_env()

        # Volume mounts: configs, outputs, HF cache, input_data
        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/outputs",
                "mode": "rw",
            },
            os.path.expanduser("~/.cache/huggingface"): {
                "bind": "/root/.cache/huggingface",
                "mode": "rw",
            },
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            volume_bindings[dataset_dir] = {
                "bind": "/workspace/input_data",
                "mode": "rw",
            }

        # Launch training container
        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            ulimits=[
                docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
                docker.types.Ulimit(name="stack", soft=67108864, hard=67108864),
            ],
            shm_size="64g",
            device_requests=device_requests,
            detach=True,
            tty=True,
        )

        stream_logs(container)
        result = container.wait()
        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {e}")
        raise

    finally:
        # Make HF repo public if created
        repo = config.get("hub_model_id", None)
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_settings(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")
        if 'container' in locals():
            try:
                container.remove(force=True)
                logger.info("Container removed")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
