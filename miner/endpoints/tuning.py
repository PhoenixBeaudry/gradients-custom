import os
from datetime import datetime
from datetime import timedelta

import toml
import yaml
import redis
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_get_request
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError
from rq import Queue
from rq.job import Job # Correct import for Job class
from rq.registry import StartedJobRegistry, FailedJobRegistry
from rq.exceptions import NoSuchJobError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
# from miner.config import WorkerConfig # Removed
# from miner.dependencies import get_worker_config # Removed
from miner.logic.job_handler import create_job_diffusion
from miner.logic.job_handler import create_job_text
from miner.logic.job_handler import start_tuning_container, start_tuning_container_diffusion # Import job functions


logger = get_logger(__name__)

# Connect to Redis and initialize RQ Queue
redis_conn = redis.Redis(
    host=cst.REDIS_HOST,
    port=cst.REDIS_PORT,
    password=cst.REDIS_PASSWORD, # Add password from constants
    db=0
    # decode_responses=True # REMOVED - RQ expects raw bytes for pickled job data
)
rq_queue = Queue(connection=redis_conn)


async def tune_model_text(
    train_request: TrainRequestText,
    # worker_config: WorkerConfig = Depends(get_worker_config), # Removed
):
    # global current_job_finish_time # Removed
    logger.info("Starting model tuning.")

    # current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete) # Removed
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    # worker_config.trainer.enqueue_job(job) # Replaced with RQ
    rq_job = rq_queue.enqueue(
        start_tuning_container,
        job,
        train_request.hours_to_complete,
        job_timeout=int(train_request.hours_to_complete * 3600 * 1.1), # Add timeout buffer
        result_ttl=86400, # Keep result for 1 day
        failure_ttl=86400  # Keep failure info for 1 day
    )
    logger.info(f"Enqueued job {rq_job.id} to RQ")

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_diffusion(
    train_request: TrainRequestImage,
    # worker_config: WorkerConfig = Depends(get_worker_config), # Removed
):
    # global current_job_finish_time # Removed
    logger.info("Starting model tuning.")

    # current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete) # Removed
    logger.info(f"Job received is {train_request}")
    # try: # Remove pre-download
    #     train_request.dataset_zip = await download_s3_file(
    #         train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
    #     )
    #     logger.info(train_request.dataset_zip)
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))

    # Pass the original dataset_zip URI (S3 path) to the job object
    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip, # Pass URI directly
        model=train_request.model,
        model_type=train_request.model_type,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    # worker_config.trainer.enqueue_job(job) # Replaced with RQ
    rq_job = rq_queue.enqueue(
        start_tuning_container_diffusion,
        job,
        train_request.hours_to_complete,
        job_timeout=int(train_request.hours_to_complete * 3600 * 1.1), # Add timeout buffer
        result_ttl=86400, # Keep result for 1 day
        failure_ttl=86400  # Keep failure info for 1 day
    )
    logger.info(f"Enqueued job {rq_job.id} to RQ")

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def get_latest_model_submission(task_id: str) -> str:
    try:
        # Temporary work around in order to not change the vali a lot
        # Could send the task type from vali instead of matching file names
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                return config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                return config_data.get("huggingface_repo_id", None)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    # worker_config: WorkerConfig = Depends(get_worker_config), # Removed
) -> MinerTaskResponse:
    try:
        logger.info("An offer has come through")
        logger.info(f"Model: {request.model.lower()}, Time: {request.hours_to_complete}")
        if request.task_type == TaskType.INSTRUCTTEXTTASK:
            logger.info("Task Type: Instruct")
        if request.task_type == TaskType.DPOTASK:
            logger.info("Task Type: DPO")

        if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK]:
            return MinerTaskResponse(
                message=f"This endpoint only accepts text tasks: "
                        f"{TaskType.INSTRUCTTEXTTASK} and {TaskType.DPOTASK}",
                accepted=False
            )

        # Check model parameter count
        # Reject if model size is 32B or larger
        if request.model_params_count is not None and request.model_params_count >= 32_000_000_000:
            logger.info(f"Rejecting offer: Model size too large ({request.model_params_count / 1_000_000_000:.1f}B >= 32B)")
            return MinerTaskResponse(message="Model size too large (>= 32B)", accepted=False)

        # Check RQ queue length and running jobs
        queued_count = rq_queue.count
        started_registry = StartedJobRegistry(queue=rq_queue)
        running_count = started_registry.count
        total_active = queued_count + running_count
        capacity = 4 # TODO: Make this configurable?

        if total_active >= capacity: # Keep existing buffer logic
            logger.info(f"Rejecting offer: Queue full (queued={queued_count}, running={running_count}, total={total_active})")
            return MinerTaskResponse(message=f"Queue full ({total_active})", accepted=False)

        # optional: still reject absurdly long jobs if you want
        if request.hours_to_complete >= 48:
            logger.info(f"Rejecting offer: too long ({request.hours_to_complete}h)")
            return MinerTaskResponse(message="Job too long", accepted=False)

        # otherwise accept
        logger.info(f"Accepting offer ({total_active+1}/{capacity}): {request.model} ({request.hours_to_complete}h)")
        return MinerTaskResponse(message="-----:)-----", accepted=True)

    except ValidationError as e:
        logger.error(f"Validation error in task_offer: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    # worker_config: WorkerConfig = Depends(get_worker_config), # Removed
) -> MinerTaskResponse:
    try:
        logger.info("An image offer has come through")
        ####### TEMPORARY #######
        # WE ARE CURRENTLY NOT ACCEPTING IMAGE TASKS
        logger.info(f"Rejecting offer for image task.)")
        return MinerTaskResponse(message="No image tasks for now :(", accepted=False)

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def requeue_job(job_id: str):
    """
    Requeue a job that is either failed or finished.
    """
    try:
        # Fetch the job by ID using the connection
        job = Job.fetch(job_id, connection=redis_conn)

        # Check if the job is failed or finished
        if job.is_failed or job.is_finished:
            original_status = "failed" if job.is_failed else "finished"
            logger.info(f"Requeuing {original_status} job {job_id}")
            # job.requeue() resets status and puts it back in the queue
            job.requeue()
            return {"message": f"{original_status.capitalize()} job {job_id} successfully requeued."}
        else:
            # If job exists but isn't failed or finished, report its status
            current_status = job.get_status()
            logger.warning(f"Job {job_id} found but is not failed or finished (Status: {current_status}). Cannot requeue.")
            raise HTTPException(
                status_code=409, # Conflict status code
                detail=f"Job {job_id} exists but is not failed or finished (Status: {current_status}). Only failed or finished jobs can be requeued."
            )

    except NoSuchJobError:
        # Handle case where job ID doesn't exist in Redis at all
        logger.error(f"Job {job_id} not found in Redis.")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    except Exception as e:
        logger.error(f"Error processing requeue for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing requeue for job {job_id}: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake)],
    )
    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )
    
    # Add route for requeueing jobs
    router.add_api_route(
        "/requeue_job/{job_id}", # Renamed route
        requeue_job,             # Renamed function
        tags=["Admin"],
        methods=["POST"],
        summary="Requeue a Job", # Updated summary
        description="Requeue a job that is either failed or finished using its ID.", # Updated description
        # Consider adding authentication/authorization dependency here for production
    )

    return router
