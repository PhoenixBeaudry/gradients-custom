import glob
import json
import os
from typing import List, Tuple

from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob, TextJob, Job, JobStatus
from core.models.utility_models import InstructDatasetType, DPODatasetType, ImageModelType

logger = get_logger(__name__)


def job_to_dict(job: Job) -> dict:
    """
    Convert a Job object to a dictionary for serialization.
    
    Args:
        job: The Job object to convert
        
    Returns:
        A dictionary representation of the job
        
    Raises:
        ValueError: If the job type is not supported
    """
    if isinstance(job, TextJob):
        # For TextJob, we need to handle the dataset_type specially
        dataset_type = job.dataset_type
        if isinstance(dataset_type, InstructDatasetType):
            dataset_type_dict = {
                "system_prompt": dataset_type.system_prompt,
                "system_format": dataset_type.system_format,
                "field_system": dataset_type.field_system,
                "field_instruction": dataset_type.field_instruction,
                "field_input": dataset_type.field_input,
                "field_output": dataset_type.field_output,
                "format": dataset_type.format,
                "no_input_format": dataset_type.no_input_format,
                "field": dataset_type.field,
            }
        elif isinstance(dataset_type, DPODatasetType):
            dataset_type_dict = {
                "field_prompt": dataset_type.field_prompt,
                "field_system": dataset_type.field_system,
                "field_chosen": dataset_type.field_chosen,
                "field_rejected": dataset_type.field_rejected,
                "prompt_format": dataset_type.prompt_format,
                "chosen_format": dataset_type.chosen_format,
                "rejected_format": dataset_type.rejected_format,
            }
        else:
            raise ValueError(f"Unsupported dataset_type: {type(dataset_type)}")
            
        return {
            "type": "TextJob",
            "job_id": job.job_id,
            "model": job.model,
            "status": job.status,
            "error_message": job.error_message,
            "expected_repo_name": job.expected_repo_name,
            "dataset": job.dataset,
            "dataset_type": dataset_type_dict,
            "file_format": job.file_format,
        }
    elif isinstance(job, DiffusionJob):
        return {
            "type": "DiffusionJob",
            "job_id": job.job_id,
            "model": job.model,
            "status": job.status,
            "error_message": job.error_message,
            "expected_repo_name": job.expected_repo_name,
            "dataset_zip": job.dataset_zip,
            "model_type": job.model_type,
        }
    else:
        raise ValueError(f"Unsupported job type: {type(job)}")


def dict_to_job(job_dict: dict) -> Job:
    """
    Convert a dictionary to a Job object.
    
    Args:
        job_dict: The dictionary to convert
        
    Returns:
        A Job object
        
    Raises:
        ValueError: If the job type is not supported
    """
    job_type = job_dict.pop("type")
    
    # Convert status from string to enum if it's a string
    if "status" in job_dict and isinstance(job_dict["status"], str):
        job_dict["status"] = JobStatus(job_dict["status"])
        
    if job_type == "TextJob":
        dataset_type_dict = job_dict.pop("dataset_type")
        
        # Determine if it's an InstructDatasetType or DPODatasetType
        if "field_prompt" in dataset_type_dict:
            dataset_type = DPODatasetType(**dataset_type_dict)
        else:
            dataset_type = InstructDatasetType(**dataset_type_dict)
            
        job_dict["dataset_type"] = dataset_type
        return TextJob(**job_dict)
    elif job_type == "DiffusionJob":
        # Convert model_type from string to enum if it's a string
        if "model_type" in job_dict and isinstance(job_dict["model_type"], str):
            job_dict["model_type"] = ImageModelType(job_dict["model_type"])
            
        return DiffusionJob(**job_dict)
    else:
        raise ValueError(f"Unsupported job type: {job_type}")


class JobPersistenceManager:
    """
    Manages the persistence of jobs to the file system.
    
    This class is responsible for:
    - Serializing/deserializing jobs to/from JSON
    - Writing/reading jobs to/from files
    - Managing the queue order file
    """
    
    def __init__(self, persistence_dir: str):
        """
        Initialize the JobPersistenceManager.
        
        Args:
            persistence_dir: The directory where jobs and queue state will be persisted
        """
        self.persistence_dir = persistence_dir
        self.jobs_dir = os.path.join(persistence_dir, "jobs")
        self.queue_file = os.path.join(persistence_dir, "queue.json")
        
        # Create directories if they don't exist
        os.makedirs(self.jobs_dir, exist_ok=True)
        
    def persist_job(self, job: Job) -> None:
        """
        Serialize a job to JSON and write it to a file.
        
        Args:
            job: The job to persist
        """
        try:
            job_dict = job_to_dict(job)
            job_file = os.path.join(self.jobs_dir, f"{job.job_id}.json")
            
            with open(job_file, "w") as f:
                json.dump(job_dict, f, indent=2)
                
            logger.debug(f"Persisted job {job.job_id} to {job_file}")
        except Exception as e:
            logger.error(f"Error persisting job {job.job_id}: {str(e)}")
            raise
        
    def load_job(self, job_id: str) -> Job:
        """
        Read a job from a file and deserialize it.
        
        Args:
            job_id: The ID of the job to load
            
        Returns:
            The deserialized job, or None if the job could not be loaded
        """
        job_file = os.path.join(self.jobs_dir, f"{job_id}.json")
        
        try:
            with open(job_file, "r") as f:
                job_dict = json.load(f)
                return dict_to_job(job_dict)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading job {job_id}: {str(e)}")
            return None
        
    def delete_job(self, job_id: str) -> None:
        """
        Delete a job file.
        
        Args:
            job_id: The ID of the job to delete
        """
        job_file = os.path.join(self.jobs_dir, f"{job_id}.json")
        
        try:
            os.remove(job_file)
            logger.debug(f"Deleted job file for {job_id}")
        except FileNotFoundError:
            logger.warning(f"Job file not found for deletion: {job_id}")
        
    def persist_queue(self, running_jobs: List[str], queued_jobs: List[str]) -> None:
        """
        Write the queue order to a file.
        
        Args:
            running_jobs: List of job IDs that are currently running
            queued_jobs: List of job IDs that are currently queued
        """
        try:
            queue_data = {
                "running": running_jobs,
                "queued": queued_jobs
            }
            
            with open(self.queue_file, "w") as f:
                json.dump(queue_data, f, indent=2)
                
            logger.debug(f"Persisted queue state: {len(running_jobs)} running, {len(queued_jobs)} queued")
        except Exception as e:
            logger.error(f"Error persisting queue state: {str(e)}")
            raise
        
    def load_queue(self) -> Tuple[List[str], List[str]]:
        """
        Read the queue order from a file.
        
        Returns:
            A tuple of (running_jobs, queued_jobs)
        """
        try:
            with open(self.queue_file, "r") as f:
                queue_data = json.load(f)
                running_jobs = queue_data.get("running", [])
                queued_jobs = queue_data.get("queued", [])
                logger.debug(f"Loaded queue state: {len(running_jobs)} running, {len(queued_jobs)} queued")
                return running_jobs, queued_jobs
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading queue state: {str(e)}")
            return [], []
        
    def list_jobs(self) -> List[str]:
        """
        List all job files.
        
        Returns:
            A list of job IDs
        """
        job_files = glob.glob(os.path.join(self.jobs_dir, "*.json"))
        job_ids = [os.path.basename(f).replace(".json", "") for f in job_files]
        logger.debug(f"Found {len(job_ids)} job files")
        return job_ids