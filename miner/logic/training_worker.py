import json
import os
import queue
import threading
from uuid import UUID

import docker
from fiber.logging_utils import get_logger

from core import constants as cst
from core.models.utility_models import DiffusionJob, TextJob, Job, JobStatus
from miner.logic.job_handler import (
    start_tuning_container,
    start_tuning_container_diffusion,
)

logger = get_logger(__name__)


def get_job_backup_filepath(job: Job) -> str:
    """
    Generate the backup filepath for a job.
    
    Args:
        job: The job to generate a filepath for
        
    Returns:
        The filepath where the job backup should be stored
    """
    # Sanitize the model name to create a valid filename
    # Replace slashes and other invalid characters
    sanitized_model = job.model.replace('/', '_').replace('\\', '_')
    
    # Create a filename with job_id and model name
    filename = f"{job.job_id}_{sanitized_model}.json"
    return os.path.join(cst.JOB_BACKUP_DIR, filename)


def save_job_to_disk(job: Job) -> None:
    """
    Save a job to disk using its model name as part of the filename.
    This provides a backup in case the job fails during processing.
    
    Args:
        job: The job to save
    """
    # Ensure the backup directory exists
    os.makedirs(cst.JOB_BACKUP_DIR, exist_ok=True)
    
    filepath = get_job_backup_filepath(job)
    
    # Convert the job to a dictionary
    job_dict = job.dict()
    
    # Serialize the job to JSON and save it
    with open(filepath, 'w') as f:
        json.dump(job_dict, indent=2, fp=f)
        
    logger.info(f"Saved job backup to {filepath}")


def delete_job_backup(job: Job) -> None:
    """
    Delete the backup file for a job.
    
    Args:
        job: The job whose backup should be deleted
    """
    filepath = get_job_backup_filepath(job)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"Deleted job backup at {filepath}")
    else:
        logger.warning(f"No backup file found at {filepath}")


def load_job_backups() -> list[Job]:
    """
    Load all job backups from disk.
    
    Returns:
        A list of Job objects loaded from backup files
    """
    jobs = []
    
    # Ensure the backup directory exists
    if not os.path.exists(cst.JOB_BACKUP_DIR):
        os.makedirs(cst.JOB_BACKUP_DIR, exist_ok=True)
        return jobs
    
    # List all files in the backup directory
    for filename in os.listdir(cst.JOB_BACKUP_DIR):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(cst.JOB_BACKUP_DIR, filename)
        
        try:
            with open(filepath, 'r') as f:
                job_dict = json.load(f)
                
            # Determine the job type and create the appropriate object
            if 'dataset_zip' in job_dict:
                job = DiffusionJob(**job_dict)
            elif 'dataset' in job_dict:
                job = TextJob(**job_dict)
            else:
                job = Job(**job_dict)
                
            jobs.append(job)
            logger.info(f"Loaded job backup from {filepath}")
        except Exception as e:
            logger.error(f"Error loading job backup from {filepath}: {str(e)}")
    
    return jobs

class TrainingWorker:
    def __init__(self, max_workers: int = 1):
        logger.info("=" * 80)
        logger.info("STARTING A TRAINING WORKER")
        logger.info("=" * 80)

        self.max_workers = max_workers
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}

        # track how many jobs are actively running
        self._running_count = 0
        self._run_lock = threading.Lock()

        # spin up N worker threads
        self.threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

        self.docker_client = docker.from_env()
        
        # Restore and re-run backup jobs
        self.restore_backup_jobs()
        
    def restore_backup_jobs(self):
        """
        Restore and re-run jobs from backup files.
        This is called during initialization to recover any jobs that failed
        during a previous run.
        """
        logger.info("Checking for backup jobs to restore...")
        backup_jobs = load_job_backups()
        
        if not backup_jobs:
            logger.info("No backup jobs found.")
            return
            
        logger.info(f"Found {len(backup_jobs)} backup jobs to restore.")
        
        # Re-enqueue each backup job
        for job in backup_jobs:
            logger.info(f"Restoring job {job.job_id} (model: {job.model})")
            # Clear any previous error message
            job.error_message = None
            # Enqueue the job with is_restored=True to avoid saving it again
            self.enqueue_job(job, is_restored=True)
            logger.info(f"Job {job.job_id} restored and queued for processing")

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break

            # increment running count
            with self._run_lock:
                self._running_count += 1
            job.status = JobStatus.RUNNING

            try:
                if isinstance(job, TextJob):
                    start_tuning_container(job)
                elif isinstance(job, DiffusionJob):
                    start_tuning_container_diffusion(job)
                job.status = JobStatus.COMPLETED
                # Delete the job backup when completed successfully
                delete_job_backup(job)
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                # Keep the backup file for failed jobs for debugging
            finally:
                # decrement running count and mark task done
                with self._run_lock:
                    self._running_count -= 1
                self.job_queue.task_done()

    def enqueue_job(self, job: Job, is_restored: bool = False):
        """
        Enqueue a job for processing.
        
        Args:
            job: The job to enqueue
            is_restored: Whether this job was restored from a backup
        """
        job.status = JobStatus.QUEUED
        
        # Save the job to disk before enqueueing it, unless it's a restored job
        if not is_restored:
            save_job_to_disk(job)
            
        self.job_queue.put(job)
        self.job_store[job.job_id] = job

    def active_job_count(self) -> int:
        """
        Returns the total in‐flight jobs: those queued plus those currently running.
        """
        with self._run_lock:
            running = self._running_count
        queued = self.job_queue.qsize()
        return running + queued

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))
        return job.status if job else JobStatus.NOT_FOUND

    def shutdown(self):
        # signal threads to exit
        for _ in self.threads:
            self.job_queue.put(None)
        for t in self.threads:
            t.join()
        self.docker_client.close()
