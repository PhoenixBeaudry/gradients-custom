import os
import queue
import threading
from typing import List, Optional
from uuid import UUID

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob, TextJob, Job, JobStatus
from miner.logic.job_handler import (
    start_tuning_container,
    start_tuning_container_diffusion,
)
from miner.logic.job_persistence import JobPersistenceManager

logger = get_logger(__name__)

class TrainingWorker:
    def __init__(self, max_workers: int = 1, persistence_dir: Optional[str] = None):
        logger.info("=" * 80)
        logger.info("STARTING A TRAINING WORKER")
        logger.info("=" * 80)

        self.max_workers = max_workers
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}

        # track how many jobs are actively running
        self._running_count = 0
        self._run_lock = threading.Lock()
        
        # Initialize persistence manager if persistence_dir is provided
        self.persistence_manager = None
        if persistence_dir:
            logger.info(f"Initializing job persistence with directory: {persistence_dir}")
            self.persistence_manager = JobPersistenceManager(persistence_dir)
            self._recover_jobs()

        # spin up N worker threads
        self.threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

        self.docker_client = docker.from_env()
        
    def _recover_jobs(self):
        """Recover jobs from persistence storage on startup."""
        logger.info("Recovering jobs from persistence storage...")
        
        try:
            # Load running and queued jobs from queue file
            running_job_ids, queued_job_ids = self.persistence_manager.load_queue()
            
            # Load all jobs from job files
            all_job_ids = self.persistence_manager.list_jobs()
            
            # First, requeue jobs that were running
            for job_id in running_job_ids:
                if job_id in all_job_ids:
                    job = self.persistence_manager.load_job(job_id)
                    if job:
                        logger.info(f"Recovering previously running job: {job_id}")
                        job.status = JobStatus.QUEUED  # Reset status to QUEUED
                        self.job_store[job_id] = job
                        self.job_queue.put(job)
            
            # Then, requeue jobs that were queued
            for job_id in queued_job_ids:
                if job_id in all_job_ids:
                    job = self.persistence_manager.load_job(job_id)
                    if job:
                        logger.info(f"Recovering queued job: {job_id}")
                        self.job_store[job_id] = job
                        self.job_queue.put(job)
                        
            logger.info(f"Recovered {len(running_job_ids) + len(queued_job_ids)} jobs")
            
        except Exception as e:
            logger.error(f"Error recovering jobs: {str(e)}")
            
    def _get_queue_state(self):
        """Get the current state of running and queued jobs."""
        with self._run_lock:
            # Get running job IDs (jobs being processed by worker threads)
            running_jobs = [job_id for job_id, job in self.job_store.items()
                           if job.status == JobStatus.RUNNING]
            
            # Get queued job IDs (jobs in the queue but not yet running)
            queued_jobs = [job_id for job_id, job in self.job_store.items()
                          if job.status == JobStatus.QUEUED]
            
            return running_jobs, queued_jobs

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break

            # increment running count and update job status
            with self._run_lock:
                self._running_count += 1
            job.status = JobStatus.RUNNING
            
            # Persist job status change if persistence is enabled
            if self.persistence_manager:
                try:
                    self.persistence_manager.persist_job(job)
                    running_jobs, queued_jobs = self._get_queue_state()
                    self.persistence_manager.persist_queue(running_jobs, queued_jobs)
                except Exception as e:
                    logger.error(f"Error persisting job status change for {job.job_id}: {str(e)}")

            try:
                if isinstance(job, TextJob):
                    start_tuning_container(job)
                elif isinstance(job, DiffusionJob):
                    start_tuning_container_diffusion(job)
                job.status = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                # Persist final job status if persistence is enabled
                if self.persistence_manager:
                    try:
                        self.persistence_manager.persist_job(job)
                        running_jobs, queued_jobs = self._get_queue_state()
                        self.persistence_manager.persist_queue(running_jobs, queued_jobs)
                    except Exception as e:
                        logger.error(f"Error persisting final job status for {job.job_id}: {str(e)}")
                
                # decrement running count and mark task done
                with self._run_lock:
                    self._running_count -= 1
                self.job_queue.task_done()

    def enqueue_job(self, job: Job):
        """Enqueue a job and persist it if persistence is enabled."""
        job.status = JobStatus.QUEUED
        self.job_queue.put(job)
        self.job_store[job.job_id] = job
        
        # Persist job and queue state if persistence is enabled
        if self.persistence_manager:
            try:
                self.persistence_manager.persist_job(job)
                running_jobs, queued_jobs = self._get_queue_state()
                queued_jobs.append(job.job_id)  # Add the new job to queued jobs
                self.persistence_manager.persist_queue(running_jobs, queued_jobs)
                logger.debug(f"Persisted job {job.job_id} and updated queue state")
            except Exception as e:
                logger.error(f"Error persisting job {job.job_id}: {str(e)}")

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
        """Shutdown the worker threads and clean up resources."""
        logger.info("Shutting down training worker...")
        
        # signal threads to exit
        for _ in self.threads:
            self.job_queue.put(None)
        for t in self.threads:
            t.join()
        self.docker_client.close()
        
        logger.info("Training worker shutdown complete")
