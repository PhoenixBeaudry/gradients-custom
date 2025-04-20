import queue
import threading
from uuid import UUID

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob, TextJob, Job, JobStatus
from miner.logic.job_handler import (
    start_tuning_container,
    start_tuning_container_diffusion,
)

logger = get_logger(__name__)

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
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                # decrement running count and mark task done
                with self._run_lock:
                    self._running_count -= 1
                self.job_queue.task_done()

    def enqueue_job(self, job: Job):
        job.status = JobStatus.QUEUED
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
