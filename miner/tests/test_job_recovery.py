import os
import shutil
import tempfile
import time
import unittest
from uuid import uuid4

from core.models.utility_models import TextJob, DiffusionJob, JobStatus, InstructDatasetType, FileFormat, ImageModelType
from miner.logic.job_persistence import JobPersistenceManager
from miner.logic.training_worker import TrainingWorker


class MockStartTuningContainer:
    """Mock function for start_tuning_container to simulate job processing."""
    def __init__(self, delay=0.1, should_fail=False):
        self.delay = delay
        self.should_fail = should_fail
        self.called_with = []
        
    def __call__(self, job):
        self.called_with.append(job)
        time.sleep(self.delay)  # Simulate processing time
        if self.should_fail:
            raise Exception("Simulated failure")


class TestJobRecovery(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_job_recovery_after_crash(self):
        """Test that jobs are recovered after a simulated crash."""
        # Create mock functions for start_tuning_container
        mock_start_tuning = MockStartTuningContainer(delay=0.1)
        mock_start_tuning_diffusion = MockStartTuningContainer(delay=0.1)
        
        # Create test jobs
        text_job_id = str(uuid4())
        text_job = TextJob(
            job_id=text_job_id,
            model="mistralai/Mistral-7B-v0.1",
            dataset="test-dataset.json",
            dataset_type=InstructDatasetType(),
            file_format=FileFormat.JSON
        )
        
        diffusion_job_id = str(uuid4())
        diffusion_job = DiffusionJob(
            job_id=diffusion_job_id,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            dataset_zip="test-dataset.zip",
            model_type=ImageModelType.SDXL
        )
        
        # Create a worker with persistence
        worker1 = TrainingWorker(max_workers=1, persistence_dir=self.test_dir)
        
        # Patch the start_tuning_container functions
        import miner.logic.training_worker
        original_start_tuning = miner.logic.training_worker.start_tuning_container
        original_start_tuning_diffusion = miner.logic.training_worker.start_tuning_container_diffusion
        miner.logic.training_worker.start_tuning_container = mock_start_tuning
        miner.logic.training_worker.start_tuning_container_diffusion = mock_start_tuning_diffusion
        
        try:
            # Enqueue the jobs
            worker1.enqueue_job(text_job)
            worker1.enqueue_job(diffusion_job)
            
            # Wait for the jobs to be persisted
            time.sleep(0.2)
            
            # Verify that the jobs are persisted
            persistence_manager = JobPersistenceManager(self.test_dir)
            self.assertIn(text_job_id, persistence_manager.list_jobs())
            self.assertIn(diffusion_job_id, persistence_manager.list_jobs())
            
            # Simulate a crash by shutting down the worker without processing the jobs
            worker1.shutdown()
            
            # Create a new worker with the same persistence directory
            worker2 = TrainingWorker(max_workers=1, persistence_dir=self.test_dir)
            
            # Wait for the jobs to be recovered and processed
            time.sleep(0.5)
            
            # Verify that the jobs were processed
            self.assertEqual(len(mock_start_tuning.called_with) + len(mock_start_tuning_diffusion.called_with), 2)
            
            # Shutdown the worker
            worker2.shutdown()
            
        finally:
            # Restore the original functions
            miner.logic.training_worker.start_tuning_container = original_start_tuning
            miner.logic.training_worker.start_tuning_container_diffusion = original_start_tuning_diffusion
            
    def test_running_job_recovery(self):
        """Test that running jobs are recovered after a simulated crash."""
        # Create a mock function that will "hang" to simulate a long-running job
        mock_start_tuning = MockStartTuningContainer(delay=10.0)  # Long delay
        
        # Create a test job
        job_id = str(uuid4())
        job = TextJob(
            job_id=job_id,
            model="mistralai/Mistral-7B-v0.1",
            dataset="test-dataset.json",
            dataset_type=InstructDatasetType(),
            file_format=FileFormat.JSON
        )
        
        # Create a worker with persistence
        worker1 = TrainingWorker(max_workers=1, persistence_dir=self.test_dir)
        
        # Patch the start_tuning_container function
        import miner.logic.training_worker
        original_start_tuning = miner.logic.training_worker.start_tuning_container
        miner.logic.training_worker.start_tuning_container = mock_start_tuning
        
        try:
            # Enqueue the job
            worker1.enqueue_job(job)
            
            # Wait for the job to start running
            time.sleep(0.2)
            
            # Verify that the job is persisted with RUNNING status
            persistence_manager = JobPersistenceManager(self.test_dir)
            persisted_job = persistence_manager.load_job(job_id)
            self.assertEqual(persisted_job.status, JobStatus.RUNNING)
            
            # Simulate a crash by shutting down the worker while the job is running
            worker1.shutdown()
            
            # Create a new worker with the same persistence directory
            worker2 = TrainingWorker(max_workers=1, persistence_dir=self.test_dir)
            
            # Wait for the job to be recovered and requeued
            time.sleep(0.2)
            
            # Verify that the job was requeued
            recovered_job = persistence_manager.load_job(job_id)
            self.assertEqual(recovered_job.status, JobStatus.QUEUED)
            
            # Shutdown the worker
            worker2.shutdown()
            
        finally:
            # Restore the original function
            miner.logic.training_worker.start_tuning_container = original_start_tuning


if __name__ == "__main__":
    unittest.main()