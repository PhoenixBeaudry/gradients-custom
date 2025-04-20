import os
import shutil
import tempfile
import unittest
from uuid import uuid4

from core.models.utility_models import TextJob, DiffusionJob, JobStatus, InstructDatasetType, FileFormat, ImageModelType
from miner.logic.job_persistence import JobPersistenceManager, job_to_dict, dict_to_job


class TestJobPersistence(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.persistence_manager = JobPersistenceManager(self.test_dir)
        
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_text_job_serialization(self):
        """Test serialization and deserialization of TextJob."""
        # Create a test TextJob
        dataset_type = InstructDatasetType(
            system_prompt="You are a helpful assistant.",
            field_instruction="instruction",
            field_output="output"
        )
        
        job = TextJob(
            job_id=str(uuid4()),
            model="mistralai/Mistral-7B-v0.1",
            dataset="test-dataset.json",
            dataset_type=dataset_type,
            file_format=FileFormat.JSON,
            status=JobStatus.QUEUED
        )
        
        # Serialize to dict
        job_dict = job_to_dict(job)
        
        # Deserialize back to job
        restored_job = dict_to_job(job_dict)
        
        # Check that the restored job matches the original
        self.assertEqual(job.job_id, restored_job.job_id)
        self.assertEqual(job.model, restored_job.model)
        self.assertEqual(job.dataset, restored_job.dataset)
        self.assertEqual(job.file_format, restored_job.file_format)
        self.assertEqual(job.status, restored_job.status)
        
        # Check dataset_type fields
        self.assertEqual(job.dataset_type.system_prompt, restored_job.dataset_type.system_prompt)
        self.assertEqual(job.dataset_type.field_instruction, restored_job.dataset_type.field_instruction)
        self.assertEqual(job.dataset_type.field_output, restored_job.dataset_type.field_output)
        
    def test_diffusion_job_serialization(self):
        """Test serialization and deserialization of DiffusionJob."""
        # Create a test DiffusionJob
        job = DiffusionJob(
            job_id=str(uuid4()),
            model="stabilityai/stable-diffusion-xl-base-1.0",
            dataset_zip="test-dataset.zip",
            model_type=ImageModelType.SDXL,
            status=JobStatus.QUEUED
        )
        
        # Serialize to dict
        job_dict = job_to_dict(job)
        
        # Deserialize back to job
        restored_job = dict_to_job(job_dict)
        
        # Check that the restored job matches the original
        self.assertEqual(job.job_id, restored_job.job_id)
        self.assertEqual(job.model, restored_job.model)
        self.assertEqual(job.dataset_zip, restored_job.dataset_zip)
        self.assertEqual(job.model_type, restored_job.model_type)
        self.assertEqual(job.status, restored_job.status)
        
    def test_persist_and_load_job(self):
        """Test persisting and loading a job."""
        # Create a test job
        job_id = str(uuid4())
        job = TextJob(
            job_id=job_id,
            model="mistralai/Mistral-7B-v0.1",
            dataset="test-dataset.json",
            dataset_type=InstructDatasetType(),
            file_format=FileFormat.JSON,
            status=JobStatus.QUEUED
        )
        
        # Persist the job
        self.persistence_manager.persist_job(job)
        
        # Load the job
        loaded_job = self.persistence_manager.load_job(job_id)
        
        # Check that the loaded job matches the original
        self.assertEqual(job.job_id, loaded_job.job_id)
        self.assertEqual(job.model, loaded_job.model)
        self.assertEqual(job.dataset, loaded_job.dataset)
        self.assertEqual(job.file_format, loaded_job.file_format)
        self.assertEqual(job.status, loaded_job.status)
        
    def test_persist_and_load_queue(self):
        """Test persisting and loading the queue state."""
        # Create test job IDs
        running_jobs = [str(uuid4()), str(uuid4())]
        queued_jobs = [str(uuid4()), str(uuid4()), str(uuid4())]
        
        # Persist the queue state
        self.persistence_manager.persist_queue(running_jobs, queued_jobs)
        
        # Load the queue state
        loaded_running, loaded_queued = self.persistence_manager.load_queue()
        
        # Check that the loaded queue state matches the original
        self.assertEqual(running_jobs, loaded_running)
        self.assertEqual(queued_jobs, loaded_queued)
        
    def test_list_jobs(self):
        """Test listing all jobs."""
        # Create and persist test jobs
        job_ids = [str(uuid4()) for _ in range(5)]
        for job_id in job_ids:
            job = TextJob(
                job_id=job_id,
                model="mistralai/Mistral-7B-v0.1",
                dataset="test-dataset.json",
                dataset_type=InstructDatasetType(),
                file_format=FileFormat.JSON
            )
            self.persistence_manager.persist_job(job)
            
        # List all jobs
        listed_jobs = self.persistence_manager.list_jobs()
        
        # Check that all job IDs are in the list
        for job_id in job_ids:
            self.assertIn(job_id, listed_jobs)
            
        # Check that the count matches
        self.assertEqual(len(job_ids), len(listed_jobs))
        
    def test_delete_job(self):
        """Test deleting a job."""
        # Create and persist a test job
        job_id = str(uuid4())
        job = TextJob(
            job_id=job_id,
            model="mistralai/Mistral-7B-v0.1",
            dataset="test-dataset.json",
            dataset_type=InstructDatasetType(),
            file_format=FileFormat.JSON
        )
        self.persistence_manager.persist_job(job)
        
        # Verify the job exists
        self.assertIn(job_id, self.persistence_manager.list_jobs())
        
        # Delete the job
        self.persistence_manager.delete_job(job_id)
        
        # Verify the job no longer exists
        self.assertNotIn(job_id, self.persistence_manager.list_jobs())


if __name__ == "__main__":
    unittest.main()