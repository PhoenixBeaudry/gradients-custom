# Job Persistence System

This directory contains persisted job data for the miner server. The job persistence system allows the server to recover from crashes and continue processing jobs that were queued or running when the crash occurred.

## Structure

- `jobs/`: Contains serialized job data in JSON format, with one file per job
- `queue.json`: Contains the current state of the job queue, including running and queued jobs

## How It Works

1. When a job is enqueued, it is serialized to JSON and stored in the `jobs/` directory
2. The queue state is also persisted to `queue.json`
3. When the job status changes (e.g., from QUEUED to RUNNING, or from RUNNING to COMPLETED), the job file is updated
4. On server startup, the system loads all persisted jobs and requeues those that were in QUEUED or RUNNING state
5. Jobs that were in RUNNING state when the server crashed are put back in the queue with the highest priority

## Recovery Process

When the server starts up, the `TrainingWorker` initializes a `JobPersistenceManager` which:

1. Loads all job files from the `jobs/` directory
2. Loads the queue order from the `queue.json` file
3. Reconstructs the job objects
4. Requeues jobs that were in QUEUED or RUNNING state
5. Prioritizes jobs that were in RUNNING state

## Job File Format

Each job file contains the serialized job data in JSON format:

```json
{
  "type": "TextJob",
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "model": "mistralai/Mistral-7B-v0.1",
  "status": "Running",
  "error_message": null,
  "expected_repo_name": "my-fine-tuned-model",
  "dataset": "my-dataset.json",
  "dataset_type": {
    "system_prompt": "You are a helpful assistant.",
    "system_format": "{system}",
    "field_system": "system",
    "field_instruction": "instruction",
    "field_input": "input",
    "field_output": "output",
    "format": "{system}\n\n{instruction}\n\n{input}",
    "no_input_format": "{system}\n\n{instruction}",
    "field": null
  },
  "file_format": "json"
}
```

## Queue File Format

The queue file contains an ordered list of job IDs:

```json
{
  "running": ["123e4567-e89b-12d3-a456-426614174000"],
  "queued": ["456e7890-e12b-34d5-a678-426614174001", "789e0123-e45b-67d8-a901-426614174002"]
}
```

## Maintenance

This directory is managed automatically by the server. Manual modifications to the files in this directory are not recommended as they may cause inconsistencies in the job queue.

If you need to clear the job queue, you can safely delete the contents of this directory when the server is not running.