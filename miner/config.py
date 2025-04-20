import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from miner.logic.training_worker import TrainingWorker


load_dotenv()


T = TypeVar("T", bound=BaseModel)


@dataclass
class WorkerConfig:
    trainer: TrainingWorker


@lru_cache
def factory_worker_config() -> WorkerConfig:
    # Create a directory for job persistence
    persistence_dir = os.path.join(os.path.dirname(__file__), "..", "job_persistence")
    
    return WorkerConfig(
        trainer=TrainingWorker(max_workers=1, persistence_dir=persistence_dir),
    )
