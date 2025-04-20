import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fiber.logging_utils import get_logger
from fiber.miner.core import configuration

from miner.config import factory_worker_config
from miner.endpoints.tuning import factory_router as tuning_factory_router
from miner.logic.training_worker import load_job_backups


logger = get_logger(__name__)


def factory_app(debug: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = configuration.factory_config()
        metagraph = config.metagraph
        sync_thread = None
        if metagraph.substrate is not None:
            sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
            sync_thread.start()
        
        # Initialize the worker config to create the TrainingWorker
        worker_config = factory_worker_config()
        
        # Check for backup jobs at server startup
        logger.info("Checking for backup jobs at server startup...")
        backup_jobs = load_job_backups()
        
        if backup_jobs:
            logger.info(f"Found {len(backup_jobs)} backup jobs to restore.")
            # Process backup jobs
            for job in backup_jobs:
                logger.info(f"Restoring job {job.job_id} (model: {job.model})")
                # Clear any previous error message
                job.error_message = None
                # Enqueue the job with is_restored=True to avoid saving it again
                worker_config.trainer.enqueue_job(job, is_restored=True)
                logger.info(f"Job {job.job_id} restored and queued for processing")
        else:
            logger.info("No backup jobs found at server startup.")

        yield

        logger.info("Shutting down...")

        metagraph.shutdown()
        if metagraph.substrate is not None and sync_thread is not None:
            sync_thread.join()

    app = FastAPI(lifespan=lifespan, debug=debug)

    return app


logger = get_logger(__name__)

app = factory_app(debug=True)


tuning_router = tuning_factory_router()

app.include_router(tuning_router)

# if os.getenv("ENV", "prod").lower() == "dev":
#    configure_extra_logging_middleware(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7999)
