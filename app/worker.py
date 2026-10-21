import asyncio
import logging
from app.models.db import init_db
from app.services.task_manager import get_task_manager
from app.services.housekeeper import start_periodic_housekeeper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Initializing Database...")
    init_db()
    
    logger.info("Starting Background Ingestion Worker & Maintenance Housekeeper...")
    tm = get_task_manager()
    tm._is_running = True
    
    # Launch background periodic housekeeper loop (runs every 5 minutes)
    housekeeper_task = asyncio.create_task(start_periodic_housekeeper(interval_seconds=300))
    
    logger.info("Worker running and listening to Redis ingestion queue. (Scalable via docker-compose)")
    try:
        await tm._worker_loop()
    except asyncio.CancelledError:
        logger.info("Worker task cancelled.")
    finally:
        housekeeper_task.cancel()
        await tm.stop_worker()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker shutting down gracefully.")
