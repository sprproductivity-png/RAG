import asyncio
import logging
from app.models.db import init_db
from app.services.task_manager import get_task_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Initializing Database...")
    init_db()
    
    logger.info("Starting Background Queue Worker...")
    tm = get_task_manager()
    # Explicitly start the worker loop
    tm.start_worker()
    
    logger.info("Worker is running and listening to Redis queue. To scale, use docker-compose up --scale worker=3")
    
    # Keep alive
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker shutting down.")
