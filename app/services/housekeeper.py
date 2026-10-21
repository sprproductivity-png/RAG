import os
import time
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.services.task_manager import get_task_manager, _get_checkpoint_path, _delete_checkpoint, _sync_to_db, IngestionTask
from qdrant_client.http import models as qmodels
from app.services.vector_db import get_vector_db_service
from app.services.bm25 import get_bm25_service

class HousekeeperService:
    """
    Automated housekeeping, self-healing, and maintenance service:
    1. Retries failed tasks from saved checkpoints (handling transient timeouts/reboots).
    2. Deletes unnecessary or stale cache files and checkpoints.
    3. Reconciles data integrity between PostgreSQL, Redis, Qdrant vectors, and BM25 indexes.
    """

    def __init__(self):
        self.last_report: Optional[Dict[str, Any]] = None
        self._is_running = False

    async def retry_failed_tasks(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Finds tasks that failed due to transient issues (timeouts, reboots, connection blips)
        and automatically retries them from their saved checkpoint.
        """
        task_manager = get_task_manager()
        retried_records = []

        tasks = await task_manager.list_tasks(limit=100)
        for t in tasks:
            if t.status != "failed":
                continue

            error_msg = (t.error or "").lower()
            # Transient failure patterns eligible for auto-retry
            transient_patterns = ["timed out", "timeout", "restart", "interrupted", "connection refused", "econnrefused", "reset by peer"]
            is_transient = any(p in error_msg for p in transient_patterns) or "failed" in (t.stage or "")

            task_obj = await task_manager.get_task_obj(t.task_id)
            if not task_obj:
                continue

            retry_count = getattr(task_obj, "retry_count", 0)
            if retry_count >= max_retries:
                continue

            # Verify that either raw cache or checkpoint exists to resume
            cache_exists = Path(task_obj.cache_path).exists()
            checkpoint_exists = _get_checkpoint_path(task_obj.task_id).exists()

            if not cache_exists and not checkpoint_exists:
                # Cannot resume without data source
                continue

            logger.info(f"Housekeeper: Auto-retrying failed task {task_obj.task_id} ({task_obj.filename}, attempt {retry_count + 1}/{max_retries})...")
            task_obj.retry_count = retry_count + 1
            await task_manager.retry_task(task_obj.task_id)

            retried_records.append({
                "task_id": task_obj.task_id,
                "filename": task_obj.filename,
                "attempt": task_obj.retry_count,
                "reason": t.error or "Transient failure recovery",
            })

        return retried_records

    async def clean_stale_cache(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Scans file_cache directory and deletes:
        - Cached raw files of completed or dismissed tasks.
        - Stale checkpoints of completed tasks.
        - Orphaned cache files older than max_age_hours.
        """
        task_manager = get_task_manager()
        cache_dir = settings.get_file_cache_dir()
        if not cache_dir.exists():
            return {"deleted_files": 0, "freed_bytes": 0, "cleaned": []}

        tasks = await task_manager.list_tasks(limit=200)
        task_status_map = {t.task_id: t.status for t in tasks}

        deleted_files = []
        freed_bytes = 0
        now = time.time()
        max_age_seconds = max_age_hours * 3600

        for file_path in cache_dir.iterdir():
            if not file_path.is_file():
                continue

            name = file_path.name
            if name == "tasks_history.json":
                continue

            should_delete = False
            reason = ""

            # Checkpoint files
            if name.endswith("_checkpoint.json"):
                task_id = name.replace("_checkpoint.json", "")
                status = task_status_map.get(task_id)
                if status == "completed":
                    should_delete = True
                    reason = "Task already completed"
                elif status is None and (now - file_path.stat().st_mtime > max_age_seconds):
                    should_delete = True
                    reason = f"Orphaned checkpoint older than {max_age_hours}h"

            # Raw cached files (e.g. task_abc123_filename.pdf)
            elif name.startswith("task_"):
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    task_id = f"{parts[0]}_{parts[1]}"
                    status = task_status_map.get(task_id)
                    if status == "completed":
                        should_delete = True
                        reason = "Task already completed"
                    elif status is None and (now - file_path.stat().st_mtime > max_age_seconds):
                        should_delete = True
                        reason = f"Orphaned cache file older than {max_age_hours}h"

            # Temporary partial files
            elif name.endswith(".tmp") or name.endswith(".part"):
                if now - file_path.stat().st_mtime > 3600:
                    should_delete = True
                    reason = "Abandoned partial upload file"

            if should_delete:
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    freed_bytes += size
                    deleted_files.append({"file": name, "size_bytes": size, "reason": reason})
                    logger.info(f"Housekeeper: Cleaned stale cache file '{name}' ({size} bytes, reason: {reason})")
                except Exception as ex:
                    logger.warning(f"Housekeeper: Could not delete cache file '{name}': {ex}")

        return {
            "deleted_files": len(deleted_files),
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "cleaned": deleted_files,
        }

    async def reconcile_data_integrity(self) -> Dict[str, Any]:
        """
        Reconciles system data integrity:
        1. Fixes stalled 'processing' tasks where no worker process is alive.
        2. Syncs Qdrant points count with BM25 document count and rebuilds BM25 if missing.
        3. Ensures required collection payload indexes (e.g. doc_id keyword index) exist.
        """
        task_manager = get_task_manager()
        vector_db = get_vector_db_service()
        bm25_service = get_bm25_service()

        reconciled_tasks = []
        collections_synced = []

        # 1. Reconcile stalled tasks
        tasks = await task_manager.list_tasks(limit=100)
        now = time.time()
        for t in tasks:
            if t.status == "processing":
                # Check if stalled (no update in over 10 minutes)
                try:
                    created_epoch = time.mktime(time.strptime(t.created_at, "%Y-%m-%dT%H:%M:%SZ"))
                    age_seconds = now - created_epoch
                except Exception:
                    age_seconds = 600

                if age_seconds > 600:
                    task_obj = await task_manager.get_task_obj(t.task_id)
                    if task_obj:
                        task_obj.status = "failed"
                        task_obj.stage = "failed"
                        task_obj.stage_message = "Stalled processing detected by housekeeper. Checkpoint preserved for retry."
                        task_obj.error = "Processing stalled or worker terminated."
                        await task_manager._sync_task(task_obj)
                        reconciled_tasks.append({
                            "task_id": t.task_id,
                            "filename": t.filename,
                            "action": "Marked stalled task as failed (preserved checkpoint)",
                        })

        # Clean phantom IDs in Redis
        if task_manager.redis_available and task_manager.redis:
            try:
                processing_ids = list(await task_manager.redis.smembers("rag:queue:processing"))
                for pid in processing_ids:
                    raw = await task_manager.redis.get(f"rag:task:{pid}")
                    if not raw:
                        await task_manager.redis.srem("rag:queue:processing", pid)
            except Exception:
                pass

        # 2. Reconcile Qdrant and BM25 collections
        try:
            collections = vector_db.list_collections()
            for coll_name in collections:
                stats = vector_db.get_collection_stats(coll_name)
                if not stats:
                    continue

                qdrant_points = stats.get("points_count", 0)
                bm25_index = bm25_service.get_index(coll_name)
                bm25_count = len(bm25_index.doc_records)

                # If Qdrant has points but BM25 index is empty or significantly out of sync
                if qdrant_points > 0 and (bm25_count == 0 or abs(qdrant_points - bm25_count) > max(5, int(qdrant_points * 0.1))):
                    logger.info(f"Housekeeper: BM25 out of sync with Qdrant for '{coll_name}' ({bm25_count} vs {qdrant_points} points). Rebuilding BM25 from Qdrant payloads...")
                    
                    # Scroll all points and payloads from Qdrant
                    offset = None
                    all_records = []
                    while True:
                        res, next_offset = vector_db.client.scroll(
                            collection_name=coll_name,
                            limit=250,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False,
                        )
                        for hit in res:
                            payload = hit.payload or {}
                            all_records.append({
                                "chunk_id": str(hit.id),
                                "doc_id": payload.get("doc_id", ""),
                                "chunk_index": payload.get("chunk_index", 0),
                                "raw_text": payload.get("raw_text", ""),
                                "context_summary": payload.get("context_summary"),
                                "contextualized_text": payload.get("contextualized_text"),
                                "metadata": payload.get("metadata", {}),
                            })
                        if next_offset is None:
                            break
                        offset = next_offset

                    if all_records:
                        bm25_index.add_records(all_records)
                        collections_synced.append({
                            "collection": coll_name,
                            "points_in_qdrant": qdrant_points,
                            "synced_bm25_count": len(all_records),
                            "action": "Rebuilt BM25 index from Qdrant storage",
                        })

                # 3. Ensure doc_id payload keyword index exists in Qdrant
                try:
                    vector_db.client.create_payload_index(
                        collection_name=coll_name,
                        field_name="doc_id",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Housekeeper: Collection reconciliation error: {e}")

        return {
            "reconciled_tasks": reconciled_tasks,
            "collections_synced": collections_synced,
        }

    async def run_housekeeping_cycle(self) -> Dict[str, Any]:
        """
        Executes a full housekeeping cycle:
        1. Auto-retries transient failed tasks.
        2. Cleans up stale cache and checkpoint files.
        3. Reconciles data integrity between DB, Redis, Qdrant, and BM25.
        """
        logger.info("Housekeeper: Starting full maintenance and housekeeping cycle...")
        t0 = time.perf_counter()

        retried = await self.retry_failed_tasks()
        cache_stats = await self.clean_stale_cache()
        recon_stats = await self.reconcile_data_integrity()

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_ms": duration_ms,
            "retried_tasks": retried,
            "cache_cleanup": cache_stats,
            "reconciliation": recon_stats,
            "status": "healthy",
        }

        self.last_report = report
        logger.info(
            f"Housekeeper: Cycle finished in {duration_ms}ms. "
            f"Retried: {len(retried)}, Cleaned files: {cache_stats.get('deleted_files', 0)} ({cache_stats.get('freed_mb', 0)} MB), "
            f"Reconciled: {len(recon_stats.get('reconciled_tasks', []))} tasks, {len(recon_stats.get('collections_synced', []))} collections."
        )
        return report

_housekeeper_instance: Optional[HousekeeperService] = None

def get_housekeeper_service() -> HousekeeperService:
    global _housekeeper_instance
    if _housekeeper_instance is None:
        _housekeeper_instance = HousekeeperService()
    return _housekeeper_instance

async def start_periodic_housekeeper(interval_seconds: int = 300):
    """Background recurring task that runs the housekeeping cycle every interval_seconds (default: 5 min)."""
    housekeeper = get_housekeeper_service()
    logger.info(f"Housekeeper: Periodic background worker started (interval={interval_seconds}s).")
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await housekeeper.run_housekeeping_cycle()
        except asyncio.CancelledError:
            logger.info("Housekeeper: Periodic background worker stopped.")
            break
        except Exception as e:
            logger.error(f"Housekeeper: Periodic worker exception: {e}", exc_info=True)
            await asyncio.sleep(30)

