import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import redis.asyncio as aioredis
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import ChunkDetail, TaskResponse, TaskStage, TaskStatus
from app.models.db import DBTask, SessionLocal
from app.services.bm25 import get_bm25_service
from app.services.chunking import chunk_document
from app.services.contextual import enrich_chunks_with_context
from app.services.embeddings import get_embedding_service
from app.services.file_parser import parse_uploaded_file
from app.services.vector_db import get_vector_db_service

def make_qdrant_uuid(doc_id: str, chunk_idx: int) -> str:
    """Generates deterministic UUID string for Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{chunk_idx}"))

def sanitize_filename(filename: str) -> str:
    """Removes special or dangerous path characters from filenames."""
    cleaned = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return cleaned[:100] or "unnamed_file"

@dataclass
class IngestionTask:
    task_id: str
    status: TaskStatus
    stage: TaskStage
    progress: int
    stage_message: str
    filename: str
    file_size: int
    file_type: str
    collection_name: str
    title: Optional[str]
    cache_path: str
    chunking_strategy: str = "semantic"
    use_contextual_retrieval: bool = True
    context_provider: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[float] = None
    chunks_created: Optional[int] = None
    error: Optional[str] = None
    sample_chunks: Optional[List[Dict[str, Any]]] = None

    def to_response(self) -> TaskResponse:
        sample_chunk_models = None
        if self.sample_chunks:
            sample_chunk_models = [ChunkDetail(**c) for c in self.sample_chunks]

        return TaskResponse(
            task_id=self.task_id,
            status=self.status,
            stage=self.stage,
            progress=self.progress,
            stage_message=self.stage_message,
            filename=self.filename,
            file_size=self.file_size,
            file_type=self.file_type,
            collection_name=self.collection_name,
            title=self.title,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            duration_ms=self.duration_ms,
            chunks_created=self.chunks_created,
            error=self.error,
            sample_chunks=sample_chunk_models,
        )

class TaskManager:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = asyncio.Lock()
        self.cache_dir = settings.get_file_cache_dir()
        self.redis: Optional[aioredis.Redis] = None
        self.redis_available: bool = False

    def _sync_task(self, task: IngestionTask):
        db = SessionLocal()
        try:
            db_task = db.query(DBTask).filter(DBTask.task_id == task.task_id).first()
            if not db_task:
                db_task = DBTask(task_id=task.task_id)
                db.add(db_task)
            
            db_task.status = task.status
            db_task.stage = task.stage
            db_task.progress = task.progress
            db_task.stage_message = task.stage_message
            db_task.filename = task.filename
            db_task.file_size = task.file_size
            db_task.file_type = task.file_type
            db_task.collection_name = task.collection_name
            db_task.title = task.title
            db_task.cache_path = task.cache_path
            db_task.chunking_strategy = task.chunking_strategy
            db_task.use_contextual_retrieval = task.use_contextual_retrieval
            db_task.context_provider = task.context_provider
            db_task.created_at = task.created_at
            db_task.started_at = task.started_at
            db_task.completed_at = task.completed_at
            db_task.duration_ms = task.duration_ms
            db_task.chunks_created = task.chunks_created
            db_task.error = task.error
            db_task.sample_chunks = task.sample_chunks
            
            db.commit()
        except Exception as e:
            logger.error(f"Error saving task to DB: {e}")
            db.rollback()
        finally:
            db.close()

        # Also sync to Redis if available
        if self.redis_available and self.redis:
            try:
                # Need event loop context for this inside a sync method, but wait!
                # _sync_task is async!
                pass # We handle Redis sync in the async version
            except Exception as e:
                pass

    async def _async_sync_task(self, task: IngestionTask):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_task, task)
        
        if self.redis_available and self.redis:
            try:
                data_str = json.dumps(asdict(task))
                await self.redis.set(f"rag:task:{task.task_id}", data_str)
                await self.redis.zadd("rag:tasks:index", {task.task_id: time.time()})
            except Exception as e:
                logger.warning(f"Failed to sync task {task.task_id} to Redis: {e}")

    async def get_task_obj(self, task_id: str) -> Optional[IngestionTask]:
        def fetch_db():
            db = SessionLocal()
            try:
                db_task = db.query(DBTask).filter(DBTask.task_id == task_id).first()
                if not db_task: return None
                return IngestionTask(
                    task_id=db_task.task_id, status=db_task.status, stage=db_task.stage,
                    progress=db_task.progress, stage_message=db_task.stage_message,
                    filename=db_task.filename, file_size=db_task.file_size,
                    file_type=db_task.file_type, collection_name=db_task.collection_name,
                    title=db_task.title, cache_path=db_task.cache_path,
                    chunking_strategy=db_task.chunking_strategy,
                    use_contextual_retrieval=db_task.use_contextual_retrieval,
                    context_provider=db_task.context_provider, created_at=db_task.created_at,
                    started_at=db_task.started_at, completed_at=db_task.completed_at,
                    duration_ms=db_task.duration_ms, chunks_created=db_task.chunks_created,
                    error=db_task.error, sample_chunks=db_task.sample_chunks
                )
            finally:
                db.close()
                
        loop = asyncio.get_running_loop()
        task = await loop.run_in_executor(None, fetch_db)
        if task: return task

        if self.redis_available and self.redis:
            try:
                raw = await self.redis.get(f"rag:task:{task_id}")
                if raw:
                    return IngestionTask(**json.loads(raw))
            except Exception as e:
                logger.debug(f"Redis get_task_obj error: {e}")
        return None

    async def get_task(self, task_id: str) -> Optional[TaskResponse]:
        task = await self.get_task_obj(task_id)
        if task:
            return task.to_response()
        return None

    async def list_tasks(self, limit: int = 50) -> List[TaskResponse]:
        def fetch_all():
            db = SessionLocal()
            try:
                tasks = db.query(DBTask).order_by(DBTask.created_at.desc()).limit(limit).all()
                return [IngestionTask(
                    task_id=db_task.task_id, status=db_task.status, stage=db_task.stage,
                    progress=db_task.progress, stage_message=db_task.stage_message,
                    filename=db_task.filename, file_size=db_task.file_size,
                    file_type=db_task.file_type, collection_name=db_task.collection_name,
                    title=db_task.title, cache_path=db_task.cache_path,
                    chunking_strategy=db_task.chunking_strategy,
                    use_contextual_retrieval=db_task.use_contextual_retrieval,
                    context_provider=db_task.context_provider, created_at=db_task.created_at,
                    started_at=db_task.started_at, completed_at=db_task.completed_at,
                    duration_ms=db_task.duration_ms, chunks_created=db_task.chunks_created,
                    error=db_task.error, sample_chunks=db_task.sample_chunks
                ).to_response() for db_task in tasks]
            finally:
                db.close()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fetch_all)

    async def delete_task(self, task_id: str) -> bool:
        def do_delete():
            db = SessionLocal()
            try:
                task = db.query(DBTask).filter(DBTask.task_id == task_id).first()
                if not task: return False
                try:
                    Path(task.cache_path).unlink(missing_ok=True)
                except: pass
                db.delete(task)
                db.commit()
                return True
            except:
                db.rollback()
                return False
            finally:
                db.close()
        loop = asyncio.get_running_loop()
        deleted = await loop.run_in_executor(None, do_delete)

        if self.redis_available and self.redis:
            try:
                await self.redis.delete(f"rag:task:{task_id}")
                await self.redis.zrem("rag:tasks:index", task_id)
            except Exception:
                pass
        return deleted

    async def ensure_redis(self) -> bool:
        """Attempts connection to Redis queue if configured. Returns True if connected."""
        if not settings.REDIS_URL or not settings.ENABLE_REDIS_QUEUE:
            self.redis_available = False
            return False

        if self.redis is not None and self.redis_available:
            try:
                await self.redis.ping()
                return True
            except Exception:
                self.redis_available = False
                self.redis = None

        try:
            client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await client.ping()
            self.redis = client
            self.redis_available = True
            logger.info(f"Connected to Redis ingestion queue at {settings.REDIS_URL}")
            await self._init_redis_queue_state()
            return True
        except Exception as e:
            logger.debug(f"Redis connection offline ({e}), falling back to in-memory queue.")
            self.redis = None
            self.redis_available = False
            return False

    async def _init_redis_queue_state(self):
        """Initializes and ensures queue inspection keys exist in Redis."""
        if not self.redis or not self.redis_available:
            return
        try:
            task_ids = await self.redis.zrevrange("rag:tasks:index", 0, 99)
            if task_ids:
                completed_in_redis = set(await self.redis.lrange("rag:queue:completed", 0, 99))
                for tid in reversed(task_ids):
                    if tid not in completed_in_redis:
                        raw = await self.redis.get(f"rag:task:{tid}")
                        if raw:
                            try:
                                d = json.loads(raw)
                                if d.get("status") == "completed":
                                    await self.redis.lpush("rag:queue:completed", tid)
                                elif d.get("status") == "failed":
                                    await self.redis.lpush("rag:queue:failed", tid)
                            except Exception:
                                pass
                await self.redis.ltrim("rag:queue:completed", 0, 99)
                await self.redis.ltrim("rag:queue:failed", 0, 99)
        except Exception as e:
            logger.debug(f"Redis queue state sync: {e}")

    async def _sync_task(self, task: IngestionTask):
        """Synchronizes task state to memory, disk history, and Redis (if connected)."""
        self.tasks[task.task_id] = task
        if self.redis_available and self.redis:
            try:
                data_str = json.dumps(asdict(task))
                await self.redis.set(f"rag:task:{task.task_id}", data_str)
                await self.redis.zadd("rag:tasks:index", {task.task_id: time.time()})
            except Exception as e:
                logger.warning(f"Failed to sync task {task.task_id} to Redis: {e}")
        self._save_history()

    def start_worker(self):
        if self.worker_task is None or self.worker_task.done():
            self._is_running = True
            self.worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Background Ingestion Task Worker started.")

    async def stop_worker(self):
        self._is_running = False
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        if self.redis is not None:
            try:
                await self.redis.aclose()
            except Exception:
                pass
            self.redis = None
            self.redis_available = False
        logger.info("Background Ingestion Task Worker stopped.")

    async def _worker_loop(self):
        await self.ensure_redis()
        logger.info(f"Ingestion Task Worker loop running (Redis active: {self.redis_available})")

        while self._is_running:
            try:
                task_id = None
                if self.redis_available and self.redis:
                    try:
                        # BLPOP with a 1 second timeout so worker can check self._is_running
                        pop_res = await self.redis.blpop(["rag:queue:ingestion"], timeout=1)
                        if pop_res:
                            _, task_id = pop_res
                            await self.redis.sadd("rag:queue:processing", task_id)
                    except Exception as e:
                        logger.warning(f"Redis queue pop error: {e}. Re-verifying Redis connection...")
                        await self.ensure_redis()
                        await asyncio.sleep(0.5)
                        continue
                else:
                    try:
                        task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if settings.REDIS_URL and settings.ENABLE_REDIS_QUEUE and not self.redis_available:
                            await self.ensure_redis()
                        continue

                if task_id:
                    task = await self.get_task_obj(task_id)
                    if task:
                        await self._process_task(task)
                    if not self.redis_available:
                        self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task worker loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _process_task(self, task: IngestionTask):
        t_start = time.perf_counter()
        task.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        task.status = "processing"

        try:
            # 1. Parsing Phase
            task.stage = "parsing"
            task.progress = 15
            task.stage_message = f"Reading cached file '{task.filename}' and extracting text..."
            await self._async_sync_task(task)

            cache_path = Path(task.cache_path)
            if not cache_path.exists():
                raise FileNotFoundError(f"Cached file '{task.cache_path}' not found on disk.")

            file_bytes = cache_path.read_bytes()
            extracted_text, inferred_title, metadata = parse_uploaded_file(file_bytes, task.filename)

            final_title = task.title.strip() if task.title and task.title.strip() else inferred_title
            task.title = final_title
            metadata["title"] = final_title
            metadata["task_id"] = task.task_id

            # 2. Chunking Phase
            task.stage = "chunking"
            task.progress = 35
            task.stage_message = f"Executing {task.chunking_strategy} chunking on extracted text ({len(extracted_text):,} chars)..."
            await self._async_sync_task(task)

            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            raw_chunks = chunk_document(
                text=extracted_text,
                strategy=task.chunking_strategy  # type: ignore
            )

            if not raw_chunks:
                raise ValueError("No valid chunks generated from document text.")

            # 3. Batch Processing Phase
            task.stage = "contextualizing"
            task.progress = 40
            task.stage_message = f"Dividing {len(raw_chunks)} chunks into batches for parallel processing..."
            await self._async_sync_task(task)

            embedding_service = get_embedding_service()
            vector_db_service = get_vector_db_service()
            bm25_service = get_bm25_service()

            vector_db_service.ensure_collection(
                collection_name=task.collection_name,
                vector_size=embedding_service.get_dimension()
            )

            batch_size = 50
            batches = [raw_chunks[i:i + batch_size] for i in range(0, len(raw_chunks), batch_size)]
            
            all_chunk_ids = []
            sample_chunks_preview = []
            
            semaphore = asyncio.Semaphore(4)
            completed_batches = 0
            
            async def process_batch(b_idx: int, b_chunks: list):
                nonlocal completed_batches
                async with semaphore:
                    def do_enrich():
                        return enrich_chunks_with_context(
                            document_text=extracted_text,
                            chunks=b_chunks,
                            document_title=final_title,
                            provider=task.context_provider,
                            enabled=task.use_contextual_retrieval
                        )
                    loop = asyncio.get_running_loop()
                    enriched = await loop.run_in_executor(None, do_enrich)
                    
                    b_ids = []
                    b_texts = []
                    b_payloads = []
                    b_samples = []
                    
                    for chunk in enriched:
                        chunk_idx = chunk["chunk_index"]
                        point_id = make_qdrant_uuid(doc_id, chunk_idx)
                        raw_text = chunk["raw_text"]
                        context_summary = chunk.get("context_summary")
                        contextualized_text = chunk.get("contextualized_text", raw_text)

                        payload = {
                            "doc_id": doc_id,
                            "chunk_id": point_id,
                            "chunk_index": chunk_idx,
                            "raw_text": raw_text,
                            "context_summary": context_summary,
                            "contextualized_text": contextualized_text,
                            "metadata": {**metadata, **(chunk.get("metadata") or {})},
                        }
                        b_ids.append(point_id)
                        b_texts.append(contextualized_text)
                        b_payloads.append(payload)
                        if len(b_samples) < 1:
                            b_samples.append({
                                "chunk_id": point_id,
                                "doc_id": doc_id,
                                "chunk_index": chunk_idx,
                                "raw_text": raw_text,
                                "context_summary": context_summary,
                                "contextualized_text": contextualized_text,
                                "metadata": payload["metadata"],
                            })
                            
                    b_embeddings = await loop.run_in_executor(
                        None, embedding_service.embed_texts, b_texts
                    )
                    b_vectors = [v.tolist() for v in b_embeddings]
                    
                    await loop.run_in_executor(
                        None,
                        vector_db_service.upsert_chunks,
                        task.collection_name,
                        b_ids,
                        b_vectors,
                        b_payloads
                    )
                    await loop.run_in_executor(
                        None,
                        bm25_service.add_chunks,
                        task.collection_name,
                        b_payloads
                    )
                    
                    completed_batches += 1
                    task.progress = int(40 + (50 * (completed_batches / len(batches))))
                    task.stage_message = f"Parallel processing batches... ({completed_batches}/{len(batches)} done)"
                    await self._async_sync_task(task)
                    
                    return b_ids, b_samples

            batch_tasks = [process_batch(i, b) for i, b in enumerate(batches)]
            results = await asyncio.gather(*batch_tasks)
            
            for b_ids, b_samples in results:
                all_chunk_ids.extend(b_ids)
                if len(sample_chunks_preview) < 3:
                    sample_chunks_preview.extend(b_samples)
            
            sample_chunks_preview = sample_chunks_preview[:3]

            # 6. Completed Phase
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            task.status = "completed"
            task.stage = "completed"
            task.progress = 100
            task.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            task.duration_ms = round(duration_ms, 2)
            task.chunks_created = len(all_chunk_ids)
            try:
                Path(task.cache_path).unlink(missing_ok=True)
            except Exception:
                pass
            task.sample_chunks = sample_chunks_preview
            task.stage_message = f"Successfully indexed {len(all_chunk_ids)} chunks into '{task.collection_name}' in {task.duration_ms}ms."

            logger.info(f"Task {task.task_id} completed successfully for '{task.filename}' ({task.chunks_created} chunks)")
            await self._async_sync_task(task)
            if self.redis_available and self.redis:
                try:
                    await self.redis.srem("rag:queue:processing", task.task_id)
                    await self.redis.lpush("rag:queue:completed", task.task_id)
                    await self.redis.ltrim("rag:queue:completed", 0, 99)
                except Exception as ex:
                    logger.warning(f"Failed to record completed task in Redis: {ex}")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            task.status = "failed"
            task.stage = "failed"
            task.progress = 100
            task.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            task.duration_ms = round(duration_ms, 2)
            task.error = str(e)
            task.stage_message = f"Failed during stage '{task.stage}': {str(e)}"
            try:
                Path(task.cache_path).unlink(missing_ok=True)
            except Exception:
                pass
            await self._async_sync_task(task)
            if self.redis_available and self.redis:
                try:
                    await self.redis.srem("rag:queue:processing", task.task_id)
                    await self.redis.lpush("rag:queue:failed", task.task_id)
                    await self.redis.ltrim("rag:queue:failed", 0, 99)
                except Exception as ex:
                    logger.warning(f"Failed to record failed task in Redis: {ex}")

    async def submit_file_task(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str,
        title: Optional[str] = None,
        chunking_strategy: str = "semantic",
        use_contextual_retrieval: bool = True,
        context_provider: Optional[str] = None,
    ) -> TaskResponse:
        """
        Step 1 of 2-step approach:
        Immediately persists raw file in local cache directory and queues background processing in Redis or in-memory queue.
        Returns TaskResponse instantly.
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        clean_name = sanitize_filename(filename)
        _, ext = os.path.splitext(clean_name)
        cached_file_path = self.cache_dir / f"{task_id}_{clean_name}"

        # Write raw file to disk cache
        cached_file_path.write_bytes(file_bytes)
        file_size = len(file_bytes)

        task = IngestionTask(
            task_id=task_id,
            status="queued",
            stage="queued",
            progress=5,
            stage_message=f"File cached ({file_size:,} bytes). Ingestion task queued for processing.",
            filename=filename,
            file_size=file_size,
            file_type=ext.lower() or "text",
            collection_name=collection_name or settings.DEFAULT_COLLECTION,
            title=title,
            cache_path=str(cached_file_path),
            chunking_strategy=chunking_strategy,
            use_contextual_retrieval=use_contextual_retrieval,
            context_provider=context_provider,
        )

        async with self._lock:
            await self._async_sync_task(task)

            await self.ensure_redis()

            if self.redis_available and self.redis:
                try:
                    await self.redis.rpush("rag:queue:ingestion", task_id)
                    logger.info(f"Queued file ingestion task {task_id} in Redis queue.")
                except Exception as e:
                    logger.warning(f"Failed to push task to Redis ({e}), falling back to in-memory queue.")
                    await self.queue.put(task_id)
            else:
                await self.queue.put(task_id)
                logger.info(f"Queued file ingestion task {task_id} in in-memory queue.")

        return task.to_response()

    async def get_task_obj(self, task_id: str) -> Optional[IngestionTask]:
        """Fetches IngestionTask object from memory or Redis."""
        task = self.tasks.get(task_id)
        if task:
            return task
        if self.redis_available and self.redis:
            try:
                raw = await self.redis.get(f"rag:task:{task_id}")
                if raw:
                    data = json.loads(raw)
                    task = IngestionTask(**data)
                    self.tasks[task_id] = task
                    return task
            except Exception as e:
                logger.debug(f"Redis get_task_obj error: {e}")
        return None

    async def get_task(self, task_id: str) -> Optional[TaskResponse]:
        task = await self.get_task_obj(task_id)
        if task:
            return task.to_response()
        return None

    async def list_tasks(self, limit: int = 50) -> List[TaskResponse]:
        if self.redis_available and self.redis:
            try:
                task_ids = await self.redis.zrevrange("rag:tasks:index", 0, limit - 1)
                if task_ids:
                    keys = [f"rag:task:{tid}" for tid in task_ids]
                    records = await self.redis.mget(keys)
                    for rec in records:
                        if rec:
                            try:
                                d = json.loads(rec)
                                t = IngestionTask(**d)
                                self.tasks[t.task_id] = t
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(f"Redis list_tasks error: {e}")

        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]
        return [t.to_response() for t in sorted_tasks]

    async def delete_task(self, task_id: str) -> bool:
        deleted = False
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            try:
                p = Path(task.cache_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            deleted = True

        if self.redis_available and self.redis:
            try:
                await self.redis.delete(f"rag:task:{task_id}")
                await self.redis.zrem("rag:tasks:index", task_id)
                await self.redis.srem("rag:queue:processing", task_id)
                await self.redis.lrem("rag:queue:completed", 0, task_id)
                await self.redis.lrem("rag:queue:failed", 0, task_id)
                await self.redis.lrem("rag:queue:ingestion", 0, task_id)
                deleted = True
            except Exception:
                pass

        self._save_history()
        return deleted

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Returns real-time queue health and breakdown across pending, processing, completed, and failed states."""
        await self.ensure_redis()
        if self.redis_available and self.redis:
            try:
                pending_count = await self.redis.llen("rag:queue:ingestion")
                processing_ids = list(await self.redis.smembers("rag:queue:processing"))
                completed_count = await self.redis.llen("rag:queue:completed")
                completed_ids = await self.redis.lrange("rag:queue:completed", 0, 9)
                failed_count = await self.redis.llen("rag:queue:failed")
                failed_ids = await self.redis.lrange("rag:queue:failed", 0, 9)

                return {
                    "queue_backend": "redis",
                    "redis_connected": True,
                    "pending_count": pending_count,
                    "processing_count": len(processing_ids),
                    "completed_count": completed_count,
                    "failed_count": failed_count,
                    "active_processing_task_ids": processing_ids,
                    "recent_completed_task_ids": completed_ids,
                    "recent_failed_task_ids": failed_ids,
                }
            except Exception as e:
                logger.warning(f"Redis get_queue_stats error: {e}")

        # In-memory fallback
        pending = self.queue.qsize()
        processing = []
        completed = []
        failed = []
        return {
            "queue_backend": "in_memory",
            "redis_connected": False,
            "pending_count": pending,
            "processing_count": len(processing),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "active_processing_task_ids": processing,
            "recent_completed_task_ids": completed[:10],
            "recent_failed_task_ids": failed[:10],
        }

_task_manager_instance: Optional[TaskManager] = None

def get_task_manager() -> TaskManager:
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager()
    return _task_manager_instance

