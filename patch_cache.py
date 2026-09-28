import re

with open('app/services/task_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# For the success block
success_target = """
            logger.info(f"Task {task.task_id} completed successfully for '{task.filename}' ({task.chunks_created} chunks)")
            await self._sync_task(task)

            if self.redis_available and self.redis:
"""
success_replacement = """
            logger.info(f"Task {task.task_id} completed successfully for '{task.filename}' ({task.chunks_created} chunks)")
            await self._sync_task(task)

            try:
                p = Path(task.cache_path)
                if p.exists():
                    p.unlink()
                    logger.info(f"Deleted cached file {task.cache_path}")
            except Exception as ex:
                logger.warning(f"Could not delete cache file {task.cache_path}: {ex}")

            if self.redis_available and self.redis:
"""

# For the failure block
failure_target = """
            task.stage_message = f"Failed during stage '{task.stage}': {str(e)}"
            await self._sync_task(task)
            if self.redis_available and self.redis:
"""
failure_replacement = """
            task.stage_message = f"Failed during stage '{task.stage}': {str(e)}"
            await self._sync_task(task)

            try:
                p = Path(task.cache_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

            if self.redis_available and self.redis:
"""

content = content.replace(success_target.strip('\n'), success_replacement.strip('\n'))
content = content.replace(failure_target.strip('\n'), failure_replacement.strip('\n'))

with open('app/services/task_manager.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully")
