with open('app/services/task_manager.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)):
    if "logger.info(f\"Task {task.task_id} completed successfully" in lines[i]:
        lines[i] = "            " + lines[i].lstrip()
    elif "await self._sync_task(task)" in lines[i]:
        lines[i] = "            " + lines[i].lstrip()
    elif "# Clean up raw cached file" in lines[i]:
        lines[i] = "            " + lines[i].lstrip()
    elif "try:" in lines[i] and "Path(task.cache_path)" in ''.join(lines[i:i+3]):
        lines[i] = "            " + lines[i].lstrip()
    elif "p = Path(task.cache_path)" in lines[i]:
        lines[i] = "                " + lines[i].lstrip()
    elif "if p.exists():" in lines[i]:
        lines[i] = "                " + lines[i].lstrip()
    elif "p.unlink()" in lines[i]:
        lines[i] = "                    " + lines[i].lstrip()
    elif "logger.info(f\"Deleted cached file {task.cache_path}\")" in lines[i]:
        lines[i] = "                    " + lines[i].lstrip()
    elif "except Exception as ex:" in lines[i]:
        lines[i] = "            " + lines[i].lstrip()
    elif "logger.warning(f\"Could not delete cache file" in lines[i]:
        lines[i] = "                " + lines[i].lstrip()
    elif "task.stage_message = f\"Failed during stage" in lines[i]:
        lines[i] = "            " + lines[i].lstrip()
    elif "except Exception:" in lines[i] and "pass" in ''.join(lines[i:i+2]):
        lines[i] = "            " + lines[i].lstrip()
    elif "pass\n" == lines[i].lstrip() and "except Exception:" in lines[i-1]:
        lines[i] = "                " + lines[i].lstrip()

with open('app/services/task_manager.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
