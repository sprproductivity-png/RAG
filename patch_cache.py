import re

with open('app/services/task_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(
    r'(task\.error = str\(e\)\n\s*task\.stage_message = f"Failed during stage \'{task\.stage}\': \{str\(e\)\}")',
    re.DOTALL
)

replacement = r'\1\n            try:\n                Path(task.cache_path).unlink(missing_ok=True)\n            except Exception:\n                pass'

new_content, count = pattern.subn(replacement, content)
if count > 0:
    with open('app/services/task_manager.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'Replaced cache deletion logic on failure ({count} occurrences).')
else:
    print('Failed to patch.')
