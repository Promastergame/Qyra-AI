"""
merge_ru_datasets.py — Объединяет русские датасеты для финетюна.
"""

import json
import glob
import os

# Путь к директории с данными
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "finetune")

# Ищем все русские датасеты
patterns = [
    "lmsys_clean_ru_queries_*.jsonl",
    "ru_chat_*.jsonl",
    "russian_*.jsonl",
]

output_lines = []
seen = set()

for pattern in patterns:
    for filepath in glob.glob(os.path.join(DATA_DIR, pattern)):
        print(f"Reading: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line not in seen:
                    output_lines.append(line)
                    seen.add(line)

# Также добавляем существующий chat.jsonl если есть
existing_chat = os.path.join(DATA_DIR, "chat.jsonl")
if os.path.exists(existing_chat):
    print(f"Reading: {existing_chat}")
    with open(existing_chat, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line not in seen:
                output_lines.append(line)
                seen.add(line)

# Сохраняем объединённый файл
output_path = os.path.join(DATA_DIR, "chat_merged.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(line + "\n" for line in output_lines)

print(f"\n✅ Total: {len(output_lines)} unique conversations")
print(f"Saved to: {output_path}")

# Копируем как основной chat.jsonl
import shutil
shutil.copy(output_path, os.path.join(DATA_DIR, "chat.jsonl"))
print("Copied to: data/finetune/chat.jsonl")
