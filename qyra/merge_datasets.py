"""
merge_datasets.py — Объединение нескольких JSONL файлов в один.

Usage:
    python merge_datasets.py --output data/finetune/chat_merged.jsonl
"""

import os
import json

DATA_DIR = "data/finetune"

# Файлы для объединения
input_files = [
    "chat.jsonl",              # 3500 синтетических диалогов
    "chat_converted.jsonl",    # 145 из dataset.txt
    "chat_extra_converted.jsonl",  # 208 из dataset_extra.txt
    "chat_v2_converted.jsonl", # 341 из dataset_v2.txt
    "chat_v3_converted.jsonl", # 224 из dataset_v3.txt
]

output_file = os.path.join(DATA_DIR, "chat_merged.jsonl")

total = 0
with open(output_file, 'w', encoding='utf-8') as out:
    for fname in input_files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Skip (not found): {fname}")
            continue
        
        count = 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + '\n')
                    count += 1
        
        print(f"Added {count} dialogues from {fname}")
        total += count

print(f"\nTotal: {total} dialogues saved to {output_file}")
