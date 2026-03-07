"""
merge_all_data.py — Объединяет все датасеты для Qyra.
"""

import json
import os
import random
import sys

# Исправляем кодировку для Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data/finetune"
OUTPUT_PATH = "data/finetune/qyra_all.jsonl"

# Файлы для объединения (все JSONL кроме tool_chat)
FILES = [
    "chat_converted.jsonl",
    "chat_extra_converted.jsonl",
    "chat_merged.jsonl",
    "chat_v2_converted.jsonl",
    "chat_v3_converted.jsonl",
    "chat_v4_converted.jsonl",
    "emotions_support.jsonl",
    "fun_conversations.jsonl",
    "scp_chat_converted.jsonl",
    "waifu_chat.jsonl",
    # tool_chat.jsonl — НЕ берём, там инструменты
]

SYSTEM_QYRA = "You are Qyra, a helpful and friendly AI assistant."

def load_jsonl(filepath):
    """Загружает JSONL файл."""
    data = []
    if not os.path.exists(filepath):
        print(f"  ⚠ Не найден: {filepath}")
        return data
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Ошибка JSON в {filepath}: {e}")
    return data


def normalize_conversation(obj):
    """Приводим к единому формату."""
    messages = obj.get("messages", [])
    
    # Проверяем есть ли user и assistant
    roles = [m.get("role", "") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None
    
    # Добавляем system если нет
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_QYRA})
    
    # Убираем tool_call и tool_result
    cleaned = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("tool_call", "tool_result"):
            cleaned.append(msg)
    
    # Убираем дубликаты подряд идущих ролей
    deduped = []
    prev_role = None
    for msg in cleaned:
        if msg.get("role") != prev_role:
            deduped.append(msg)
            prev_role = msg.get("role")
    
    return {"messages": deduped}


def main():
    print("=== Поиск файлов...")
    
    all_data = []
    
    for filename in FILES:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"  Загружаю: {filename}")
        data = load_jsonl(filepath)
        all_data.extend(data)
    
    print(f"\n=== Всего загружено: {len(all_data)} диалогов")
    
    # Нормализуем
    print("=== Нормализуем...")
    normalized = []
    skipped = 0
    
    for obj in all_data:
        norm = normalize_conversation(obj)
        if norm:
            normalized.append(norm)
        else:
            skipped += 1
    
    print(f"   Успешно: {len(normalized)}")
    print(f"   Пропущено: {skipped}")
    
    # Перемешиваем
    random.shuffle(normalized)
    
    # Сохраняем
    print(f"\n=== Сохраняю в {OUTPUT_PATH}...")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for obj in normalized:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"\n=== ГОТОВО!")
    print(f"   Файл: {OUTPUT_PATH}")
    print(f"   Диалогов: {len(normalized):,}")
    print(f"\n=== Теперь обучай:")
    print(f"   python finetune.py --data {OUTPUT_PATH} --checkpoint checkpoints/best_pretrain.pt --epochs 5 --batch_size 8")


if __name__ == "__main__":
    main()
