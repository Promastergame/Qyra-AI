"""
fix_data.py — Очистка данных от инструментов и переименование в Qyra.
"""

import json
import os
import sys

# Исправляем кодировку для Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "data/finetune/chat_merged.jsonl"
OUTPUT_PATH = "data/finetune/chat_clean.jsonl"

SYSTEM_QYRA = "You are Qyra, a helpful and friendly AI assistant."

def clean_conversation(messages):
    """Убираем tool_call и tool_result, оставляем только диалог."""
    cleaned = []
    
    # Добавляем системное сообщение с именем Qyra
    cleaned.append({
        "role": "system",
        "content": SYSTEM_QYRA
    })
    
    skip_until_assistant = False
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Пропускаем инструменты
        if role in ("tool_call", "tool_result"):
            skip_until_assistant = True
            continue
        
        # Пропускаем ассистента сразу после tool_result (он обычно про результат)
        if skip_until_assistant and role == "assistant":
            # Но если это не про инструмент — оставляем
            if "result" not in content.lower() and "calculate" not in content.lower():
                cleaned.append({"role": role, "content": content})
            skip_until_assistant = False
            continue
        
        # Оставляем system, user, assistant
        if role in ("system", "user", "assistant"):
            # Не дублируем system
            if role == "system" and len(cleaned) > 0:
                continue
            cleaned.append({"role": role, "content": content})
    
    # Убираем дубликаты подряд идущих сообщений одной роли
    deduped = []
    prev_role = None
    for msg in cleaned:
        if msg["role"] != prev_role:
            deduped.append(msg)
            prev_role = msg["role"]
    
    return deduped


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Файл не найден: {DATA_PATH}")
        return
    
    total = 0
    cleaned_count = 0
    
    with open(DATA_PATH, "r", encoding="utf-8") as f, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        
        for line in f:
            total += 1
            obj = json.loads(line.strip())
            messages = obj.get("messages", [])
            
            cleaned = clean_conversation(messages)
            
            # Оставляем только если есть user и assistant
            roles = [m["role"] for m in cleaned]
            if "user" in roles and "assistant" in roles:
                out.write(json.dumps({"messages": cleaned}, ensure_ascii=False) + "\n")
                cleaned_count += 1
    
    print(f"✅ Готово!")
    print(f"   Было: {total} диалогов")
    print(f"   Стало: {cleaned_count} диалогов (без инструментов)")
    print(f"   Файл: {OUTPUT_PATH}")
    print(f"\nТеперь обучай:")
    print(f"   python finetune.py --data {OUTPUT_PATH} --checkpoint checkpoints/best_pretrain.pt --epochs 3 --batch_size 8")


if __name__ == "__main__":
    main()
