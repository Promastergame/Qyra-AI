"""
check_tokenizer.py — Проверка токенизатора на твоих данных.
"""

import sentencepiece as spm
import json
import os

sp = spm.SentencePieceProcessor()
sp.Load('tokenizer/tok.model')

print(f"Tokenizer vocab size: {sp.GetPieceSize()}")
print(f"PAD={sp.pad_id()} UNK={sp.unk_id()} BOS={sp.bos_id()} EOS={sp.eos_id()}")

# Проверка special tokens
SPECIAL_TOKENS = ['<|system|>', '<|user|>', '<|assistant|>', '<|end|>', '<|tool_start|>', '<|tool_end|>', '<|tool_result|>']
print("\nSpecial tokens:")
for tok in SPECIAL_TOKENS:
    tid = sp.PieceToId(tok)
    status = "OK" if tid != sp.unk_id() else "NOT FOUND"
    print(f"  {tok} -> id {tid} [{status}]")

# Проверка на реальных данных
print("\n--- Testing on your data ---")
data_path = "data/finetune/chat_merged.jsonl"
if os.path.exists(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            for msg in obj.get('messages', []):
                text = msg.get('content', '')
                encoded = sp.Encode(text)
                decoded = sp.Decode(encoded)
                print(f"\nText: \"{text[:50]}...\"")
                print(f"Tokens: {encoded[:10]}... ({len(encoded)} total)")
                print(f"Round-trip OK: {text == decoded}")
else:
    print(f"File not found: {data_path}")

print("\n--- Tokenizer stats ---")
# Статистика по всему датасету
if os.path.exists(data_path):
    total_tokens = 0
    max_len = 0
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for msg in obj.get('messages', []):
                text = msg.get('content', '')
                ids = sp.Encode(text)
                total_tokens += len(ids)
                max_len = max(max_len, len(ids))
                count += 1
    
    print(f"Total messages: {count}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens/message: {total_tokens/count:.1f}")
    print(f"Max tokens in message: {max_len}")
    print(f"\nRecommended max_seq_len: {max_len + 20} (current: 128)")
