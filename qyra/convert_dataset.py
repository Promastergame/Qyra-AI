"""
convert_dataset.py — Конвертация dataset.txt в JSONL формат для Qyra.

Usage:
    python convert_dataset.py --input dataset.txt --output data/finetune/chat.jsonl
"""

import json
import argparse
import os


def convert_to_jsonl(input_path, output_path):
    """Конвертировать <human>/<bot> формат в JSONL."""
    
    dialogues = []
    current_human = None
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if '<human>' in line and '<bot>' in line:
                parts = line.split('<bot>')
                human = parts[0].replace('<human>', '').strip()
                bot = parts[1].strip()
                
                if human and bot:
                    dialogue = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": human},
                            {"role": "assistant", "content": bot}
                        ]
                    }
                    dialogues.append(dialogue)
    
    print(f"Loaded {len(dialogues)} dialogues from {input_path}")
    
    # Сохраняем в JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for dlg in dialogues:
            f.write(json.dumps(dlg, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(dialogues)} dialogues to {output_path}")
    return len(dialogues)


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to JSONL")
    parser.add_argument("--input", type=str, default="dataset.txt",
                        help="Input file in <human>/<bot> format")
    parser.add_argument("--output", type=str, 
                        default="data/finetune/chat.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return
    
    convert_to_jsonl(args.input, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
