"""
merge_ru_datasets.py — Объединение нескольких JSONL файлов в один с расширенными параметрами.

Поддерживает:
  - Гибкое указание файлов для объединения
  - Фильтрацию по минимальной длине диалога
  - Перемешивание данных
  - Лимит на общее количество записей (например, 60 млн)
  - Валидацию JSON
  - Статистику по результату

Usage:
    python merge_ru_datasets.py --output data/finetune/chat_merged.jsonl \\
        --inputs chat.jsonl chat_converted.jsonl chat_extra.jsonl \\
        --min_turns 2 --max_samples 60000000 --shuffle --seed 42
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Optional, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple JSONL datasets with advanced options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/finetune/chat_merged.jsonl",
        help="Output JSONL file path",
    )
    
    # Input files
    parser.add_argument(
        "--inputs", "-i",
        type=str,
        nargs="+",
        default=None,
        help="Input JSONL files (relative to DATA_DIR or absolute paths)",
    )
    
    # Data directory for relative paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/finetune",
        help="Base directory for relative input paths",
    )
    
    # Filtering
    parser.add_argument(
        "--min_turns",
        type=int,
        default=1,
        help="Minimum number of turns (user+assistant pairs) per dialogue",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=0,
        help="Minimum total tokens in dialogue (approximate by whitespace split)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum total samples to keep (e.g., 60000000 for 60M)",
    )
    
    # Shuffling
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle all data before saving",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and sampling",
    )
    
    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate JSON structure (require 'messages' or 'conversation' key)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail on any invalid JSON line",
    )
    
    # Stats
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics after merging",
    )
    
    # Test
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests instead of merging",
    )
    
    return parser.parse_args()


def count_turns(record: Dict[str, Any]) -> int:
    """Count number of turns (messages) in a record."""
    messages = record.get("messages") or record.get("conversation") or record.get("dialog", [])
    return len(messages)


def count_tokens_approx(text: str) -> int:
    """Approximate token count by whitespace splitting."""
    return len(text.split())


def validate_record(record: Dict[str, Any]) -> bool:
    """Check if record has required structure."""
    has_messages = any(
        key in record for key in ["messages", "conversation", "dialog", "dialogue"]
    )
    return has_messages


def load_jsonl_file(
    filepath: str,
    validate: bool = False,
    strict: bool = False,
    min_turns: int = 1,
    min_tokens: int = 0,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Load records from a JSONL file with optional filtering.
    
    Returns:
        (records, valid_count, skipped_count)
    """
    records = []
    valid_count = 0
    skipped_count = 0
    
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return records, 0, 0
    
    print(f"  Loading: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    raise ValueError(f"Invalid JSON at {filepath}:{line_num}: {e}")
                skipped_count += 1
                continue
            
            # Validate structure
            if validate and not validate_record(record):
                skipped_count += 1
                continue
            
            # Filter by min_turns
            if min_turns > 1:
                turns = count_turns(record)
                if turns < min_turns:
                    skipped_count += 1
                    continue
            
            # Filter by min_tokens
            if min_tokens > 0:
                text = json.dumps(record, ensure_ascii=False)
                tokens = count_tokens_approx(text)
                if tokens < min_tokens:
                    skipped_count += 1
                    continue
            
            records.append(record)
            valid_count += 1
    
    return records, valid_count, skipped_count


def save_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    """Save records to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_stats(records: List[Dict[str, Any]], output_path: str) -> None:
    """Compute and print detailed statistics."""
    if not records:
        print("\n[STATS] No records to analyze")
        return
    
    turns_list = [count_turns(r) for r in records]
    texts = [json.dumps(r, ensure_ascii=False) for r in records]
    tokens_list = [count_tokens_approx(t) for t in texts]
    
    total_tokens = sum(tokens_list)
    avg_turns = sum(turns_list) / len(records)
    avg_tokens = total_tokens / len(records)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"  Total records:     {len(records):,}")
    print(f"  File size:         {file_size_mb:.2f} MB")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Avg turns/record:  {avg_turns:.2f}")
    print(f"  Avg tokens/record: {avg_tokens:.1f}")
    print(f"  Min turns:         {min(turns_list)}")
    print(f"  Max turns:         {max(turns_list)}")
    print(f"  Min tokens:        {min(tokens_list)}")
    print(f"  Max tokens:        {max(tokens_list)}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("MERGE RU DATASETS")
    print("=" * 60)
    print(f"Output:      {args.output}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Min turns:   {args.min_turns}")
    print(f"Min tokens:  {args.min_tokens}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'unlimited'}")
    print(f"Shuffle:     {args.shuffle}")
    print(f"Seed:        {args.seed}")
    print(f"Validate:    {args.validate}")
    print(f"Strict:      {args.strict}")
    print("=" * 60)
    
    # Determine input files
    if args.inputs:
        input_files = args.inputs
    else:
        # Default files to merge
        input_files = [
            "chat.jsonl",
            "chat_converted.jsonl",
            "chat_extra_converted.jsonl",
            "chat_v2_converted.jsonl",
            "chat_v3_converted.jsonl",
        ]
    
    # Resolve paths
    resolved_files = []
    for fname in input_files:
        if os.path.isabs(fname):
            resolved_files.append(fname)
        else:
            resolved_files.append(os.path.join(args.data_dir, fname))
    
    # Load all records
    all_records: List[Dict[str, Any]] = []
    total_valid = 0
    total_skipped = 0
    
    print("\nLoading files:")
    for fpath in resolved_files:
        records, valid, skipped = load_jsonl_file(
            fpath,
            validate=args.validate,
            strict=args.strict,
            min_turns=args.min_turns,
            min_tokens=args.min_tokens,
        )
        all_records.extend(records)
        total_valid += valid
        total_skipped += skipped
        if valid > 0:
            print(f"    [+] Added {valid:,} records")
    
    print(f"\nLoaded: {total_valid:,} valid, {total_skipped:,} skipped")
    
    # Shuffle if requested
    if args.shuffle:
        print(f"\nShuffling with seed {args.seed}...")
        random.shuffle(all_records)
    
    # Limit samples
    if args.max_samples and len(all_records) > args.max_samples:
        print(f"\nLimiting to {args.max_samples:,} samples...")
        all_records = all_records[:args.max_samples]
    
    # Save
    print(f"\nSaving to {args.output}...")
    save_jsonl(all_records, args.output)
    
    # Final stats
    saved_count = len(all_records)
    print(f"\nDone! Saved {saved_count:,} records to {args.output}")
    
    if args.stats:
        compute_stats(all_records, args.output)

    print("\n" + "=" * 60)


# ============================================================================
# TESTS
# ============================================================================

def _create_test_data(tmpdir: str) -> List[str]:
    """Create test JSONL files."""
    files = []
    
    # File 1: Valid dialogues
    f1 = os.path.join(tmpdir, "test1.jsonl")
    with open(f1, "w", encoding="utf-8") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "Good!"}]}, ensure_ascii=False) + "\n")
    files.append(f1)
    
    # File 2: More dialogues
    f2 = os.path.join(tmpdir, "test2.jsonl")
    with open(f2, "w", encoding="utf-8") as f:
        f.write(json.dumps({"conversation": [{"role": "user", "content": "Test"}, {"role": "assistant", "content": "Response"}]}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"invalid": "no messages key"}, ensure_ascii=False) + "\n")  # Invalid
        f.write(json.dumps({"messages": [{"role": "user", "content": "Long message with many tokens " * 10}, {"role": "assistant", "content": "Short"}]}, ensure_ascii=False) + "\n")
    files.append(f2)
    
    # File 3: Empty/edge cases
    f3 = os.path.join(tmpdir, "test3.jsonl")
    with open(f3, "w", encoding="utf-8") as f:
        f.write("\n")  # Empty line
        f.write(json.dumps({"messages": []}, ensure_ascii=False) + "\n")  # Empty messages
        f.write("not valid json\n")  # Invalid JSON
    files.append(f3)
    
    return files


def test_merge_basic(tmpdir: str = "test_tmp") -> bool:
    """Test basic merge functionality."""
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp(prefix="merge_test_")
    output_file = os.path.join(test_dir, "output.jsonl")
    
    try:
        files = _create_test_data(test_dir)
        
        # Test 1: Basic merge
        print("\n[Test 1] Basic merge...")
        import subprocess
        result = subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output_file],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Merge failed: {result.stderr}"
        
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 6, f"Expected 6 records, got {len(lines)}"
        print("  [PASS] Basic merge passed")
        
        # Test 2: Filter by min_turns
        print("\n[Test 2] Filter min_turns=2...")
        output2 = os.path.join(test_dir, "output2.jsonl")
        result = subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output2, "--min_turns", "2"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        with open(output2, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 4, f"Expected 4 records with min_turns=2, got {len(lines)}"
        print("  [PASS] min_turns filter passed")
        
        # Test 3: Validate structure
        print("\n[Test 3] Validate structure...")
        output3 = os.path.join(test_dir, "output3.jsonl")
        result = subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output3, "--validate"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        with open(output3, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 5, f"Expected 5 valid records, got {len(lines)}"
        print("  [PASS] Validation passed")
        
        # Test 4: Max samples limit
        print("\n[Test 4] Max samples limit...")
        output4 = os.path.join(test_dir, "output4.jsonl")
        result = subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output4, "--max_samples", "2"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        with open(output4, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2, f"Expected 2 records with max_samples=2, got {len(lines)}"
        print("  [PASS] Max samples limit passed")
        
        # Test 5: Shuffle with seed (reproducibility)
        print("\n[Test 5] Shuffle reproducibility...")
        output5a = os.path.join(test_dir, "output5a.jsonl")
        output5b = os.path.join(test_dir, "output5b.jsonl")
        subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output5a, "--shuffle", "--seed", "42"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output5b, "--shuffle", "--seed", "42"],
            capture_output=True, text=True,
        )
        with open(output5a, "r", encoding="utf-8") as f:
            content_a = f.read()
        with open(output5b, "r", encoding="utf-8") as f:
            content_b = f.read()
        assert content_a == content_b, "Shuffle with same seed should be reproducible"
        print("  [PASS] Shuffle reproducibility passed")
        
        # Test 6: Stats output
        print("\n[Test 6] Stats output...")
        result = subprocess.run(
            ["python", __file__, "-i"] + files + ["-o", output_file, "--stats"],
            capture_output=True,
            text=True,
        )
        assert "STATISTICS" in result.stdout, "Stats should be in output"
        assert "Total records:" in result.stdout
        print("  [PASS] Stats output passed")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


# Run tests with: python merge_ru_datasets.py --test

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test_merge_basic()
    else:
        main()
