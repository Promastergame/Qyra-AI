"""
train_tokenizer.py — Train a SentencePiece BPE tokeniser on your corpus.

Usage:
    python train_tokenizer.py [--vocab_size 8000] [--data_dir data/raw]

This reads all .md / .txt files, writes a temporary combined file,
then trains a SentencePiece model with custom special tokens.
"""

import os
import glob
import argparse
import tempfile
import sentencepiece as spm
from config import TOKENIZER_DIR, TOKENIZER_PREFIX, DATA_RAW_DIR, SPECIAL_TOKENS


def train_tokenizer(data_dir: str, vocab_size: int):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # Collect all text files
    patterns = ["*.md", "*.txt", "*.markdown"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No .md / .txt files found in {data_dir}. "
            "Add your training documents there first."
        )
    print(f"Found {len(files)} files for tokeniser training.")

    # Concatenate into a temp file (SentencePiece wants a single file)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    total_chars = 0
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        tmp.write(text + "\n")
        total_chars += len(text)
    tmp.close()
    print(f"Combined corpus: {total_chars:,} characters -> {tmp.name}")

    # Train SentencePiece
    # ── Key options ───────────────────────────────────────────────────
    #   model_type       : "bpe" (byte-pair encoding)
    #   vocab_size       : target vocabulary size
    #   user_defined_symbols : our chat special tokens
    #   pad_id / unk_id / bos_id / eos_id : reserved IDs
    #   character_coverage : 1.0 for English (lower for CJK)
    #   byte_fallback     : handle unknown bytes gracefully
    spm.SentencePieceTrainer.Train(
        input=tmp.name,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        user_defined_symbols=SPECIAL_TOKENS,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0,
        num_threads=os.cpu_count() or 1,
        byte_fallback=True,
        split_digits=True,           # treat each digit as a separate token
        max_sentence_length=16384,   # allow long documents
    )

    os.unlink(tmp.name)  # clean up temp file

    model_path = TOKENIZER_PREFIX + ".model"
    print(f"\nOK Tokeniser saved to {model_path}")

    # Quick sanity check
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    print(f"  Vocab size : {sp.GetPieceSize()}")
    print(f"  PAD={sp.pad_id()} UNK={sp.unk_id()} BOS={sp.bos_id()} EOS={sp.eos_id()}")
    for tok in SPECIAL_TOKENS:
        tid = sp.PieceToId(tok)
        print(f"  {tok} -> id {tid}")

    test = "Hello, world! This is a test."
    encoded = sp.Encode(test)
    decoded = sp.Decode(encoded)
    print(f'  Test encode: "{test}"')
    print(f"  Token IDs:   {encoded}")
    print(f'  Decoded:     "{decoded}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece tokeniser")
    parser.add_argument("--data_dir", type=str, default=DATA_RAW_DIR)
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    train_tokenizer(args.data_dir, args.vocab_size)
