"""
chat.py — Interactive CLI with tool support.

Usage:
  python chat.py --mode chat --checkpoint checkpoints/best_finetune.pt --tools
  python chat.py --mode chat --checkpoint checkpoints/best_finetune.pt --tools --auto-approve
"""

import os
import json
import argparse
from typing import Optional, List, Dict, Any, cast
import torch  # type: ignore
import sentencepiece as spm  # type: ignore

from config import TOKENIZER_PREFIX, CHECKPOINT_DIR, SPECIAL_TOKENS, TOOL_CONFIG  # type: ignore
from model import Qyra  # type: ignore
from tools.registry import ToolRegistry  # type: ignore
from tools.generation import generate_with_tools  # type: ignore


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mcfg = ckpt["model_config"]
    model = Qyra(mcfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, mcfg


def chat_mode(model: Qyra, sp: spm.SentencePieceProcessor, device: torch.device, mcfg: Any, args: argparse.Namespace):
    """Multi-turn chat with tool support."""

    # Setup tools
    tool_registry: Optional[ToolRegistry] = None
    use_tools: bool = getattr(args, "tools", True)
    if use_tools:
        auto_approve: bool = getattr(args, "auto_approve", False)
        tool_registry = ToolRegistry(require_confirmation=not auto_approve)
        print(f"  Tools enabled: {', '.join(tool_registry.list_tools())}")
        if auto_approve:
            print("  ⚠ Auto-approve: code will execute without confirmation")

    # Resolve special tokens
    role_ids: Dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
        tid = int(sp.PieceToId(tok))
        if tid != sp.unk_id():
            role_ids[tok] = tid

    end_val = role_ids.get("<|end|>", sp.eos_id())
    end_id = int(end_val) if end_val is not None else int(sp.eos_id())

    # Build system prompt
    base_system = str(args.system or "You are a helpful assistant.")
    if tool_registry is not None:
        tr = cast(ToolRegistry, tool_registry)
        system_prompt = base_system + "\n\n" + str(tr.get_descriptions())
    else:
        system_prompt = base_system

    # Encode system prompt
    sys_val = role_ids.get("<|system|>", sp.bos_id())
    sys_id_val = int(sys_val) if sys_val is not None else int(sp.bos_id())
    sys_ids: List[int] = (
        [sys_id_val]
        + [int(i) for i in sp.Encode(""+system_prompt)]
        + [end_id]
    )
    history_ids: List[int] = list(sys_ids)

    print("\n═══ Qyra Chat ═══")
    if tool_registry is not None:
        print("Tools: calculator, python, go")
    print("Commands: 'quit', 'reset', 'tools'")
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            history_ids = sys_ids.copy()
            print("[Conversation reset]\n")
            continue
        if user_input.lower() == "tools":
            if tool_registry is not None:
                tr2 = cast(ToolRegistry, tool_registry)
                print(tr2.get_descriptions())
            else:
                print("Tools are disabled.")
            print()
            continue

        # Build input sequence
        usr_val = role_ids.get("<|user|>", sp.bos_id())
        usr_id_val = int(usr_val) if usr_val is not None else int(sp.bos_id())
        user_ids: List[int] = (
            [usr_id_val]
            + [int(i) for i in sp.Encode(""+user_input)]
            + [end_id]
        )
        history_ids.extend(user_ids)
        asst_val = role_ids.get("<|assistant|>", sp.bos_id())
        asst_id_val = int(asst_val) if asst_val is not None else int(sp.bos_id())
        history_ids.append(asst_id_val)

        # Truncate if too long
        max_tokens_val = int(getattr(args, "max_tokens", 128))
        max_ctx = int(mcfg.max_seq_len) - max_tokens_val
        if len(history_ids) > max_ctx:
            # Keep system prompt, truncate middle
            keep_len = max_ctx - len(sys_ids)
            history_ids = list(sys_ids) + list(history_ids[-keep_len:])

        input_tensor = torch.tensor([history_ids], dtype=torch.long, device=device)

        # Generate response
        if tool_registry is not None:
            result = generate_with_tools(
                model=model,
                sp=sp,
                input_ids=input_tensor,
                tool_registry=tool_registry,
                device=device,
                max_new_tokens=max_tokens_val,
                max_tool_rounds=int(TOOL_CONFIG.get("max_tool_rounds", 5)),
                temperature=float(getattr(args, "temperature", 0.7)),
                top_k=int(getattr(args, "top_k", 50)),
                repetition_penalty=float(getattr(args, "rep_penalty", 1.1)),
                verbose=bool(getattr(args, "verbose", False)),
            )
            response_text = str(result.display_text)

            if bool(getattr(args, "verbose", False)) and result.tool_calls:
                for tc in result.tool_calls:
                    print(f"  🔧 {tc['tool']}({json.dumps(tc['args'])}) → {tc['result']}")

            history_ids = [int(i) for i in result.token_ids[0].tolist()]
        else:
            output = model.generate(
                input_tensor,
                max_new_tokens=max_tokens_val,
                temperature=float(getattr(args, "temperature", 0.7)),
                top_k=int(getattr(args, "top_k", 50)),
                repetition_penalty=float(getattr(args, "rep_penalty", 1.1)),
                eos_token_id=end_id,
            )
            generated = [int(i) for i in output[0, len(history_ids):].tolist()]  # type: ignore
            if generated and generated[-1] == end_id:
                generated = generated[:-1]
            response_text = str(sp.Decode(generated)).strip()
            history_ids = [int(i) for i in output[0].tolist()]

        print(f"Bot> {response_text}")
        print()


def complete_mode(model: Qyra, sp: spm.SentencePieceProcessor, device: torch.device, mcfg: Any, args: argparse.Namespace):
    """Free-form text completion (no tools)."""
    print("\n═══ Qyra Text Completion ═══")
    print("Type a prompt. 'quit' to exit.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        input_ids = [int(i) for i in sp.Encode(prompt)]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        max_tokens_val = int(getattr(args, "max_tokens", 128))
        output = model.generate(
            input_tensor,
            max_new_tokens=max_tokens_val,
            temperature=float(getattr(args, "temperature", 0.7)),
            top_k=int(getattr(args, "top_k", 50)),
            eos_token_id=int(sp.eos_id()),
        )

        full_text = str(sp.Decode(output[0].tolist()))
        print(f"\n{full_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Qyra Chat with Tools")
    parser.add_argument("--mode", choices=["complete", "chat"], default="chat")
    parser.add_argument("--checkpoint", type=str,
                        default=str(os.path.join(CHECKPOINT_DIR, "best_finetune.pt")))
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--rep_penalty", type=float, default=1.1)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Show tool details")

    parser.add_argument("--tools", action="store_true", default=True, help="Enable tools")
    parser.add_argument("--no-tools", dest="tools", action="store_false", help="Disable tools")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve code execution")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sp_model_path = str(TOKENIZER_PREFIX + ".model")
    if not os.path.isfile(sp_model_path):
        raise FileNotFoundError(f"Tokenizer not found at {sp_model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    # Verify tool tokens
    use_tools = bool(getattr(args, "tools", True))
    if use_tools:
        missing = []
        for tok in ["<|tool_start|>", "<|tool_end|>", "<|tool_result|>"]:
            if sp.PieceToId(tok) == sp.unk_id():
                missing.append(tok)
        if missing:
            print(f"⚠ Tool tokens missing: {missing}. Retrain tokenizer.")
            setattr(args, "tools", False)

    model, mcfg = load_model(str(args.checkpoint), device)
    print(f"Model: {model.count_parameters():,} params")

    if args.mode == "complete":
        complete_mode(model, sp, device, mcfg, args)
    else:
        chat_mode(model, sp, device, mcfg, args)


if __name__ == "__main__":
    main()
