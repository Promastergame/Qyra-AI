"""
tools/generation.py — Generation with automatic tool execution.
"""

import torch  # type: ignore
import sentencepiece as spm  # type: ignore
from typing import Optional, List
from dataclasses import dataclass

from tools.registry import ToolRegistry  # type: ignore
from tools.parser import ToolParser  # type: ignore


@dataclass
class GenerationResult:
    """Result of tool-augmented generation."""
    full_text: str
    display_text: str
    tool_calls: list
    token_ids: torch.Tensor


def generate_with_tools(
    model,
    sp: spm.SentencePieceProcessor,
    input_ids: torch.Tensor,
    tool_registry: ToolRegistry,
    device: torch.device,
    max_new_tokens: int = 128,
    max_tool_rounds: int = 5,
    temperature: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    verbose: bool = False,
) -> GenerationResult:
    """Generate text with automatic tool call detection and execution."""

    tool_start_id = sp.PieceToId("<|tool_start|>")
    tool_end_id = sp.PieceToId("<|tool_end|>")
    tool_result_id = sp.PieceToId("<|tool_result|>")
    end_id = sp.PieceToId("<|end|>")
    eos_id = sp.eos_id()

    unk = sp.unk_id()
    for name, tid in [("<|tool_start|>", tool_start_id),
                      ("<|tool_end|>", tool_end_id),
                      ("<|tool_result|>", tool_result_id)]:
        if tid == unk:
            raise ValueError(f"Token {name} not in vocabulary. Retrain tokenizer.")

    stop_tokens = {tool_end_id, end_id, eos_id}
    prompt_len = input_ids.shape[1]
    tool_calls_log = []
    current_ids = input_ids

    for round_num in range(max_tool_rounds):
        tokens_remaining = max_new_tokens - int(current_ids.shape[1] - prompt_len)
        if tokens_remaining <= 0:
            break

        output = model.generate(
            current_ids,
            max_new_tokens=min(tokens_remaining, max_new_tokens),
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_tokens,
        )

        new_tokens = output[0, prompt_len:].tolist()

        if tool_start_id in new_tokens:
            generated_text = sp.Decode(new_tokens)
            tool_call = ToolParser.parse(generated_text)

            if tool_call is not None:
                if verbose:
                    print(f"\n  🔧 Tool call: {tool_call.tool_name}")
                    print(f"     Args: {tool_call.args}")

                result_str = tool_registry.execute(
                    tool_call.tool_name,
                    tool_call.args_json,
                )

                if verbose:
                    print(f"     Result: {result_str[:100]}")

                tool_calls_log.append({
                    "tool": tool_call.tool_name,
                    "args": tool_call.args,
                    "result": result_str,
                    "round": round_num,
                })

                result_tokens = (
                    [tool_result_id]
                    + sp.Encode(result_str)
                    + [tool_end_id]
                )
                result_tensor = torch.tensor(
                    [result_tokens], dtype=torch.long, device=device
                )

                current_ids = torch.cat([output, result_tensor], dim=1)

                max_ctx = model.cfg.max_seq_len
                if current_ids.shape[1] > max_ctx:
                    current_ids = current_ids[:, -max_ctx:]
                    prompt_len = max(0, prompt_len - int(current_ids.shape[1] - max_ctx))

                continue

        current_ids = output
        break

    all_tokens = current_ids[0, prompt_len:].tolist()
    full_text = sp.Decode(all_tokens)

    display_text = full_text
    for marker in ["<|tool_start|>", "<|tool_end|>", "<|tool_result|>",
                    "<|end|>", "<|assistant|>"]:
        display_text = display_text.replace(marker, "")

    segments = ToolParser.extract_text_segments(full_text)
    if segments["tool_calls"]:
        parts = []
        if segments["before"]:
            parts.append(segments["before"])
        for tc in tool_calls_log:
            parts.append(f"[Used {tc['tool']}: {tc['result']}]")
        if segments["after"]:
            parts.append(segments["after"])
        display_text = " ".join(parts)

    display_text = display_text.strip()

    return GenerationResult(
        full_text=full_text,
        display_text=display_text,
        tool_calls=tool_calls_log,
        token_ids=current_ids,
    )
