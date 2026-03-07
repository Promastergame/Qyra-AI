"""
generate_tool_data.py — Generate synthetic training data for tool use.

Creates conversations teaching the model when and how to use tools:
  - Calculator (math expressions)
  - Python (code execution)
  - No-tool (direct answers — teaches when NOT to use tools)
  - Multi-step (sequential tool calls)

Usage:
    python generate_tool_data.py --output data/finetune/tool_chat.jsonl --count 3000
"""

import json
import random
import math
import os
import argparse
import subprocess
import sys
import tempfile
from typing import List, Dict


class CalculatorDataGenerator:
    """Generate calculator training examples."""

    ARITHMETIC_TEMPLATES = [
        "What is {a} + {b}?", "Calculate {a} - {b}.", "What is {a} times {b}?",
        "What is {a} * {b}?", "How much is {a} divided by {b}?", "What is {a} / {b}?",
        "Calculate {a} + {b} + {c}.", "What is {a} minus {b}?", "Multiply {a} by {b}.",
    ]

    FUNCTION_TEMPLATES = [
        ("What is the square root of {n}?", "sqrt({n})"),
        ("What is sin({deg}) degrees?", "sin({deg})"),
        ("What is log({n})?", "log({n})"),
        ("What is the factorial of {small_int}?", "factorial({small_int})"),
        ("What is {n_big} choose {k}?", "comb({n_big},{k})"),
    ]

    WORD_PROBLEMS = [
        {
            "question": "A rectangle has length {a} and width {b}. What is its area?",
            "expression": "{a} * {b}",
            "answer": "The area is {result} square units.",
        },
        {
            "question": "You travel {a} miles in {b} hours. What is your average speed?",
            "expression": "{a} / {b}",
            "answer": "Your average speed is {result} miles per hour.",
        },
        {
            "question": "A circle has radius {a}. What is its area?",
            "expression": "pi * {a}**2",
            "answer": "The area is approximately {result} square units.",
        },
    ]

    RESPONSE_BEFORE = ["Let me calculate that.", "I'll compute that.", "Sure, let me calculate.", ""]
    RESPONSE_AFTER = ["The result is {result}.", "The answer is {result}.", "That equals {result}."]

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or "You are a helpful assistant with access to a calculator tool."

    def generate(self, count: int) -> List[Dict]:
        examples = []
        for _ in range(count):
            r = random.random()
            if r < 0.4:
                ex = self._gen_arithmetic()
            elif r < 0.6:
                ex = self._gen_function()
            else:
                ex = self._gen_word_problem()
            if ex:
                examples.append(ex)
        return examples

    def _gen_arithmetic(self) -> Dict:
        a, b, c = random.randint(1, 999), random.randint(1, 999), random.randint(1, 99)
        template = random.choice(self.ARITHMETIC_TEMPLATES)
        question = template.format(a=a, b=b, c=c)

        if "+" in template and "-" not in template and "*" not in template and "/" not in template:
            expr = f"{a} + {b}"
        elif "-" in template or "minus" in template:
            expr = f"{a} - {b}"
        elif "*" in template or "times" in template or "Multiply" in template or "×" in template:
            expr = f"{a} * {b}"
        elif "/" in template or "divided" in template:
            expr = f"{a} / {b}"
        else:
            expr = f"{a} + {b}"

        return self._build_conversation(question, expr)

    def _gen_function(self) -> Dict:
        template_q, template_e = random.choice(self.FUNCTION_TEMPLATES)
        n = random.choice([4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 169, 196])
        deg = random.choice([0, 30, 45, 60, 90, 180, 270, 360])
        small_int = random.randint(1, 12)
        n_big = random.randint(5, 20)
        k = random.randint(1, min(5, n_big))

        question = template_q.format(n=n, deg=deg, small_int=small_int, n_big=n_big, k=k)
        expression = template_e.format(n=n, deg=deg, small_int=small_int, n_big=n_big, k=k)
        return self._build_conversation(question, expression)

    def _gen_word_problem(self) -> Dict:
        template = random.choice(self.WORD_PROBLEMS)
        a = random.randint(2, 100)
        b = random.randint(2, 100)

        question = template["question"].format(a=a, b=b)
        expression = template["expression"].format(a=a, b=b)

        safe_ns = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi, "e": math.e}
        try:
            result = eval(expression, {"__builtins__": {}}, safe_ns)
        except Exception:
            return self._gen_arithmetic()

        if isinstance(result, float):
            result_str = f"{result:.4f}".rstrip("0").rstrip(".") if abs(result) > 0.001 else f"{result:.6f}"
        else:
            result_str = str(result)

        before = random.choice(self.RESPONSE_BEFORE)
        after = template["answer"].format(result=result_str)
        return self._format_conversation(question, expression, result_str, before, after)

    def _build_conversation(self, question: str, expression: str) -> Dict:
        safe_ns = {
            "__builtins__": {}, "sqrt": math.sqrt, "sin": lambda x: math.sin(math.radians(x)),
            "cos": lambda x: math.cos(math.radians(x)), "factorial": math.factorial,
            "comb": math.comb, "log": math.log10, "pi": math.pi, "e": math.e,
        }
        try:
            result = eval(expression, {"__builtins__": {}}, safe_ns)
        except Exception:
            return None

        if isinstance(result, float):
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.6f}".rstrip("0").rstrip(".")
        else:
            result_str = str(result)

        before = random.choice(self.RESPONSE_BEFORE)
        after = random.choice(self.RESPONSE_AFTER).format(result=result_str)
        return self._format_conversation(question, expression, result_str, before, after)

    def _format_conversation(self, question, expression, result_str, before, after) -> Dict:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": question}]
        if before:
            messages.append({"role": "assistant", "content": before})
        messages.append({"role": "tool_call", "content": f'calculator\n{{"expression":"{expression}"}}'})
        messages.append({"role": "tool_result", "content": result_str})
        messages.append({"role": "assistant", "content": after})
        return {"messages": messages}


class PythonDataGenerator:
    """Generate Python code execution examples."""

    EXAMPLES = [
        {"q": "Write Python to compute the first {n} Fibonacci numbers.",
         "code": "def fib(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\nprint(fib({n}))",
         "params": {"n": lambda: random.randint(5, 15)}},
        {"q": "Write Python to check if {n} is prime.",
         "code": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\n\nprint(is_prime({n}))",
         "params": {"n": lambda: random.choice([7, 11, 13, 17, 23, 15, 21, 25, 49, 97])}},
        {"q": "Use Python to find the sum of numbers from 1 to {n}.",
         "code": "print(sum(range(1, {n} + 1)))",
         "params": {"n": lambda: random.randint(10, 500)}},
        {"q": "Write Python to reverse the string '{s}'.",
         "code": "s = '{s}'\nprint(s[::-1])",
         "params": {"s": lambda: random.choice(["hello", "python", "world", "transformer"])}},
    ]

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or "You are a helpful assistant with a Python tool."

    def generate(self, count: int) -> List[Dict]:
        examples = []
        for _ in range(count):
            template = random.choice(self.EXAMPLES)
            params = {k: v() for k, v in template["params"].items()}
            question = template["q"].format(**params)
            code = template["code"].format(**params)

            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.flush()
                result = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=5)
                output = result.stdout.strip()
                os.unlink(f.name)
                if not output or result.returncode != 0:
                    continue
            except Exception:
                continue

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": "I'll write and run that code for you."},
                {"role": "tool_call", "content": f'python\n{{"code":"{self._escape(code)}"}}'},
                {"role": "tool_result", "content": output},
                {"role": "assistant", "content": f"Here's the result:\n{output}"},
            ]
            examples.append({"messages": messages})
        return examples

    @staticmethod
    def _escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class NoToolDataGenerator:
    """Generate examples where the model should answer directly (NO tools)."""

    EXAMPLES = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare."),
        ("What color is the sky?", "The sky is typically blue during the day."),
        ("What is Python?", "Python is a high-level programming language."),
        ("Say hello.", "Hello! How can I help you today?"),
        ("What is a neural network?", "A neural network is a computing system inspired by biological neurons."),
        ("Thank you!", "You're welcome! Let me know if you need anything else."),
    ]

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or "You are a helpful assistant."

    def generate(self, count: int) -> List[Dict]:
        examples = []
        for _ in range(count):
            q, a = random.choice(self.EXAMPLES)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
            examples.append({"messages": messages})
        return examples


def generate_dataset(output_path: str, total_count: int = 3000, calc_ratio: float = 0.5,
                     python_ratio: float = 0.2, no_tool_ratio: float = 0.3):
    """Generate complete tool-use training dataset."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    calc_count = int(total_count * calc_ratio)
    python_count = int(total_count * python_ratio)
    no_tool_count = total_count - calc_count - python_count

    print(f"Generating {total_count} examples:")
    print(f"  Calculator: {calc_count}, Python: {python_count}, No-tool: {no_tool_count}")

    all_examples = []

    calc_gen = CalculatorDataGenerator()
    calc_examples = calc_gen.generate(calc_count)
    all_examples.extend(calc_examples)
    print(f"  OK Calculator: {len(calc_examples)}")

    py_gen = PythonDataGenerator()
    py_examples = py_gen.generate(python_count)
    all_examples.extend(py_examples)
    print(f"  OK Python: {len(py_examples)}")

    no_tool_gen = NoToolDataGenerator()
    no_tool_examples = no_tool_gen.generate(no_tool_count)
    all_examples.extend(no_tool_examples)
    print(f"  OK No-tool: {len(no_tool_examples)}")

    random.shuffle(all_examples)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nOK Dataset saved to {output_path} ({len(all_examples)} examples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tool-use training data")
    parser.add_argument("--output", type=str, default=os.path.join("data", "finetune", "tool_chat.jsonl"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    generate_dataset(args.output, args.count)
