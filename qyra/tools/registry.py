"""
tools/registry.py — Central registry for all available tools.
"""

import json
from typing import Callable, Dict, List, Optional, Any, cast
from tools.calculator import SafeCalculator  # type: ignore
from tools.code_runner import PythonRunner, GoRunner  # type: ignore


TOOL_CONFIG = {
    "max_tool_rounds": 5,
    "calculator_timeout": 2.0,
    "python_timeout": 10.0,
    "go_timeout": 30.0,
    "max_output_chars": 2000,
    "require_confirmation": True,
}


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, require_confirmation: Optional[bool] = None):
        # Explicitly handle the default value from config
        config_val = TOOL_CONFIG.get("require_confirmation", True)
        # Ensure we have a boolean
        self.require_confirmation: bool = bool(require_confirmation if require_confirmation is not None else config_val)
        
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_defaults()

    def _register_defaults(self):
        calc_timeout = cast(float, TOOL_CONFIG.get("calculator_timeout", 2.0))
        calc = SafeCalculator(timeout=calc_timeout)
        py_runner = PythonRunner()
        go_runner = GoRunner()

        self.register(
            name="calculator",
            handler=self._handle_calculator(calc),
            description=(
                "Evaluates mathematical expressions. "
                "Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, factorial, etc. "
                'Example: {"expression": "sqrt(3**2 + 4**2)"}'
            ),
            confirm=False,
        )

        self.register(
            name="python",
            handler=self._handle_python(py_runner),
            description=(
                "Executes Python code and returns stdout. "
                'Example: {"code": "print(sum(range(100)))"}'
            ),
            confirm=True,
        )

        self.register(
            name="go",
            handler=self._handle_go(go_runner),
            description=(
                "Compiles and executes Go code. "
                'Example: {"code": "package main\\nimport \\\"fmt\\\"\\nfunc main() { fmt.Println(42) }"}'
            ),
            confirm=True,
        )

    def register(self, name: str, handler: Callable, description: str = "", confirm: bool = False):
        self._tools[name] = {
            "handler": handler,
            "description": description,
            "confirm": confirm,
        }

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_descriptions(self) -> str:
        lines = ["Available tools:"]
        for name, info in self._tools.items():
            lines.append(f"  - {name}: {info['description']}")
        return "\n".join(lines)

    def execute(self, tool_name: str, args_json: str) -> str:
        if tool_name not in self._tools:
            available = ", ".join(self._tools.keys())
            return f"Error: Unknown tool '{tool_name}'. Available: {available}"

        tool = self._tools[tool_name]

        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments: {e}"

        confirm_needed = cast(bool, tool["confirm"])
        if confirm_needed and self.require_confirmation:
            if not self._get_user_confirmation(tool_name, args):
                return "Error: Execution cancelled by user."

        try:
            handler = cast(Callable, tool["handler"])
            result = handler(args)
            return str(result)
        except Exception as e:
            return f"Error: Tool execution failed: {e}"

    def _get_user_confirmation(self, tool_name: str, args: Dict[str, Any]) -> bool:
        print(f"\n{'='*50}")
        print(f"⚠  Tool '{tool_name}' wants to execute:")
        print(f"{'─'*50}")

        if "code" in args:
            code_str = str(args["code"])
            lines = code_str.split("\n")
            for i, line in enumerate(lines, 1):
                print(f"  {i:3d} │ {line}")
        else:
            print(f"  Args: {json.dumps(args, indent=2)}")

        print(f"{'─'*50}")

        while True:
            try:
                response = input("  Execute? [y/n/always]: ").strip().lower()
                if response in ("y", "yes"):
                    return True
                elif response in ("n", "no"):
                    return False
                elif response == "always":
                    self.require_confirmation = False
                    print("  (Auto-approve enabled for this session)")
                    return True
                return False
            except (EOFError, KeyboardInterrupt):
                return False

    @staticmethod
    def _handle_calculator(calc: SafeCalculator):
        def handler(args: Dict[str, Any]) -> str:
            expression = args.get("expression", "")
            if not expression:
                return "Error: No expression provided."
            result = calc.evaluate(str(expression))
            if result["success"]:
                return str(result["formatted"])
            else:
                return f"Error: {result['error']}"
        return handler

    @staticmethod
    def _handle_python(runner: PythonRunner):
        def handler(args: Dict[str, Any]) -> str:
            code = args.get("code", "")
            if not code:
                return "Error: No code provided."
            result = runner.run(str(code))
            if result["success"]:
                return str(result["output"])
            elif result.get("timed_out"):
                return f"Error: Code timed out after {runner.timeout}s"
            else:
                return f"Error: {result['error']}"
        return handler

    @staticmethod
    def _handle_go(runner: GoRunner):
        def handler(args: Dict[str, Any]) -> str:
            code = args.get("code", "")
            if not code:
                return "Error: No code provided."
            result = runner.run(str(code))
            if result["success"]:
                return str(result["output"])
            elif result.get("timed_out"):
                return f"Error: Code timed out after {runner.timeout}s"
            else:
                return f"Error: {result['error']}"
        return handler


if __name__ == "__main__":
    registry = ToolRegistry(require_confirmation=False)
    print("Tool Registry Tests")
    print("=" * 50)
    test_res = registry.execute("calculator", '{"expression": "345*872"}')
    print(f"  calculator: 345*872 = {test_res}")
    test_res = registry.execute("python", '{"code": "print(sum(range(1, 101)))"}')
    print(f"  python: sum(1..100) = {test_res}")
