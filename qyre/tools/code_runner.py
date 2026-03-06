"""
tools/code_runner.py — Sandboxed execution of Python and Go code.

Security: Basic safety through timeouts and output limits.
For production, use OS-level isolation (Docker, seccomp, VMs).
"""

import os
import sys
import subprocess
import tempfile
import shutil
import platform
from typing import Optional, Dict, List, Any


class CodeRunner:
    """Base class for code execution tools."""

    def __init__(self, timeout: float = 10.0, max_output: int = 2000):
        self.timeout = timeout
        self.max_output = max_output

    def _run_subprocess(self, cmd: List[str], cwd: Optional[str] = None, stdin_data: Optional[str] = None) -> Dict[str, Any]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tempfile.gettempdir(),
            "LANG": "en_US.UTF-8",
        }

        if platform.system() == "Linux":
            prefix = "ulimit -v 262144 2>/dev/null; "
            shell_cmd = prefix + " ".join(cmd)
            cmd = ["bash", "-c", shell_cmd]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE if stdin_data else None,
                input=stdin_data,
            )

            stdout: str = result.stdout
            stderr: str = result.stderr

            if len(stdout) > self.max_output:
                stdout = str(stdout)[:self.max_output] + f"\n... (truncated)"  # type: ignore
            if len(stderr) > self.max_output:
                stderr = str(stderr)[:self.max_output] + f"\n... (truncated)"  # type: ignore

            return {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": result.returncode,
                "timed_out": False,
                "error": None if result.returncode == 0 else stderr.strip(),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "timed_out": True,
                "error": f"Execution timed out after {self.timeout}s",
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "timed_out": False,
                "error": f"Command not found: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "timed_out": False,
                "error": f"Execution error: {e}",
            }


class PythonRunner(CodeRunner):
    """Execute Python code in a subprocess."""

    def __init__(self, timeout: float = 10.0):
        super().__init__(timeout=timeout, max_output=2000)

    def run(self, code: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not code.strip():
            return {"success": False, "output": "", "error": "Empty code", "timed_out": False}

        tmpdir = tempfile.mkdtemp(prefix="qyra_py_")
        code_file = os.path.join(tmpdir, "script.py")

        try:
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)

            python_exe = sys.executable
            # Optionally update timeout if passed
            old_timeout = self.timeout
            if timeout is not None:
                self.timeout = timeout
            
            result = self._run_subprocess([python_exe, "-u", code_file], cwd=tmpdir)
            self.timeout = old_timeout

            output = result["stdout"].strip()
            if not output and result["stderr"] and not result["success"]:
                output = result["stderr"].strip()

            return {
                "success": result["success"],
                "output": output if output else "(no output)",
                "error": result["error"],
                "timed_out": result["timed_out"],
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "timed_out": False}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class GoRunner(CodeRunner):
    """Execute Go code by compiling and running it."""

    def __init__(self, timeout: float = 30.0):
        super().__init__(timeout=timeout, max_output=2000)
        self._go_available = self._check_go()

    def _check_go(self) -> bool:
        try:
            result = subprocess.run(["go", "version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(self, code: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self._go_available:
            return {"success": False, "output": "", "error": "Go compiler not found", "timed_out": False}

        if not code.strip():
            return {"success": False, "output": "", "error": "Empty code", "timed_out": False}

        tmpdir = tempfile.mkdtemp(prefix="qyra_go_")
        code_file = os.path.join(tmpdir, "main.go")

        try:
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)

            old_timeout = self.timeout
            if timeout is not None:
                self.timeout = timeout
                
            result = self._run_subprocess(["go", "run", code_file], cwd=tmpdir)
            self.timeout = old_timeout

            output = result["stdout"].strip()
            if not output and result["stderr"] and not result["success"]:
                output = result["stderr"].strip()

            return {
                "success": result["success"],
                "output": output if output else "(no output)",
                "error": result["error"],
                "timed_out": result["timed_out"],
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "timed_out": False}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("PythonRunner Test")
    print("=" * 40)
    py = PythonRunner(timeout=5)
    test_result = py.run("print(2 + 2)")
    print(f"  2+2 = {test_result['output']}")
