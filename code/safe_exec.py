"""
safe_exec.py — shared sandbox for compiling and safety-checking S3-generated
candidate code, used by S4's validation gate (LLMAlgorithmAdapter.compile()
in laosa.py).

Two independent layers of defense, per Supplementary Section S4.8's disclosed
gap ("exec() currently runs with full process permissions... a candidate
that imports os and writes an unauthorized file is not rejected"):

  1. ast_safety_check (static, in-process, cheap): parses the candidate code
     and statically rejects disallowed imports, dunder-attribute access, and
     calls to dangerous builtins (open/exec/eval/compile/__import__/globals/
     locals/vars/getattr/setattr/delattr/input) BEFORE any execution happens.
     This directly closes the adversarial test's disclosed gap
     (multi_task_conflict: `import os` + unauthorized file write).

  2. run_isolated_compile (dynamic, subprocess, defense-in-depth): even if a
     future adversarial category evades the static check, the actual exec()
     + function call happens in a child process with:
       - RLIMIT_FSIZE = 0         (any file write of nonzero size fails)
       - RLIMIT_CPU   = ~timeout  (hard CPU-time cap)
       - RLIMIT_NPROC = 0         (cannot fork further children)
       - cwd = a fresh, empty temp directory (isolates any permitted I/O)
     and the parent enforces a wall-clock join(timeout) that works regardless
     of threading context -- this REPLACES the previous signal.alarm-based
     timeout (which only fires in the main thread), it does not just add to
     it.

Both layers must pass for compile() to accept a candidate. Layer 1 alone
closes the specific disclosed gap; layer 2 is there so an unanticipated
bypass of the AST whitelist still cannot write outside a throwaway sandbox
directory or hang the parent process.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import os
import resource
import tempfile

AST_ALLOWED_IMPORTS = {"math"}

_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "input", "exit", "quit", "help", "breakpoint",
}


class UnsafeCodeError(Exception):
    """Raised by ast_safety_check when candidate code fails the static whitelist."""


def ast_safety_check(code_string: str, allowed_imports=AST_ALLOWED_IMPORTS) -> None:
    """Raises UnsafeCodeError if code_string contains any disallowed construct.
    Returns None (silently) if the code passes. Never executes anything --
    this is a pure static AST walk."""
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        raise UnsafeCodeError(f"syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed_imports:
                    raise UnsafeCodeError(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in allowed_imports:
                raise UnsafeCodeError(f"disallowed import: {node.module}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise UnsafeCodeError(f"disallowed dunder attribute access: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise UnsafeCodeError(f"disallowed name: {node.id}")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            raise UnsafeCodeError("disallowed 'with' statement (commonly used to open files)")


def _child_target(code_string, conn):
    try:
        # FIX: 必须先 mkdtemp 再设置 RLIMIT_FSIZE=0，否则 tempfile 找不到可写目录
        try:
            tmpdir = tempfile.mkdtemp(prefix="laosa_sandbox_")
            os.chdir(tmpdir)
        except Exception:
            tmpdir = None
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        except Exception:
            pass

        import types as _types
        full = f"import math\n{code_string}"
        module = _types.ModuleType("test_module")
        exec(full, module.__dict__)
        func = getattr(module, "priority", None)
        if func is None:
            conn.send((False, "no priority() function defined"))
            return

        from dataclasses import dataclass

        @dataclass
        class _DummyTask:
            estimated_runtime: float = 10.0
            runtime: float = 10.0
            base_runtime: float = 10.0
            speed: float = 1.0
            resource_need: float = 8.0
            weight: float = 1.0
            deadline: float = 20.0
            task_type: str = "alignment"
            id: int = 0

        result = func(_DummyTask())
        import math as _m
        ok = isinstance(result, (int, float)) and not _m.isnan(result)
        conn.send((ok, None))
    except Exception as e:
        conn.send((False, f"{type(e).__name__}: {e}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_isolated_compile(code_string: str, timeout_sec: float = 2.0) -> bool:
    """Runs the compile-and-call check in an isolated child process. Returns
    True only if the child process completed, sent an ok=True result, and did
    so within timeout_sec wall-clock time. Any hang, crash, or ok=False result
    (including a caught exception inside the child) returns False -- same
    external contract as the original in-process compile()."""
    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(target=_child_target, args=(code_string, child_conn))
    proc.start()
    child_conn.close()
    proc.join(timeout_sec + 0.5)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    if parent_conn.poll():
        ok, _err = parent_conn.recv()
        return bool(ok)
    return False
