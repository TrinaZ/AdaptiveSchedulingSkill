"""
govern_sandbox.py -- isolated execution of untrusted dispatching-rule code for the Govern skill
(assumption G1 of the paper). Linux / WSL only (uses fork + resource limits).

Two layers, both required:
  1. static_check(code): AST whitelist (safe_exec.ast_safety_check from the repo, plus a stricter
     node whitelist here: only `math` imports, no dunder attributes, no forbidden builtins, no
     with/try/global/nonlocal/class/lambda-default tricks, no attribute access on anything except
     task/state/math/local variables).
  2. isolated_costs(code, params, instances, objective_fn, simulate_fn, timeout):
     forks a child that
       - sets RLIMIT_FSIZE=0 (no file can grow), RLIMIT_NPROC=0 (no new processes),
         RLIMIT_CPU (CPU seconds), RLIMIT_AS (address space),
       - chdirs to an empty temp dir, closes stdin,
       - exec()s the code with a minimal builtins dict,
       - simulates every instance and returns per-instance costs through a pipe.
     The parent kills the child at the wall-clock timeout. Any exception, kill, or bad value
     -> the candidate is rejected (returns None plus a reason).
Because instances live in the parent's memory and are inherited by fork, the child can read them
but any modification stays in the child; the parent's evaluator and validation data are untouched.
"""
import ast, os, sys, math, resource, tempfile, multiprocessing as mp

ALLOWED_IMPORTS = {"math"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "exit", "quit", "help", "breakpoint", "memoryview",
                   "object", "type", "super", "classmethod", "staticmethod", "property", "dir", "id",
                   "print", "os", "sys", "subprocess", "socket", "shutil", "pathlib", "builtins"}
FORBIDDEN_NODES = (ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                   ast.Await, ast.Yield, ast.YieldFrom, ast.Try, ast.Raise, ast.Delete)
def _math_only_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name != "math":
        raise ImportError(f"import of {name} blocked")
    return math


SAFE_BUILTINS = {"__import__": _math_only_import, "abs": abs, "min": min, "max": max, "float": float, "int": int, "bool": bool, "round": round,
                 "len": len, "sum": sum, "range": range, "pow": pow, "True": True, "False": False, "None": None}


class Rejected(Exception):
    pass


def static_check(code):
    """Raise Rejected if code is not in the whitelist; return the parsed tree otherwise."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise Rejected(f"syntax: {e.msg}")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(f.name == "priority" for f in funcs):
        raise Rejected("no priority() defined")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise Rejected(f"forbidden construct: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise Rejected(f"import: {a.name}")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise Rejected(f"import: {node.module}")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            raise Rejected(f"attribute assignment: .{node.attr}")
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) and not isinstance(node.value, ast.Name):
            raise Rejected("subscript assignment on non-local object")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise Rejected(f"private/dunder attribute: {node.attr}")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise Rejected(f"forbidden name: {node.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("__"):
            raise Rejected("dunder string constant")
    # top level may only contain imports of math and function definitions
    for n in tree.body:
        if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Expr, ast.Assign)):
            raise Rejected(f"top-level statement not allowed: {type(n).__name__}")
    return tree


def _child(code, params, instances, objective_fn, simulate_fn, oname, cpu_sec, conn):
    try:
        try:
            d = tempfile.mkdtemp(prefix="govern_sandbox_"); os.chdir(d)
        except Exception:
            pass
        for lim, val in ((resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NPROC, 0),
                         (resource.RLIMIT_CPU, int(cpu_sec) + 1), (resource.RLIMIT_AS, 2 * 1024 ** 3)):
            try:
                resource.setrlimit(lim, (val, val))
            except Exception:
                pass
        try:
            sys.stdin.close()
        except Exception:
            pass
        ns = {"__builtins__": SAFE_BUILTINS, "math": math}
        exec(compile(code, "<candidate>", "exec"), ns)
        f = ns.get("priority")
        if f is None:
            conn.send(("err", "no priority()")); return
        rule = lambda t, s: _num(f(t, s, **params))
        costs = [objective_fn(oname, reqs, simulate_fn(reqs, R, rule)) for reqs, R in instances]
        conn.send(("ok", costs))
    except BaseException as e:   # includes SystemExit / KeyboardInterrupt raised by candidate code
        try:
            conn.send(("err", f"{type(e).__name__}: {str(e)[:150]}"))
        except Exception:
            pass
    finally:
        os._exit(0)


def _num(v):
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("non-finite priority")
    return v


def isolated_costs(code, params, instances, objective_fn, simulate_fn, oname, timeout=None):
    """Return (costs, None) or (None, reason). Never raises for candidate misbehaviour."""
    try:
        static_check(code)
    except Rejected as e:
        return None, f"static: {e}"
    timeout = timeout if timeout is not None else 5.0 + 0.05 * len(instances)
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(code, params, instances, objective_fn, simulate_fn, oname, timeout, child))
    p.start(); child.close()
    if not parent.poll(timeout):
        p.kill(); p.join()
        return None, f"timeout ({timeout:.0f}s)"
    try:
        status, val = parent.recv()
    except EOFError:
        p.join(); return None, f"child died (exit code {p.exitcode})"
    p.join()
    if status != "ok":
        return None, f"runtime: {val}"
    return val, None
