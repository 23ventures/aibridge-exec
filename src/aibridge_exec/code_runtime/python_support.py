from __future__ import annotations

import ast
from pathlib import Path

from aibridge_exec.code_runtime.contracts import PolicyViolation


DENIED_IMPORTS = {
    "boto3",
    "ctypes",
    "fcntl",
    "ftplib",
    "glob",
    "http",
    "importlib",
    "multiprocessing",
    "pathlib2",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
}


def validate_python_code(
    code: str,
    *,
    allowed_imports: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate syntax and the local-process import denylist.

    Docker is the production security boundary. This validation is retained as
    defense-in-depth for the development-only local process backend.
    """

    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        raise PolicyViolation(
            f"Generated Python has invalid syntax: {exc}"
        ) from exc

    allowed = set(allowed_imports or ())
    import_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_roots.add(_validate_import(alias.name, allowed=allowed))
        elif isinstance(node, ast.ImportFrom):
            import_roots.add(
                _validate_import(node.module or "", allowed=allowed)
            )
    return tuple(sorted(root for root in import_roots if root))


def _validate_import(module_name: str, *, allowed: set[str]) -> str:
    # ``allowed`` remains part of the function contract because profiles define
    # the modules advertised to generated code. The hard security restriction
    # for the local backend is the denylist below, not a brittle allowlist.
    del allowed
    root = (module_name or "").split(".", 1)[0]
    if root in DENIED_IMPORTS:
        raise PolicyViolation(
            f"Generated Python import is not allowed: {root or module_name}"
        )
    return root


def write_runtime_helpers(root: Path) -> None:
    """Materialize the small, stable helper module available in sandboxes."""

    helper = r'''
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_ROOT = Path.cwd().resolve()


def _safe_path(path):
    resolved = Path(path).resolve()
    allowed_roots = ((_ROOT / "input").resolve(), (_ROOT / "output").resolve())
    if not any(resolved == base or base in resolved.parents for base in allowed_roots):
        raise ValueError(f"path is outside sandbox input/output: {path}")
    return resolved


def _require_command(name):
    command = shutil.which(name)
    if not command:
        raise RuntimeError(f"Required sandbox command is unavailable: {name}")
    return command


def pdf_to_text(path, *, layout=False, first_page=None, last_page=None, encoding="UTF-8"):
    """Extract PDF text using Poppler pdftotext and return it as a string."""
    pdf_path = _safe_path(path)
    command = [_require_command("pdftotext"), "-enc", encoding]
    if layout:
        command.append("-layout")
    if first_page is not None:
        command.extend(["-f", str(int(first_page))])
    if last_page is not None:
        command.extend(["-l", str(int(last_page))])
    command.extend([str(pdf_path), "-"])
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"pdftotext failed: {stderr[-1000:]}")
    return completed.stdout.decode("utf-8", errors="ignore")


def pdf_info(path):
    """Return basic PDF metadata using Poppler pdfinfo."""
    pdf_path = _safe_path(path)
    completed = subprocess.run(
        [_require_command("pdfinfo"), str(pdf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"pdfinfo failed: {stderr[-1000:]}")
    info = {}
    for line in completed.stdout.decode("utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def write_json(path, data):
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
'''
    (root / "sandbox_tools.py").write_text(
        helper.lstrip(),
        encoding="utf-8",
    )
