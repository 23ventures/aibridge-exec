from __future__ import annotations

import re

from aibridge_exec.code_runtime.models import ExecutionError


MODULE_PACKAGE_SUGGESTIONS = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
}


class ResultNormalizer:
    _MISSING_MODULE = re.compile(
        r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]"
    )
    _EXCEPTION = re.compile(
        r"(?m)^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+)$"
    )
    _OPTIONAL_DEPENDENCY = re.compile(
        r"ImportError:\s+Missing optional dependency ['\"]([^'\"]+)['\"]"
    )
    _BROKEN_IMPORT = re.compile(
        r"ImportError:\s+cannot import name ['\"][^'\"]+['\"] "
        r"from ['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]"
    )

    def normalize(
        self,
        *,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool = False,
        cancelled: bool = False,
        missing_outputs: tuple[str, ...] = (),
        policy_code: str | None = None,
        policy_message: str | None = None,
    ) -> ExecutionError | None:
        if cancelled:
            return ExecutionError(
                category="cancelled",
                type="ExecutionCancelled",
                message=policy_message or "Code execution was cancelled",
                retryable=True,
            )
        if timed_out:
            return ExecutionError(
                category="timeout",
                type="TimeoutExpired",
                message=policy_message or "Code execution exceeded its timeout",
                retryable=True,
            )
        if policy_code:
            category = (
                "resource_limit"
                if "quota" in policy_code or "limit" in policy_code
                else "permission_error"
            )
            return ExecutionError(
                category=category,
                type="PolicyViolation",
                message=policy_message or "Code execution policy rejected the request",
                retryable=False,
                details={"policy_code": policy_code},
            )
        if missing_outputs:
            return ExecutionError(
                category="expected_output_missing",
                type="ExpectedOutputMissing",
                message="Expected output was not generated: "
                + ", ".join(missing_outputs),
                retryable=True,
                details={"missing_outputs": list(missing_outputs)},
            )
        if exit_code in (0, None):
            return None

        combined = f"{stderr}\n{stdout}".strip()
        missing = self._MISSING_MODULE.search(combined)
        if missing:
            module = missing.group(1).split(".", 1)[0]
            return ExecutionError(
                category="missing_dependency",
                type="ModuleNotFoundError",
                module=module,
                suggested_package=MODULE_PACKAGE_SUGGESTIONS.get(module, module),
                message=f"No module named '{missing.group(1)}'",
                retryable=True,
            )
        optional = self._OPTIONAL_DEPENDENCY.search(combined)
        if optional:
            module = optional.group(1).split(".", 1)[0]
            return ExecutionError(
                category="missing_dependency",
                type="ImportError",
                module=module,
                suggested_package=MODULE_PACKAGE_SUGGESTIONS.get(module, module),
                message=f"Missing optional dependency '{optional.group(1)}'",
                retryable=True,
            )
        broken_import = self._BROKEN_IMPORT.search(combined)
        if broken_import:
            module = broken_import.group(1).split(".", 1)[0]
            return ExecutionError(
                category="dependency_error",
                type="ImportError",
                module=module,
                suggested_package=MODULE_PACKAGE_SUGGESTIONS.get(module, module),
                message=broken_import.group(0).removeprefix("ImportError: "),
                retryable=True,
            )
        exception = self._EXCEPTION.search(combined)
        exception_type = exception.group(1) if exception else None
        message = exception.group(2).strip() if exception else combined[-2000:]
        if "SyntaxError" in combined:
            category = "syntax_error"
            exception_type = "SyntaxError"
        elif exception_type in {"PermissionError"}:
            category = "permission_error"
        elif exception_type in {"MemoryError"} or "file too large" in combined.lower():
            category = "resource_limit"
        elif exit_code is not None and exit_code < 0:
            category = "process_terminated"
        else:
            category = "runtime_error"
        return ExecutionError(
            category=category,
            type=exception_type,
            message=message or f"Python exited with status {exit_code}",
            retryable=category in {
                "syntax_error",
                "missing_dependency",
                "runtime_error",
            },
        )
