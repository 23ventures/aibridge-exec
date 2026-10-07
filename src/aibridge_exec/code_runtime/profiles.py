from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeEnvironmentProfile:
    name: str
    allowed_imports: tuple[str, ...]
    package_allowlist: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def available_imports(self, *, package_install_enabled: bool) -> tuple[str, ...]:
        available = []
        for import_name in self.allowed_imports:
            if _import_available(import_name):
                available.append(import_name)
            elif package_install_enabled and import_name in self.package_allowlist:
                available.append(import_name)
        return tuple(available)


PYTHON_BASIC_IMPORTS = (
    "collections",
    "csv",
    "datetime",
    "io",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "re",
    "statistics",
    "typing",
)

PYTHON_DATA_IMPORTS = (
    *PYTHON_BASIC_IMPORTS,
    "PIL",
    "bs4",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "pydantic",
    "scipy",
    "sklearn",
)

PYTHON_DOCUMENT_IMPORTS = (
    *PYTHON_DATA_IMPORTS,
    "sandbox_tools",
    "camelot",
    "fitz",
    "pdf2image",
    "pdfplumber",
    "pdfminer",
    "PIL",
    "pypdf",
    "pytesseract",
    "tabula",
)

DEFAULT_ENVIRONMENT_PROFILES = {
    "python-basic": CodeEnvironmentProfile(
        name="python-basic",
        allowed_imports=PYTHON_BASIC_IMPORTS,
        description="Small Python runtime for text, JSON, CSV, and simple transforms.",
    ),
    "python-data": CodeEnvironmentProfile(
        name="python-data",
        allowed_imports=PYTHON_DATA_IMPORTS,
        package_allowlist={
            "PIL": "pillow",
            "bs4": "beautifulsoup4",
            "matplotlib": "matplotlib",
            "numpy": "numpy",
            "openpyxl": "openpyxl",
            "pandas": "pandas",
            "pydantic": "pydantic",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
        },
        description="Python runtime for tabular data and spreadsheet transforms.",
    ),
    "python-documents": CodeEnvironmentProfile(
        name="python-documents",
        allowed_imports=PYTHON_DOCUMENT_IMPORTS,
        package_allowlist={
            "bs4": "beautifulsoup4",
            "camelot": "camelot-py",
            "fitz": "pymupdf",
            "matplotlib": "matplotlib",
            "numpy": "numpy",
            "openpyxl": "openpyxl",
            "pandas": "pandas",
            "pdf2image": "pdf2image",
            "pdfminer": "pdfminer.six",
            "pdfplumber": "pdfplumber",
            "PIL": "pillow",
            "pypdf": "pypdf",
            "pydantic": "pydantic",
            "pytesseract": "pytesseract",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
            "tabula": "tabula-py",
        },
        description="Python runtime for document parsing plus tabular outputs.",
    ),
}


def resolve_environment_profile(
    name: str | None,
    *,
    overrides: dict[str, Any] | None = None,
) -> CodeEnvironmentProfile:
    profiles = dict(DEFAULT_ENVIRONMENT_PROFILES)
    profiles.update(_parse_profile_overrides(overrides or {}))

    profile_name = str(name or "python-basic").strip() or "python-basic"
    profile = profiles.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown code runtime environment profile: {profile_name}")
    return profile


def parse_environment_profile_overrides(raw: str | dict | None) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid code runtime environment profile JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Code runtime environment profile config must be a JSON object")
    return parsed


def _parse_profile_overrides(raw_profiles: dict[str, Any]) -> dict[str, CodeEnvironmentProfile]:
    profiles: dict[str, CodeEnvironmentProfile] = {}
    for name, raw in raw_profiles.items():
        profile_name = str(name or "").strip()
        if not profile_name or not isinstance(raw, dict):
            continue
        imports = tuple(str(item).strip() for item in raw.get("allowed_imports") or ())
        imports = tuple(item for item in imports if item)
        package_allowlist = {
            str(import_name).strip(): str(package_spec).strip()
            for import_name, package_spec in dict(raw.get("package_allowlist") or {}).items()
            if str(import_name).strip() and str(package_spec).strip()
        }
        base = DEFAULT_ENVIRONMENT_PROFILES.get(profile_name)
        profiles[profile_name] = CodeEnvironmentProfile(
            name=profile_name,
            allowed_imports=imports or (base.allowed_imports if base else ()),
            package_allowlist=package_allowlist
            if raw.get("package_allowlist") is not None
            else (dict(base.package_allowlist) if base else {}),
            description=str(raw.get("description") or (base.description if base else "")),
        )
    return profiles


def _import_available(import_root: str) -> bool:
    try:
        return importlib.util.find_spec(import_root) is not None
    except (ImportError, ValueError):
        return False
