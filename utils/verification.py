import os
import re


RUNTIME_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt"}
DEFAULT_SKIP_DIRS = {
    ".agents",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "env",
    "generated",
    "output",
    "outputs",
    "reference",
    "venv",
}
DEFAULT_PATTERNS = {
    "windows_absolute_path": re.compile(r"[A-Za-z]:\\"),
    "home_path": re.compile(r"/home/"),
    "public_home_path": re.compile(r"/public/home/"),
    "reference_import": re.compile(r"^\s*(from|import)\s+reference(\.|\s|$)", re.MULTILINE),
    "credential_hint": re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]", re.IGNORECASE),
}


def iter_runtime_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in DEFAULT_SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.splitext(name)[1].lower() in RUNTIME_SUFFIXES:
                yield path


def scan_runtime_files(root, patterns=None):
    patterns = patterns or DEFAULT_PATTERNS
    hits = []
    self_path = os.path.abspath(__file__)
    for path in iter_runtime_files(root):
        # Ignore the scanner's own pattern definitions.
        if os.path.abspath(path) == self_path:
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append({"file": path, "line": line, "kind": name, "text": match.group(0)})
    return hits
