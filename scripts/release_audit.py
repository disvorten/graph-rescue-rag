"""Fail-fast audit for the public Graph Rescue RAG Git snapshot."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
FORBIDDEN_PARTS = {
    "datasets",
    "cache",
    "caches",
    "models",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_SUFFIXES = {".pid", ".log", ".aux", ".bbl", ".blg", ".out"}
REQUIRED = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "DATA.md",
    "REPRODUCING.md",
    "CITATION.cff",
    ".zenodo.json",
    "outputs/final_v1/analysis/policy_metrics.csv",
    "outputs/published_baselines/comparison.csv",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def tracked_files() -> list[Path]:
    result = run(
        "git",
        "-c",
        f"safe.directory={ROOT.as_posix()}",
        "ls-files",
    )
    if result.returncode:
        raise RuntimeError(
            "Git repository is not initialized or git ls-files failed:\n"
            + result.stderr.strip()
        )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def audit_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    relative = {
        path.relative_to(ROOT).as_posix()
        for path in files
        if path.exists()
    }
    for required in sorted(REQUIRED - relative):
        errors.append(f"required tracked artifact is missing: {required}")

    for path in files:
        rel = path.relative_to(ROOT)
        rel_text = rel.as_posix()
        parts = {part.lower() for part in rel.parts}
        if not path.exists():
            errors.append(f"tracked path does not exist: {rel_text}")
            continue
        if path.stat().st_size > MAX_BYTES:
            errors.append(
                f"tracked file exceeds {MAX_BYTES // (1024 * 1024)} MiB: "
                f"{rel_text} ({path.stat().st_size} bytes)"
            )
        if FORBIDDEN_PARTS & parts:
            errors.append(f"forbidden data/cache path is tracked: {rel_text}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated suffix is tracked: {rel_text}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {rel_text}")
    return errors


def main() -> int:
    try:
        files = tracked_files()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    errors = audit_files(files)
    citation = ROOT / "CITATION.cff"
    if citation.exists() and "REPLACE-WITH" in citation.read_text(
        encoding="utf-8", errors="replace"
    ):
        errors.append("CITATION.cff still contains author placeholders")

    tests = run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v")
    if tests.returncode:
        errors.append("test suite failed; inspect output below")

    if errors:
        print("RELEASE AUDIT: FAIL")
        for error in errors:
            print(f"- {error}")
        if tests.returncode:
            print(tests.stdout)
            print(tests.stderr)
        return 1

    print(f"RELEASE AUDIT: PASS ({len(files)} tracked files)")
    print("Test suite: PASS")
    print("Independent author verification remains required before archival release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
