"""Build and verify the flat JIIS LaTeX submission archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "jiis_submission"
OUTPUT = SOURCE / "Graph_Rescue_RAG_JIIS_submission.zip"
FILES = (
    "main.tex",
    "generated_results.tex",
    "failure_analysis.tex",
    "extended_results.tex",
    "references.bib",
    "main.bbl",
    "svjour3.cls",
    "svglov3.clo",
    "spmpsci.bst",
    "Fig1.png",
    "main.pdf",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [name for name in FILES if not (SOURCE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing JIIS submission files: {missing}")
    if (SOURCE / "main.pdf").stat().st_size < 10_000:
        raise ValueError("Compiled main.pdf is unexpectedly small")

    with zipfile.ZipFile(
        OUTPUT,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in FILES:
            archive.write(SOURCE / name, arcname=name)

    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if tuple(names) != FILES:
            raise ValueError(f"Unexpected archive members: {names}")
        if any("/" in name or "\\" in name for name in names):
            raise ValueError("JIIS archive must be flat")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt archive member: {bad}")

    print(f"archive={OUTPUT}")
    print(f"archive_sha256={sha256(OUTPUT)}")
    for name in FILES:
        print(f"{name}={sha256(SOURCE / name)}")


if __name__ == "__main__":
    main()
