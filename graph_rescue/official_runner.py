from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_ROOT = PROJECT_ROOT / "work" / "official_evaluators"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_metrics(stdout: str) -> dict[str, float]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError(f"Official evaluator returned no metric object: {stdout}")
    payload = stdout[start:].strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = ast.literal_eval(payload)
    if not isinstance(value, dict):
        raise ValueError("Official evaluator output is not a dictionary")
    return {str(key): float(metric) for key, metric in value.items()}


def score_official_predictions(
    *,
    dataset: str,
    prediction_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path | None = None,
    alias_path: str | Path | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    gold = Path(gold_path).resolve()
    if dataset == "hotpot":
        evaluator = EVALUATOR_ROOT / "hotpot" / "hotpot_evaluate_v1.py"
        shim = (
            "import json,runpy,sys;"
            "sys.modules['ujson']=json;"
            "sys.argv=sys.argv[1:];"
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        command = [
            sys.executable,
            "-c",
            shim,
            str(evaluator),
            str(prediction),
            str(gold),
        ]
        cwd = evaluator.parent
    elif dataset == "2wiki":
        if alias_path is None:
            raise ValueError("2WikiMultiHopQA scoring requires alias_path")
        evaluator = (
            EVALUATOR_ROOT
            / "2wiki"
            / "2wikimultihop_evaluate_v1.1.py"
        )
        aliases = Path(alias_path).resolve()
        # The untouched official script imports optional ujson. Injecting the
        # standard-library json module preserves its API and avoids altering it.
        shim = (
            "import json,runpy,sys;"
            "sys.modules['ujson']=json;"
            "sys.argv=sys.argv[1:];"
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        command = [
            sys.executable,
            "-c",
            shim,
            str(evaluator),
            str(prediction),
            str(gold),
            str(aliases),
        ]
        cwd = evaluator.parent
    elif dataset == "musique":
        evaluator = EVALUATOR_ROOT / "musique" / "evaluate_v1.0.py"
        command = [sys.executable, str(evaluator), str(prediction), str(gold)]
        cwd = evaluator.parent
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not evaluator.is_file():
        raise FileNotFoundError(
            f"Official {dataset} evaluator was not found at {evaluator}. "
            "Download the dataset's official scoring script before running "
            "official evaluation."
        )

    evaluator_env = os.environ.copy()
    evaluator_env["PYTHONUTF8"] = "1"
    evaluator_env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=evaluator_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Official {dataset} evaluator failed with code "
            f"{completed.returncode}:\n{completed.stderr}"
        )
    result: dict[str, Any] = {
        "dataset": dataset,
        "metrics": _parse_metrics(completed.stdout),
        "prediction_path": str(prediction),
        "gold_path": str(gold),
        "evaluator_path": str(evaluator),
        "evaluator_sha256": _sha256(evaluator),
        "stdout": completed.stdout.strip(),
    }
    if alias_path is not None:
        result["alias_path"] = str(Path(alias_path).resolve())
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["output_path"] = str(output.resolve())
    return result
