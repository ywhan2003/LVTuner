import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from utils.hnswlib_metrics import BUILD_PARAM_ORDER, PARAM_ORDER, parse_metrics_file, utc_now_iso


@dataclass
class RunnerConfig:
    python_bin: str
    script_path: str
    metrics_output_arg: str
    timeout_s: int
    retries: int
    output_dir: str
    workdir: str
    extra_args: Sequence[str] = field(default_factory=list)
    param_args: Mapping[str, str] | None = None


def _to_cli_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _default_param_args() -> Dict[str, str]:
    return {
        "M": "--M",
        "ef_construction": "--ef-construction",
        "ef": "--ef",
    }


def build_benchmark_command(
    python_bin: str,
    script_path: str,
    params: Dict[str, Any],
    metrics_output_path: str,
    metrics_output_arg: str = "--metrics-output",
    extra_args: Sequence[str] | None = None,
    param_args: Mapping[str, str] | None = None,
) -> List[str]:
    resolved_param_args = dict(param_args or _default_param_args())
    if set(resolved_param_args) != set(PARAM_ORDER):
        missing = [name for name in PARAM_ORDER if name not in resolved_param_args]
        extra = [name for name in resolved_param_args if name not in PARAM_ORDER]
        raise ValueError(f"HNSW benchmark param_args must match {PARAM_ORDER}; missing={missing}, extra={extra}")

    command = [python_bin, script_path]
    for name in BUILD_PARAM_ORDER:
        if name not in params:
            raise KeyError(f"Missing HNSW benchmark parameter '{name}'")
        command.extend([str(resolved_param_args[name]), _to_cli_number(params[name])])
    if "ef" not in BUILD_PARAM_ORDER:
        ef_value = params.get("ef")
        if ef_value is None:
            raise KeyError("Missing HNSW benchmark parameter 'ef'")
        command.extend([str(resolved_param_args["ef"]), _to_cli_number(ef_value)])
    command.extend([metrics_output_arg, metrics_output_path])
    if extra_args:
        command.extend(list(extra_args))
    return command


def _infer_executor_mode(script_path: str) -> str:
    if Path(script_path).name == "hnswlib_benchmark.py":
        return "hnswlib_direct"
    return "hnswlib_generic"


def _load_artifacts_sidecar(metrics_path: Path) -> Dict[str, Any]:
    sidecar_path = Path(str(metrics_path) + ".artifacts.json")
    if not sidecar_path.exists():
        return {}
    try:
        with sidecar_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        return {"artifact_error": f"sidecar_parse_failed: {exc}"}
    return {"artifact_error": "sidecar_not_dict"}


def run_trial(
    stage: str,
    params: Dict[str, Any],
    repeat_idx: int,
    config: RunnerConfig,
    proposal_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    started_at = utc_now_iso()
    run_dir = Path(config.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    command = build_benchmark_command(
        python_bin=config.python_bin,
        script_path=config.script_path,
        params=params,
        metrics_output_path=str(metrics_path),
        metrics_output_arg=config.metrics_output_arg,
        extra_args=config.extra_args,
        param_args=config.param_args,
    )

    max_attempts = max(1, config.retries + 1)
    start_ts = time.time()
    metrics = None
    error_msg = None
    for attempt in range(1, max_attempts + 1):
        try:
            completed = subprocess.run(
                command,
                cwd=config.workdir,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
                check=False,
            )

            if completed.returncode != 0:
                stderr_tail = (completed.stderr or "").strip()[-2000:]
                error_msg = (
                    f"Command failed (code={completed.returncode}) "
                    f"attempt={attempt}/{max_attempts}"
                )
                if stderr_tail:
                    error_msg += f": {stderr_tail}"
                continue

            metrics = parse_metrics_file(metrics_path)
            error_msg = None
            break
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {config.timeout_s}s attempt={attempt}/{max_attempts}"
        except Exception as exc:
            error_msg = f"Runner exception attempt={attempt}/{max_attempts}: {exc}"

    wall_clock_elapsed_s = round(time.time() - start_ts, 6)
    elapsed_s = wall_clock_elapsed_s
    status = "success" if metrics is not None else "failed"
    executor_mode = _infer_executor_mode(config.script_path)
    benchmark_artifacts = _load_artifacts_sidecar(metrics_path)
    if (
        status == "success"
        and executor_mode == "hnswlib_direct"
        and isinstance(metrics, dict)
        and bool(metrics.get("index_reused"))
        and metrics.get("original_build_time_s") is not None
    ):
        try:
            elapsed_s = round(wall_clock_elapsed_s + float(metrics["original_build_time_s"]), 6)
        except (TypeError, ValueError):
            elapsed_s = wall_clock_elapsed_s

    proposal_meta = proposal_meta or {}
    return {
        "run_id": run_id,
        "stage": stage,
        "params": params,
        "metrics": metrics,
        "repeat_idx": repeat_idx,
        "status": status,
        "error": error_msg,
        "started_at": started_at,
        "elapsed_s": elapsed_s,
        "wall_clock_elapsed_s": wall_clock_elapsed_s,
        "command": command,
        "metrics_path": str(metrics_path),
        "executor_script": config.script_path,
        "executor_mode": executor_mode,
        "benchmark_artifacts": benchmark_artifacts,
        "proposal_source": proposal_meta.get("proposal_source", "unknown"),
        "proposal_round": proposal_meta.get("proposal_round", -1),
        "proposal_note": proposal_meta.get("proposal_note", ""),
        "task_name": proposal_meta.get("task_name", ""),
    }


__all__ = ["RunnerConfig", "build_benchmark_command", "run_trial"]
