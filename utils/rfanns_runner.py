import csv
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from utils.rfanns_metrics import parse_metrics_file, utc_now_iso


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


def build_benchmark_command(
    python_bin: str,
    script_path: str,
    params: Dict[str, Any],
    metrics_output_path: str,
    metrics_output_arg: str = "--metrics-output",
    extra_args: Sequence[str] | None = None,
    param_args: Mapping[str, str] | None = None,
) -> List[str]:
    script_name = Path(script_path).name
    if script_name == "search_hsig.py":
        command = [
            python_bin,
            script_path,
            "--al_list",
            _to_cli_number(params["al"]),
            "--B",
            _to_cli_number(params["B"]),
            "--ef_list",
            _to_cli_number(params["ef"]),
            "--efConstruction",
            _to_cli_number(params["efConstruction"]),
            "--M",
            _to_cli_number(params["M"]),
            "--metrics-output",
            metrics_output_path,
        ]
        # Pass recall threshold if the tuning pipeline provides it as a param.
        select_recall = params.get("_select_recall_threshold")
        if select_recall is not None:
            command.extend(["--select-recall-threshold", _to_cli_number(select_recall)])
        select_slack = params.get("_select_recall_slack")
        if select_slack is not None:
            command.extend(["--select-recall-slack", _to_cli_number(select_slack)])
        if extra_args:
            command.extend(list(extra_args))
        return command

    if param_args:
        command = [
            python_bin,
            script_path,
        ]
        for name, flag in param_args.items():
            if name not in params:
                raise KeyError(f"Missing benchmark parameter '{name}'")
            command.extend([str(flag), _to_cli_number(params[name])])
        command.extend([metrics_output_arg, metrics_output_path])
        if extra_args:
            command.extend(list(extra_args))
        return command

    command = [
        python_bin,
        script_path,
        "--al",
        _to_cli_number(params["al"]),
        "--B",
        _to_cli_number(params["B"]),
        "--ef",
        _to_cli_number(params["ef"]),
        "--efConstruction",
        _to_cli_number(params["efConstruction"]),
        "--M",
        _to_cli_number(params["M"]),
        metrics_output_arg,
        metrics_output_path,
    ]
    if extra_args:
        command.extend(list(extra_args))
    return command


def _infer_executor_mode(script_path: str) -> str:
    if Path(script_path).name == "search_hsig.py":
        return "search_hsig_direct"
    if Path(script_path).name == "diskann_memory_filtered_benchmark.py":
        return "diskann_memory_filtered_direct"
    return "generic"


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


def _extract_flag_value(extra_args: Sequence[str], flag: str) -> str | None:
    for idx, token in enumerate(extra_args):
        if token != flag:
            continue
        if idx + 1 >= len(extra_args):
            return None
        nxt = extra_args[idx + 1]
        if isinstance(nxt, str) and nxt.startswith("--"):
            return None
        return str(nxt)
    return None


def _resolve_cli_path(value: str, workdir: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(workdir) / path


def _parse_result_save_path(extra_args: Sequence[str], workdir: str) -> Path:
    raw = _extract_flag_value(extra_args, "--result_save_path")
    if not raw:
        raise ValueError("Missing required benchmark extra arg: --result_save_path")
    return _resolve_cli_path(raw, workdir)


def _parse_index_cache_path(extra_args: Sequence[str], workdir: str) -> Path:
    raw = _extract_flag_value(extra_args, "--index_cache_path")
    if not raw:
        raise ValueError("Missing required benchmark extra arg: --index_cache_path")
    return _resolve_cli_path(raw, workdir)


def _delete_path_if_exists(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _parse_metrics_from_result_csv(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Result CSV is empty: {csv_path}")

    row = rows[0]
    if "recall" not in row or "QPS" not in row:
        raise ValueError(f"Result CSV missing required columns 'recall'/'QPS': {csv_path}")

    metrics: Dict[str, Any] = {
        "recall": float(row["recall"]),
        "qps": float(row["QPS"]),
    }
    latency = row.get("latency(ms)")
    if latency not in (None, ""):
        metrics["latency_ms_p95"] = float(latency)
    return metrics


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
    script_name = Path(config.script_path).name
    use_search_hsig_direct = script_name == "search_hsig.py"
    result_save_path: Path | None = None
    index_cache_path: Path | None = None
    parse_error = None
    if use_search_hsig_direct:
        try:
            result_save_path = _parse_result_save_path(config.extra_args, config.workdir)
            index_cache_path = _parse_index_cache_path(config.extra_args, config.workdir)
            result_save_path.parent.mkdir(parents=True, exist_ok=True)
            index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            parse_error = str(exc)

    command = build_benchmark_command(
        python_bin=config.python_bin,
        script_path=config.script_path,
        params=params,
        metrics_output_path=str(metrics_path),
        metrics_output_arg=config.metrics_output_arg,
        extra_args=config.extra_args,
        param_args=config.param_args,
    )
    if parse_error:
        executor_mode = _infer_executor_mode(config.script_path)
        proposal_meta = proposal_meta or {}
        return {
            "run_id": run_id,
            "stage": stage,
            "params": params,
            "metrics": None,
            "repeat_idx": repeat_idx,
            "status": "failed",
            "error": f"Runner config error: {parse_error}",
            "started_at": started_at,
            "elapsed_s": 0.0,
            "command": command,
            "metrics_path": str(metrics_path),
            "executor_script": config.script_path,
            "executor_mode": executor_mode,
            "benchmark_artifacts": {
                "result_save_path": str(result_save_path) if result_save_path else "",
                "csv_deleted_before": False,
                "csv_parse_error": parse_error,
                "index_cache_path": str(index_cache_path) if index_cache_path else "",
                "index_deleted_before": False,
                "index_time_deleted_before": False,
                "index_delete_error": parse_error,
            },
            "proposal_source": proposal_meta.get("proposal_source", "unknown"),
            "proposal_round": proposal_meta.get("proposal_round", -1),
            "proposal_note": proposal_meta.get("proposal_note", ""),
            "task_name": proposal_meta.get("task_name", ""),
        }

    max_attempts = max(1, config.retries + 1)
    start_ts = time.time()
    metrics = None
    error_msg = None
    csv_deleted_before = False
    csv_parse_error = ""
    index_deleted_before = False
    index_time_deleted_before = False
    index_delete_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            if use_search_hsig_direct and result_save_path and index_cache_path:
                try:
                    index_deleted_before = _delete_path_if_exists(index_cache_path) or index_deleted_before
                    index_time_deleted_before = (
                        _delete_path_if_exists(Path(str(index_cache_path) + ".time"))
                        or index_time_deleted_before
                    )
                except Exception as exc:
                    index_delete_error = str(exc)
                    error_msg = (
                        f"Index cleanup failed attempt={attempt}/{max_attempts}: {exc}"
                    )
                    break

                if result_save_path.exists():
                    result_save_path.unlink()
                    csv_deleted_before = True

            # Ensure UNIFY directory is on PYTHONPATH for hannlib import.
            env = os.environ.copy()
            if use_search_hsig_direct:
                unify_dir = str(Path(config.workdir) / "UNIFY")
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{unify_dir}:{existing}" if existing else unify_dir

            completed = subprocess.run(
                command,
                cwd=config.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
                check=False,
            )

            if completed.returncode != 0:
                error_msg = (
                    f"Command failed (code={completed.returncode}) "
                    f"attempt={attempt}/{max_attempts}"
                )
                continue

            if use_search_hsig_direct:
                try:
                    assert result_save_path is not None
                    # Prefer the JSON metrics file if it was written.
                    if metrics_path.exists():
                        metrics = parse_metrics_file(metrics_path)
                    else:
                        metrics = _parse_metrics_from_result_csv(result_save_path)
                    csv_parse_error = ""
                except Exception as exc:
                    csv_parse_error = str(exc)
                    error_msg = (
                        f"CSV/JSON parse failed attempt={attempt}/{max_attempts}: {exc}"
                    )
                    continue
            else:
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
    if use_search_hsig_direct:
        benchmark_artifacts = {
            "result_save_path": str(result_save_path) if result_save_path else "",
            "csv_deleted_before": csv_deleted_before,
            "csv_parse_error": csv_parse_error,
            "index_cache_path": str(index_cache_path) if index_cache_path else "",
            "index_deleted_before": index_deleted_before,
            "index_time_deleted_before": index_time_deleted_before,
            "index_delete_error": index_delete_error,
        }
    else:
        benchmark_artifacts = _load_artifacts_sidecar(metrics_path)
    executor_mode = _infer_executor_mode(config.script_path)
    if (
        status == "success"
        and executor_mode == "diskann_memory_filtered_direct"
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
        "metrics_path": str(metrics_path) if use_search_hsig_direct and metrics_path.exists() else (str(result_save_path) if use_search_hsig_direct and result_save_path else str(metrics_path)),
        "executor_script": config.script_path,
        "executor_mode": executor_mode,
        "benchmark_artifacts": benchmark_artifacts,
        "proposal_source": proposal_meta.get("proposal_source", "unknown"),
        "proposal_round": proposal_meta.get("proposal_round", -1),
        "proposal_note": proposal_meta.get("proposal_note", ""),
        "task_name": proposal_meta.get("task_name", ""),
    }


__all__ = [
    "RunnerConfig",
    "build_benchmark_command",
    "run_trial",
]
