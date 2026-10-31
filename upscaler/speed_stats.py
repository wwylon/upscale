"""Learn MP/s from completed upscales and feed time estimates."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .benchmark import user_config_dir
from .estimates import megapixels
from .pipeline import compute_passes

_lock = threading.Lock()
MAX_RUNS = 200
MIN_SECONDS = 0.4
MIN_WORK_MP = 0.005


def speed_stats_path() -> Path:
    return user_config_dir() / "speed_stats.json"


def normalize_device(device: str | None) -> str:
    d = (device or "cpu").lower()
    if d in ("auto",):
        return "cpu"
    if "cuda" in d or "directml" in d:
        return "cuda"
    if d.startswith("mps") or d == "mps":
        return "mps"
    if "cpu" in d:
        return "cpu"
    return d


def work_megapixels(
    src_w: int,
    src_h: int,
    scale: float,
    *,
    model_scale: int = 4,
    algorithm: str | None = None,
) -> float:
    """Same work units used by ``estimate_time_seconds`` (and recording)."""
    if scale <= 1.0 and not algorithm:
        return 0.0
    if algorithm:
        # Classical is one resize; charge by output megapixels.
        return megapixels(max(1, round(src_w * scale)), max(1, round(src_h * scale)))
    passes = compute_passes(scale, model_scale)
    if passes == 0:
        return 0.0
    work = 0.0
    w, h = src_w, src_h
    ms = max(2, int(model_scale))
    for _ in range(passes):
        work += megapixels(w, h)
        w *= ms
        h *= ms
    return work


def _sample_key(
    *,
    model: str,
    device: str,
    half: bool,
    algorithm: str | None,
) -> str:
    dev = normalize_device(device)
    if algorithm:
        return f"algo:{algorithm}|{dev}"
    return f"{model}|{dev}|h{1 if half else 0}"


def load_speed_stats() -> list[dict[str, Any]]:
    path = speed_stats_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        runs = raw.get("runs")
        return runs if isinstance(runs, list) else []
    return []


def _save_speed_stats(runs: list[dict[str, Any]]) -> None:
    path = speed_stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"runs": runs[-MAX_RUNS:], "updated_at": time.time()}, indent=2),
        encoding="utf-8",
    )


def record_speed_sample(
    *,
    model: str,
    device: str,
    seconds: float,
    work_mp: float,
    scale: float,
    src_w: int,
    src_h: int,
    half: bool = False,
    algorithm: str | None = None,
    model_scale: int = 4,
) -> dict[str, Any] | None:
    """Persist one successful run. Returns the entry or None if skipped."""
    if seconds < MIN_SECONDS or work_mp < MIN_WORK_MP:
        return None
    mp_s = work_mp / seconds
    if mp_s <= 0 or mp_s > 500:  # absurd outliers (bad dims / clock)
        return None
    entry = {
        "key": _sample_key(
            model=model, device=device, half=half, algorithm=algorithm
        ),
        "model": model,
        "device": normalize_device(device),
        "half": bool(half),
        "algorithm": algorithm,
        "model_scale": int(model_scale),
        "scale": float(scale),
        "src_w": int(src_w),
        "src_h": int(src_h),
        "work_mp": round(work_mp, 5),
        "seconds": round(seconds, 3),
        "mp_s": round(mp_s, 5),
        "ts": time.time(),
    }
    with _lock:
        runs = load_speed_stats()
        runs.append(entry)
        _save_speed_stats(runs)
    return entry


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2.0


def lookup_measured_mp_s(
    *,
    model: str,
    device: str,
    half: bool = False,
    algorithm: str | None = None,
    work_mp: float | None = None,
    min_samples: int = 1,
) -> dict[str, Any] | None:
    """Return ``{mp_s, n, key}`` from past runs, preferring similar workload."""
    dev = normalize_device(device)
    exact = _sample_key(model=model, device=dev, half=half, algorithm=algorithm)
    soft = (
        f"algo:{algorithm}|{dev}"
        if algorithm
        else f"{model}|{dev}|h{0}"  # prefer fp32 bucket as soft fallback
    )
    with _lock:
        runs = load_speed_stats()

    def pick(predicate) -> list[dict[str, Any]]:
        return [r for r in runs if predicate(r)]

    buckets: list[list[dict[str, Any]]] = [
        pick(lambda r: r.get("key") == exact),
        pick(
            lambda r: r.get("model") == model
            and r.get("device") == dev
            and (r.get("algorithm") or None) == (algorithm or None)
        ),
        pick(lambda r: r.get("key") == soft) if not algorithm else [],
    ]
    chosen: list[dict[str, Any]] = []
    used_key = exact
    for i, bucket in enumerate(buckets):
        if len(bucket) >= min_samples:
            chosen = bucket
            used_key = str(bucket[-1].get("key") or exact)
            if i == 0:
                used_key = exact
            break
    if not chosen:
        return None

    # Prefer runs within ~4× of the planned workload when enough exist.
    pool = chosen[-40:]
    if work_mp and work_mp > 0:
        similar = [
            r
            for r in pool
            if 0.25 * work_mp <= float(r.get("work_mp") or 0) <= 4.0 * work_mp
        ]
        if len(similar) >= min_samples:
            pool = similar

    rates = [float(r["mp_s"]) for r in pool if float(r.get("mp_s") or 0) > 0]
    if len(rates) < min_samples:
        return None
    return {
        "mp_s": round(_median(rates), 5),
        "n": len(rates),
        "key": used_key,
    }


def record_from_job(
    *,
    model: str,
    device: str,
    seconds: float,
    src_w: int,
    src_h: int,
    scale: float,
    half: bool = False,
    algorithm: str | None = None,
    model_scale: int = 4,
) -> dict[str, Any] | None:
    work = work_megapixels(
        src_w,
        src_h,
        scale,
        model_scale=model_scale,
        algorithm=algorithm,
    )
    return record_speed_sample(
        model=model,
        device=device,
        seconds=seconds,
        work_mp=work,
        scale=scale,
        src_w=src_w,
        src_h=src_h,
        half=half,
        algorithm=algorithm,
        model_scale=model_scale,
    )
