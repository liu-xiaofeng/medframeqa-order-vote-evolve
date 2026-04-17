#!/home1/xl693/anaconda3/envs/vlm/bin/python
"""MedFrameQA 后处理验证脚本。

这个脚本服务于论文阶段的“补实验”，不改任何方法定义，只做额外评测：
- 对每条方法线的 `selected_generation` 和 `gen_0` 做 post-hoc 评测；
- 产出逐题预测，便于后续做 paired bootstrap；
- 产出 modality / image_count breakdown；
- 同时把跨方法汇总写到 paper_analysis_output，方便 notebook 直接读。

设计原则：
- 复用现有 task 目录下的 `initial.py/evaluate.py` 逻辑；
- 不修改已有 run 目录里的演化结果；
- 新评测结果统一写到 `paper_eval/posthoc_*`。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import string
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/gluon4/xl693/evolve")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medframeqa_runtime import (
    acquire_vlm_lock,
    deterministic_letter_fallback,
    extract_option_letter,
    generate_guided_choice_with_retries,
    get_image_columns,
    get_protocol_meta,
    get_protocol_score_key,
    get_protocol_subset,
    load_medframeqa_dataset,
    load_mutated_module,
    load_split_manifest,
    make_invalid_metrics,
    make_openai_client,
    make_protocol_metrics,
    protocol_alias,
    write_csv_rows,
    write_json,
)


RESULTS_ROOT = ROOT / "results"
OUTPUT_ROOT = ROOT / "paper_analysis_output" / "posthoc_eval"
VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")

METHOD_SPECS = {
    "fixed": {
        "task_dir": "advanced_vqa_task_fixed",
        "task_name": "advanced_vqa_task_fixed",
        "search_mini_size": 128,
        "api_timeout": 30.0,
    },
    "reasoning": {
        "task_dir": "advanced_vqa_task_reasoning",
        "task_name": "advanced_vqa_task_reasoning",
        "search_mini_size": 128,
        "api_timeout": 30.0,
    },
    "order_vote": {
        "task_dir": "advanced_vqa_task_order_vote",
        "task_name": "advanced_vqa_task_order_vote",
        "search_mini_size": 128,
        "api_timeout": 30.0,
    },
    "order_rerank": {
        "task_dir": "advanced_vqa_task_order_rerank",
        "task_name": "advanced_vqa_task_order_rerank",
        "search_mini_size": 64,
        "api_timeout": 20.0,
    },
    "order_vote_plus": {
        "task_dir": "advanced_vqa_task_order_vote_plus",
        "task_name": "advanced_vqa_task_order_vote_plus",
        "search_mini_size": 128,
        "api_timeout": 30.0,
    },
}

METHOD_PREFIXES = {
    "fixed": ("advanced_vqa_task_fixed_", ()),
    "reasoning": ("advanced_vqa_task_reasoning_", ()),
    "order_vote": ("advanced_vqa_task_order_vote_", ("advanced_vqa_task_order_vote_plus_",)),
    "order_rerank": ("advanced_vqa_task_order_rerank_", ()),
    "order_vote_plus": ("advanced_vqa_task_order_vote_plus_", ()),
}

MODALITY_ORDER = ["CT", "MRI", "X-ray", "ultrasound", "other"]
IMAGE_COUNT_ORDER = [2, 3, 4, 5]
EXPECTED_IMAGE_COUNT_SET = set(IMAGE_COUNT_ORDER)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def match_run_dir(method: str, run_dir: Path) -> bool:
    prefix, excluded = METHOD_PREFIXES[method]
    if not run_dir.name.startswith(prefix):
        return False
    return not any(token in run_dir.name for token in excluded)


def find_latest_complete_run(method: str, results_root: Path, min_generations: int = 50) -> Path:
    candidates = sorted(p for p in results_root.iterdir() if p.is_dir() and match_run_dir(method, p))
    completed = []
    for run_dir in candidates:
        summary_path = run_dir / "paper_eval" / "paper_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if int(summary.get("generation_count") or 0) < min_generations:
            continue
        completed.append(run_dir)
    if not completed:
        raise FileNotFoundError(f"找不到完整 run: {method}")
    return completed[-1]


def resolve_program_path(run_dir: Path, target: str) -> tuple[int, Path]:
    summary = load_json(run_dir / "paper_eval" / "paper_summary.json")
    if target == "selected":
        generation = summary.get("selected_generation")
        if generation is None:
            raise RuntimeError(f"{run_dir} 缺少 selected_generation")
    elif target == "gen0":
        generation = 0
    elif target.startswith("gen_") and target[4:].isdigit():
        generation = int(target[4:])
    elif target.isdigit():
        generation = int(target)
    else:
        raise ValueError(f"Unsupported target: {target}")

    program_path = run_dir / f"gen_{generation}" / "main.py"
    if not program_path.exists():
        raise FileNotFoundError(f"找不到 program_path: {program_path}")
    return generation, program_path


def make_simple_sample_result(program_module, sample, client, call_stats):
    valid_letters = list(string.ascii_uppercase[: len(sample["options"])])
    gt = sample["correct_answer"].strip().upper()
    raw = generate_guided_choice_with_retries(
        client,
        lambda sample=sample: program_module.format_vqa_payload(sample),
        valid_letters,
        VLM_MODEL,
        f"[posthoc_simple] QID={sample.get('question_id', 'UNKNOWN')}",
        call_stats=call_stats,
    )
    pred = extract_option_letter(raw, valid_letters)
    missing_pred = int(not pred)
    if not pred:
        pred = deterministic_letter_fallback(valid_letters, "alphabetical")
    return {
        "gt": gt,
        "pred": pred,
        "raw": raw,
        "missing_pred_count": missing_pred,
    }


def scalarize_result(result: dict[str, Any]) -> dict[str, Any]:
    scalars = {}
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalars[key] = value
    return scalars


def make_prediction_row(sample: dict[str, Any], result: dict[str, Any], generation: int) -> dict[str, Any]:
    image_count = len(get_image_columns(sample))
    row = {
        "question_id": sample.get("question_id"),
        "video_id": sample.get("video_id"),
        "system": sample.get("system"),
        "organ": sample.get("organ"),
        "keyword": sample.get("keyword"),
        "modality": sample.get("modality"),
        "image_count": image_count,
        "option_count": len(sample.get("options", [])),
        "generation": generation,
        "gt": result.get("gt"),
        "pred": result.get("pred"),
        "correct": int(result.get("pred") == result.get("gt")),
    }
    row.update(scalarize_result(result))
    return row


def build_breakdown_rows(predictions: list[dict[str, Any]], field: str, ordered_values: list[Any] | None = None):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row.get(field)].append(row)

    rows = []
    values = list(ordered_values or [])
    for value in sorted(grouped.keys()):
        if value not in values:
            values.append(value)

    for value in values:
        members = grouped.get(value, [])
        correct = sum(row.get("correct", 0) for row in members)
        size = len(members)
        rows.append(
            {
                field: value,
                "accuracy": (correct / size) if size else 0.0,
                "correct": correct,
                "size": size,
            }
        )
    return rows


def validate_protocol_outputs(
    protocol_mode: str,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    image_count_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    对 post-hoc 结果做协议级检查。

    这里刻意把 image_count 当成硬检查：
    - MedFrameQA 官方定义每题只有 2-5 张图；
    - 如果这里再次出现 6，就说明运行时又把 image_url 算进去了。
    """
    observed_counts = sorted({row.get("image_count") for row in predictions})
    invalid_counts = [count for count in observed_counts if count not in EXPECTED_IMAGE_COUNT_SET]
    total_from_rows = sum(int(row.get("size") or 0) for row in image_count_rows)
    score_key = f"{protocol_alias(protocol_mode)}_size"
    expected_total = int(metrics.get(score_key) or 0)

    checks = {
        "observed_image_counts": observed_counts,
        "invalid_image_counts": invalid_counts,
        "image_count_bucket_total": total_from_rows,
        "expected_total": expected_total,
        "image_count_bucket_ok": not invalid_counts,
        "image_count_total_ok": total_from_rows == expected_total,
    }
    if invalid_counts:
        raise RuntimeError(
            "Invalid image_count buckets detected. "
            f"Observed={observed_counts}, expected subset={sorted(EXPECTED_IMAGE_COUNT_SET)}"
        )
    if total_from_rows != expected_total:
        raise RuntimeError(
            "Image-count breakdown total does not match protocol size. "
            f"bucket_total={total_from_rows}, expected_total={expected_total}"
        )
    return checks


def evaluate_program(method: str, run_dir: Path, generation: int, program_path: Path, protocol_mode: str, out_dir: Path, subset_limit: int | None = None):
    spec = METHOD_SPECS[method]
    eval_module = import_module_from_path(ROOT / spec["task_dir"] / "evaluate.py", f"{method}_eval_mod")
    program_module = load_mutated_module(program_path)
    dataset = load_medframeqa_dataset(include_images=True)
    manifest = load_split_manifest()
    subset, selected_ids = get_protocol_subset(dataset, manifest, protocol_mode, spec["search_mini_size"])
    if subset_limit is not None:
        subset = subset.select(range(min(subset_limit, len(subset))))
        selected_ids = selected_ids[: min(subset_limit, len(selected_ids))]

    client = make_openai_client(VLM_URL, timeout=spec["api_timeout"])
    call_stats = {"vlm_call_count": 0}
    start_time = time.time()
    lock_wait = 0.0
    prediction_rows = []
    prediction_json_rows = []

    try:
        with acquire_vlm_lock(spec["task_name"], results_dir=str(out_dir), mode=f"posthoc_{protocol_mode}") as lock_info:
            lock_wait = float(lock_info.get("waited_for_lock_sec", 0.0))
            for sample in subset:
                if method in {"fixed", "reasoning"}:
                    result = make_simple_sample_result(program_module, sample, client, call_stats)
                else:
                    result = eval_module.evaluate_sample(program_module, sample, client, call_stats)
                prediction_rows.append(make_prediction_row(sample, result, generation))
                prediction_json_rows.append(
                    {
                        **make_prediction_row(sample, result, generation),
                        "debug": result,
                    }
                )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        metrics = make_invalid_metrics(protocol_mode, spec["search_mini_size"], type(exc).__name__, str(exc))
        metrics.update(get_protocol_meta(manifest, protocol_mode, selected_ids, spec["search_mini_size"]))
        metrics["wall_time_sec"] = round(time.time() - start_time, 3)
        metrics["vlm_lock_wait_sec"] = round(lock_wait, 3)
        metrics["generation"] = generation
        metrics["program_path"] = str(program_path)
        return metrics, prediction_rows, [], []

    correct = sum(row["correct"] for row in prediction_rows)
    total = len(prediction_rows)
    metrics = make_protocol_metrics(protocol_mode, correct, total)
    metrics.update(get_protocol_meta(manifest, protocol_mode, selected_ids, spec["search_mini_size"]))
    metrics["combined_score"] = metrics[get_protocol_score_key(protocol_mode)]
    metrics["generation"] = generation
    metrics["program_path"] = str(program_path)
    metrics["vlm_call_count"] = call_stats["vlm_call_count"]
    metrics["invalid_generation"] = 0
    metrics["error_type"] = ""
    metrics["error_message"] = ""
    metrics["wall_time_sec"] = round(time.time() - start_time, 3)
    metrics["vlm_lock_wait_sec"] = round(lock_wait, 3)

    for key in ("missing_pred_count", "missing_vote_count", "all_votes_missing", "uncertainty_triggered"):
        matching_keys = [row.get(key, 0) for row in prediction_rows if row.get(key) is not None]
        if not matching_keys:
            continue
        if key == "all_votes_missing":
            metrics["all_votes_missing_count"] = sum(matching_keys)
        elif key == "uncertainty_triggered":
            metrics["uncertainty_trigger_count"] = sum(matching_keys)
        else:
            metrics[key] = sum(matching_keys)

    modality_rows = build_breakdown_rows(prediction_rows, "modality", MODALITY_ORDER)
    image_count_rows = build_breakdown_rows(prediction_rows, "image_count", IMAGE_COUNT_ORDER)
    return metrics, prediction_json_rows, modality_rows, image_count_rows


def save_posthoc_outputs(
    out_dir: Path,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    modality_rows: list[dict[str, Any]],
    image_count_rows: list[dict[str, Any]],
    protocol_checks: dict[str, Any],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "predictions.json", predictions)
    write_csv_rows(out_dir / "predictions.csv", predictions)
    write_json(out_dir / "breakdown_modality.json", modality_rows)
    write_csv_rows(out_dir / "breakdown_modality.csv", modality_rows)
    write_json(out_dir / "breakdown_image_count.json", image_count_rows)
    write_csv_rows(out_dir / "breakdown_image_count.csv", image_count_rows)
    write_json(out_dir / "protocol_checks.json", protocol_checks)


def posthoc_dir_name(target: str, protocol_mode: str) -> str:
    return f"posthoc_{target}_{protocol_alias(protocol_mode)}"


def main():
    parser = argparse.ArgumentParser(description="对 MedFrameQA 现有 run 做 post-hoc 最终验证。")
    parser.add_argument("--methods", nargs="+", default=list(METHOD_SPECS.keys()), choices=sorted(METHOD_SPECS))
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--protocol-mode", default="independent_final_test")
    parser.add_argument("--targets", nargs="+", default=["selected", "gen0"])
    parser.add_argument("--subset-limit", type=int, default=None)
    parser.add_argument("--min-generations", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--global-output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    args.global_output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    run_catalog = {}

    for method in args.methods:
        run_dir = find_latest_complete_run(method, args.results_root, min_generations=args.min_generations)
        run_catalog[method] = str(run_dir)
        for target in args.targets:
            generation, program_path = resolve_program_path(run_dir, target)
            out_dir = run_dir / "paper_eval" / posthoc_dir_name(target, args.protocol_mode)

            use_cached = (
                not args.force
                and (out_dir / "metrics.json").exists()
                and (out_dir / "predictions.json").exists()
                and (out_dir / "breakdown_modality.json").exists()
                and (out_dir / "breakdown_image_count.json").exists()
            )

            if use_cached:
                metrics = load_json(out_dir / "metrics.json")
                modality_rows = load_json(out_dir / "breakdown_modality.json")
                image_count_rows = load_json(out_dir / "breakdown_image_count.json")
                predictions = load_json(out_dir / "predictions.json")
                try:
                    protocol_checks = validate_protocol_outputs(
                        args.protocol_mode,
                        metrics,
                        predictions,
                        image_count_rows,
                    )
                except Exception:
                    use_cached = False
                else:
                    protocol_check_path = out_dir / "protocol_checks.json"
                    if not protocol_check_path.exists():
                        write_json(protocol_check_path, protocol_checks)

            if not use_cached:
                metrics, predictions, modality_rows, image_count_rows = evaluate_program(
                    method=method,
                    run_dir=run_dir,
                    generation=generation,
                    program_path=program_path,
                    protocol_mode=args.protocol_mode,
                    out_dir=out_dir,
                    subset_limit=args.subset_limit,
                )
                protocol_checks = validate_protocol_outputs(
                    args.protocol_mode,
                    metrics,
                    predictions,
                    image_count_rows,
                )
                save_posthoc_outputs(
                    out_dir,
                    metrics,
                    predictions,
                    modality_rows,
                    image_count_rows,
                    protocol_checks,
                )

            summary_rows.append(
                {
                    "method": method,
                    "run_dir": str(run_dir),
                    "target": target,
                    "generation": generation,
                    "program_path": str(program_path),
                    "protocol_mode": args.protocol_mode,
                    "score": metrics.get(f"{protocol_alias(args.protocol_mode)}_score"),
                    "correct": metrics.get(f"{protocol_alias(args.protocol_mode)}_correct"),
                    "size": metrics.get(f"{protocol_alias(args.protocol_mode)}_size"),
                    "invalid_generation": metrics.get("invalid_generation"),
                    "vlm_call_count": metrics.get("vlm_call_count"),
                    "wall_time_sec": metrics.get("wall_time_sec"),
                    "output_dir": str(out_dir),
                }
            )
            print(
                f"{method:>15} | {target:>8} | gen={generation:<3} | "
                f"score={summary_rows[-1]['score']} | out={out_dir}"
            )

    selected_vs_gen0_rows = []
    grouped = defaultdict(dict)
    for row in summary_rows:
        grouped[row["method"]][row["target"]] = row
    for method, rows in grouped.items():
        selected = rows.get("selected")
        gen0 = rows.get("gen0")
        selected_vs_gen0_rows.append(
            {
                "method": method,
                "run_dir": run_catalog.get(method),
                "selected_generation": selected.get("generation") if selected else None,
                "selected_final_test_score": selected.get("score") if selected else None,
                "gen0_generation": gen0.get("generation") if gen0 else 0,
                "gen0_final_test_score": gen0.get("score") if gen0 else None,
                "delta_selected_minus_gen0": (
                    (selected.get("score") - gen0.get("score"))
                    if selected and gen0 and selected.get("score") is not None and gen0.get("score") is not None
                    else None
                ),
            }
        )

    write_json(args.global_output_dir / "posthoc_eval_rows.json", summary_rows)
    write_csv_rows(args.global_output_dir / "posthoc_eval_rows.csv", summary_rows)
    write_json(args.global_output_dir / "selected_vs_gen0.json", selected_vs_gen0_rows)
    write_csv_rows(args.global_output_dir / "selected_vs_gen0.csv", selected_vs_gen0_rows)
    write_json(args.global_output_dir / "posthoc_run_catalog.json", run_catalog)


if __name__ == "__main__":
    main()
