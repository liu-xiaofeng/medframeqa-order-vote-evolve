"""
固定/推理两条纯 prompt 任务线共用的评测器。

整体思路：
- 统一走共享 runtime；
- 整个评测过程受 8001 全局锁保护；
- 输出统一 metrics schema，便于 paper summary 汇总。
"""

from pathlib import Path
import sys


def _ensure_project_root():
    path = Path(__file__).resolve()
    candidates = [path.parent, *path.parents]
    for candidate in candidates:
        if (candidate / "medframeqa_runtime.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    return None


_ensure_project_root()

import argparse
import json
import os
import string
import sys
from pathlib import Path

from shinka.core import run_shinka_eval

from medframeqa_runtime import (
    deterministic_letter_fallback,
    extract_option_letter,
    generate_guided_choice_with_retries,
    get_protocol_meta,
    get_protocol_score_key,
    get_protocol_subset,
    load_medframeqa_dataset,
    load_mutated_module,
    load_split_manifest,
    make_openai_client,
    make_protocol_metrics,
    parse_generation_index,
    parse_generation_set,
    safe_run_experiment,
    write_json,
)


TASK_NAME = "advanced_vqa_task_reasoning"
VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "256"))
POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "10,20,30,40,49"))
API_TIMEOUT = float(os.environ.get("MEDFRAMEQA_API_TIMEOUT", "30"))


def evaluate_subset(module, subset, split_name, client, call_stats):
    format_fn = module.format_vqa_payload
    correct = 0
    total = len(subset)
    missing_pred_count = 0

    for sample in subset:
        valid_letters = list(string.ascii_uppercase[: len(sample["options"])])
        gt = sample["correct_answer"].strip().upper()
        raw = generate_guided_choice_with_retries(
            client,
            lambda sample=sample: format_fn(sample),
            valid_letters,
            VLM_MODEL,
            f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')}",
            call_stats=call_stats,
        )
        pred = extract_option_letter(raw, valid_letters)
        if not pred:
            missing_pred_count += 1
            pred = deterministic_letter_fallback(valid_letters, "alphabetical")
        print(
            f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')} "
            f"raw={raw!r} pred={pred!r} gt={gt!r}",
            file=sys.stderr,
        )
        if pred == gt:
            correct += 1

    metrics = make_protocol_metrics(split_name, correct, total)
    metrics["missing_pred_count"] = missing_pred_count
    return metrics


def format_feedback_score(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def build_text_feedback(metrics):
    lines = [
        f"search_score={format_feedback_score(metrics.get('search_score'))}",
        f"pool_score={format_feedback_score(metrics.get('evolution_pool_score'))}",
        f"invalid_generation={int(metrics.get('invalid_generation', 0))}",
        f"error_type={metrics.get('error_type') or 'none'}",
    ]
    if metrics.get("invalid_generation"):
        lines.append(
            "The candidate failed during evaluation. Keep the program schema valid and preserve the prompt config keys."
        )
        if metrics.get("error_message"):
            lines.append(f"error_message={metrics['error_message']}")
        return "\n".join(lines)

    lines.append(
        "Use all images in order before deciding, and reject options that conflict with modality, anatomy, laterality, or temporal sequence."
    )
    if metrics.get("missing_pred_count", 0) > 0:
        lines.append(
            "Some answers were empty or malformed. Keep the answer prefix tight and force a single capital-letter output."
        )
    search_score = metrics.get("search_score")
    pool_score = metrics.get("evolution_pool_score")
    if pool_score is not None and search_score is not None and pool_score + 1e-9 < search_score:
        lines.append(
            "The search-mini score is higher than the pool score. Make the rule less brittle and more consistent across the full evolution pool."
        )
    lines.append(
        "Prefer options supported by the whole sequence, not a single frame-level match."
    )
    return "\n".join(lines)


def _run(program_path, results_dir=None, protocol_mode=None, search_mini_size=None):
    protocol_mode = protocol_mode or DEFAULT_PROTOCOL_MODE
    search_mini_size = int(search_mini_size or DEFAULT_SEARCH_MINI_SIZE)
    module = load_mutated_module(program_path)
    dataset = load_medframeqa_dataset(include_images=True)
    manifest = load_split_manifest()
    subset, selected_ids = get_protocol_subset(dataset, manifest, protocol_mode, search_mini_size)
    client = make_openai_client(VLM_URL, timeout=API_TIMEOUT)
    call_stats = {"vlm_call_count": 0}

    metrics = evaluate_subset(module, subset, protocol_mode, client, call_stats)
    metrics.update(get_protocol_meta(manifest, protocol_mode, selected_ids, search_mini_size))

    generation_index = parse_generation_index(results_dir or program_path)
    if protocol_mode == "search_mini" and generation_index in POOL_REEVAL_GENS:
        pool_subset, _ = get_protocol_subset(dataset, manifest, "evolution_pool", search_mini_size)
        metrics.update(evaluate_subset(module, pool_subset, "evolution_pool", client, call_stats))

    metrics["vlm_call_count"] = call_stats["vlm_call_count"]
    metrics["combined_score"] = metrics[get_protocol_score_key(protocol_mode)]
    return metrics


def run_experiment(program_path, results_dir=None, protocol_mode=None, search_mini_size=None, **kwargs):
    chosen_mode = protocol_mode or DEFAULT_PROTOCOL_MODE
    chosen_search_mini_size = int(search_mini_size or DEFAULT_SEARCH_MINI_SIZE)
    metrics = safe_run_experiment(
        lambda: _run(
            program_path=program_path,
            results_dir=results_dir,
            protocol_mode=chosen_mode,
            search_mini_size=chosen_search_mini_size,
        ),
        chosen_mode,
        chosen_search_mini_size,
        task_name=TASK_NAME,
        results_dir=results_dir,
    )
    metrics["text_feedback"] = build_text_feedback(metrics)
    return metrics


def aggregate_fn(results):
    metrics = dict(results[0])
    metrics["text_feedback"] = build_text_feedback(metrics)
    return metrics


def main(program_path, results_dir):
    run_shinka_eval(
        program_path=os.path.abspath(__file__),
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=1,
        get_experiment_kwargs=lambda idx: {
            "program_path": program_path,
            "results_dir": results_dir,
            "protocol_mode": DEFAULT_PROTOCOL_MODE,
            "search_mini_size": DEFAULT_SEARCH_MINI_SIZE,
        },
        aggregate_metrics_fn=aggregate_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--single_run", action="store_true")
    parser.add_argument("--protocol_mode", default=None)
    parser.add_argument("--search_mini_size", type=int, default=None)
    args = parser.parse_args()

    if args.single_run:
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        metrics = run_experiment(
            program_path=args.program_path,
            results_dir=args.results_dir,
            protocol_mode=args.protocol_mode,
            search_mini_size=args.search_mini_size,
        )
        write_json(Path(args.results_dir) / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        main(args.program_path, args.results_dir)
