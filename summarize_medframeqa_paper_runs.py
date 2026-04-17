#!/home1/xl693/anaconda3/envs/vlm/bin/python
"""MedFrameQA 论文结果汇总脚本。

这个脚本只做读结果与汇总，不会修改任何已有实验目录。
输入是各方法 run 目录下已经生成的 `paper_eval` 产物，主要读取：
- `paper_summary.json`
- `generation_records.json`
- `top3_holdout_eval.json`

输出是统一的 JSON / CSV 表，方便后续写论文主表、消融表和复杂度表。
如果某个方法缺少结果文件，脚本会直接报错，而不是静默跳过。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = Path("/gluon4/xl693/evolve/results")
DEFAULT_OUTPUT_DIR = Path("/gluon4/xl693/evolve/paper_analysis_output")
DEFAULT_BUDGET_ABLATION_ROOT = Path("/gluon4/xl693/evolve/results_budget_ablation")
METHOD_RUN_PATTERNS = {
    "fixed": {
        "prefix": "advanced_vqa_task_fixed_",
        "exclude": (),
    },
    "reasoning": {
        "prefix": "advanced_vqa_task_reasoning_",
        "exclude": (),
    },
    "order_vote": {
        "prefix": "advanced_vqa_task_order_vote_",
        "exclude": ("advanced_vqa_task_order_vote_plus_",),
    },
    "order_rerank": {
        "prefix": "advanced_vqa_task_order_rerank_",
        "exclude": (),
    },
    "order_vote_plus": {
        "prefix": "advanced_vqa_task_order_vote_plus_",
        "exclude": (),
    },
}
METHOD_ROLES = {
    "fixed": "stable_baseline",
    "reasoning": "negative_prompt_baseline",
    "order_vote": "main_candidate",
    "order_rerank": "strong_ablation",
    "order_vote_plus": "appendix_exploration",
}
MAIN_TABLE_METHODS = ["fixed", "order_vote", "order_rerank"]
BUDGET_ABLATION_PATTERN = "advanced_vqa_task_order_vote_budget100_"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _match_method_run(run_dir: Path, method: str) -> bool:
    pattern = METHOD_RUN_PATTERNS[method]
    if not run_dir.name.startswith(pattern["prefix"]):
        return False
    return not any(excluded in run_dir.name for excluded in pattern["exclude"])


def _match_run_filter(run_dir: Path, *, repeat_only: bool = False) -> bool:
    if repeat_only and "_repeat" not in run_dir.name:
        return False
    return True


def find_complete_runs(
    results_root: Path,
    method: str,
    min_generations: int = 50,
    *,
    repeat_only: bool = False,
) -> list[Path]:
    candidates = sorted(p for p in results_root.iterdir() if p.is_dir() and _match_method_run(p, method))
    completed: list[Path] = []
    for run_dir in candidates:
        if not _match_run_filter(run_dir, repeat_only=repeat_only):
            continue
        paper_eval = run_dir / "paper_eval"
        summary_path = paper_eval / "paper_summary.json"
        records_path = paper_eval / "generation_records.json"
        if not summary_path.exists() or not records_path.exists():
            continue
        paper_summary = load_json(summary_path)
        if int(paper_summary.get("generation_count") or 0) < min_generations:
            continue
        completed.append(run_dir)
    return completed


def find_latest_complete_run(
    results_root: Path,
    method: str,
    min_generations: int = 50,
    *,
    repeat_only: bool = False,
) -> Path:
    completed = find_complete_runs(
        results_root,
        method,
        min_generations=min_generations,
        repeat_only=repeat_only,
    )
    if not completed:
        prefix = METHOD_RUN_PATTERNS[method]["prefix"]
        raise FileNotFoundError(f"no completed runs found for method={method!r} prefix={prefix!r} under {results_root}")
    return completed[-1]


def load_method_summary(run_dir: Path) -> dict[str, Any]:
    paper_eval = run_dir / "paper_eval"
    paper_summary = load_json(paper_eval / "paper_summary.json")
    generation_records = load_json(paper_eval / "generation_records.json")
    top3_holdout = load_json(paper_eval / "top3_holdout_eval.json")
    final_test = paper_summary.get("final_test") or {}

    valid_records = [row for row in generation_records if not row.get("invalid_generation")]
    best_holdout = max(top3_holdout, key=lambda row: row.get("holdout_score", float("-inf")))

    avg_generation_wall_time = (
        sum(float(row.get("wall_time_sec", 0.0) or 0.0) for row in valid_records) / len(valid_records)
        if valid_records
        else None
    )
    avg_generation_vlm_calls = (
        sum(int(row.get("vlm_call_count", 0) or 0) for row in valid_records) / len(valid_records)
        if valid_records
        else None
    )

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "valid_generation_rate": paper_summary.get("valid_generation_rate"),
        "invalid_generation_rate": paper_summary.get("invalid_generation_rate"),
        "best_generation": paper_summary.get("best_generation"),
        "best_combined_score": paper_summary.get("best_combined_score"),
        "last_generation": paper_summary.get("last_generation"),
        "last_combined_score": paper_summary.get("last_combined_score"),
        "selected_generation": paper_summary.get("selected_generation"),
        "selected_holdout_score": best_holdout.get("holdout_score"),
        "selected_holdout_correct": best_holdout.get("holdout_correct"),
        "selected_holdout_size": best_holdout.get("holdout_size"),
        "selected_holdout_wall_time_sec": best_holdout.get("wall_time_sec"),
        "selected_holdout_vlm_call_count": best_holdout.get("vlm_call_count"),
        "final_test_score": final_test.get("final_test_score"),
        "final_test_correct": final_test.get("final_test_correct"),
        "final_test_size": final_test.get("final_test_size"),
        "final_test_wall_time_sec": final_test.get("wall_time_sec"),
        "final_test_vlm_call_count": final_test.get("vlm_call_count"),
        "paper_ready_candidate": paper_summary.get("paper_ready_candidate"),
        "avg_generation_wall_time_sec": avg_generation_wall_time,
        "avg_generation_vlm_call_count": avg_generation_vlm_calls,
    }


def aggregate_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    aggregate_rows = []
    for method, method_rows in grouped.items():
        def _mean(key):
            values = [row[key] for row in method_rows if row.get(key) is not None]
            return (sum(values) / len(values)) if values else None

        def _std(key):
            values = [row[key] for row in method_rows if row.get(key) is not None]
            if len(values) <= 1:
                return 0.0 if values else None
            mean_value = sum(values) / len(values)
            variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
            return variance ** 0.5

        aggregate_rows.append(
            {
                "method": method,
                "run_count": len(method_rows),
                "final_test_score_mean": _mean("final_test_score"),
                "final_test_score_std": _std("final_test_score"),
                "selected_holdout_score_mean": _mean("selected_holdout_score"),
                "selected_holdout_score_std": _std("selected_holdout_score"),
                "invalid_generation_rate_mean": _mean("invalid_generation_rate"),
                "invalid_generation_rate_std": _std("invalid_generation_rate"),
            }
        )
    return sorted(aggregate_rows, key=lambda row: row["method"])


def find_latest_budget_ablation_run(results_root: Path) -> Path | None:
    candidates = sorted(
        p
        for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith(BUDGET_ABLATION_PATTERN) and (p / "paper_eval" / "paper_summary.json").exists()
    )
    return candidates[-1] if candidates else None


def load_budget_sensitivity_summary(run_dir: Path, aggregate_by_method: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paper_summary = load_json(run_dir / "paper_eval" / "paper_summary.json")
    top3_holdout = paper_summary.get("top3_holdout") or []
    best_holdout = None
    if top3_holdout:
        best_holdout = sorted(
            top3_holdout,
            key=lambda row: (
                -(row.get("holdout_score") or -1.0),
                row.get("generation", 10**9),
            ),
        )[0]

    order_vote_row = aggregate_by_method.get("order_vote") or {}
    fifty_final_mean = order_vote_row.get("final_test_score_mean")
    fifty_holdout_mean = order_vote_row.get("selected_holdout_score_mean")
    final_test = paper_summary.get("final_test") or {}
    budget_final = final_test.get("final_test_score")
    budget_holdout = best_holdout.get("holdout_score") if best_holdout else None
    delta_final = (budget_final - fifty_final_mean) if (budget_final is not None and fifty_final_mean is not None) else None
    delta_holdout = (
        budget_holdout - fifty_holdout_mean
        if (budget_holdout is not None and fifty_holdout_mean is not None)
        else None
    )

    interpretation = "budget_ablation_missing"
    if delta_final is not None:
        if delta_final < 0.005:
            interpretation = "keep_50_generations"
        elif delta_final < 0.01:
            interpretation = "appendix_note_small_gain"
        else:
            interpretation = "needs_additional_100gen_repeats"

    return {
        "available": True,
        "ablation_run": str(run_dir),
        "ablation_run_name": run_dir.name,
        "final_test_score": budget_final,
        "selected_holdout_score": budget_holdout,
        "invalid_generation_rate": paper_summary.get("invalid_generation_rate"),
        "best_generation": paper_summary.get("best_generation"),
        "best_combined_score": paper_summary.get("best_combined_score"),
        "last_generation": paper_summary.get("last_generation"),
        "last_combined_score": paper_summary.get("last_combined_score"),
        "fifty_gen_repeat_only_mean_final_test_score": fifty_final_mean,
        "fifty_gen_repeat_only_mean_holdout_score": fifty_holdout_mean,
        "delta_vs_50gen_mean_final_test": delta_final,
        "delta_vs_50gen_mean_holdout": delta_holdout,
        "keep_main_budget_at_50": bool(delta_final is None or delta_final < 0.005),
        "appendix_budget_sensitive": bool(delta_final is not None and delta_final >= 0.01),
        "interpretation": interpretation,
    }


def build_decision_summary(
    latest_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    budget_sensitivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest_by_method = {row["method"]: row for row in latest_rows}
    aggregate_by_method = {row["method"]: row for row in aggregate_rows}
    ranked_latest = sorted(
        latest_rows,
        key=lambda row: (
            row.get("final_test_score") is None,
            -(row.get("final_test_score") or -1.0),
            -(row.get("selected_holdout_score") or -1.0),
        ),
    )
    seed_status = {
        method: {
            "target_run_count": 3 if method in MAIN_TABLE_METHODS else 1,
            "current_run_count": int((aggregate_by_method.get(method) or {}).get("run_count") or 0),
        }
        for method in METHOD_ROLES
    }
    for payload in seed_status.values():
        payload["seed_complete"] = payload["current_run_count"] >= payload["target_run_count"]

    fixed_score = (latest_by_method.get("fixed") or {}).get("final_test_score")
    reasoning_score = (latest_by_method.get("reasoning") or {}).get("final_test_score")
    order_vote_score = (latest_by_method.get("order_vote") or {}).get("final_test_score")
    order_rerank_score = (latest_by_method.get("order_rerank") or {}).get("final_test_score")
    order_vote_plus_score = (latest_by_method.get("order_vote_plus") or {}).get("final_test_score")

    return {
        "latest_final_test_ranking": [
            {
                "rank": index + 1,
                "method": row["method"],
                "method_role": METHOD_ROLES.get(row["method"], "unassigned"),
                "final_test_score": row.get("final_test_score"),
                "selected_holdout_score": row.get("selected_holdout_score"),
            }
            for index, row in enumerate(ranked_latest)
        ],
        "current_main_candidate": ranked_latest[0]["method"] if ranked_latest else None,
        "main_table_methods": MAIN_TABLE_METHODS,
        "seed_status": seed_status,
        "core_methods_seed_complete": all(seed_status[method]["seed_complete"] for method in MAIN_TABLE_METHODS),
        "reasoning_not_better_than_fixed": (
            reasoning_score is not None and fixed_score is not None and reasoning_score <= fixed_score
        ),
        "order_vote_plus_below_order_rerank": (
            order_vote_plus_score is not None
            and order_rerank_score is not None
            and order_vote_plus_score < order_rerank_score
        ),
        "order_vote_is_current_best": (
            order_vote_score is not None
            and ranked_latest
            and ranked_latest[0]["method"] == "order_vote"
        ),
        "budget_sensitivity": budget_sensitivity or {"available": False},
        "keep_main_budget_at_50": bool((budget_sensitivity or {}).get("keep_main_budget_at_50", True)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_score(value):
    return "None" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 MedFrameQA 论文结果。")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budget-ablation-root", type=Path, default=DEFAULT_BUDGET_ABLATION_ROOT)
    parser.add_argument("--min-generations", type=int, default=50)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fixed", "reasoning", "order_vote", "order_rerank", "order_vote_plus"],
        choices=sorted(METHOD_RUN_PATTERNS),
        help="要汇总的方法列表。若包含 order_vote_plus，则会一并扫描。",
    )
    parser.add_argument(
        "--repeat-only",
        action="store_true",
        help="只汇总 run_name 包含 _repeat 的同协议高预算独立重复运行。",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for method in args.methods:
        complete_runs = find_complete_runs(
            args.results_root,
            method,
            min_generations=args.min_generations,
            repeat_only=args.repeat_only,
        )
        for run_dir in complete_runs:
            row = {"method": method, **load_method_summary(run_dir)}
            all_rows.append(row)
        latest_rows.append({"method": method, **load_method_summary(complete_runs[-1])})

    latest_json_path = args.output_dir / "method_comparison_latest.json"
    latest_csv_path = args.output_dir / "method_comparison_latest.csv"
    latest_json_path.write_text(json.dumps(latest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(latest_csv_path, latest_rows)

    all_json_path = args.output_dir / "method_comparison_all_runs.json"
    all_csv_path = args.output_dir / "method_comparison_all_runs.csv"
    all_json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(all_csv_path, all_rows)

    aggregate_rows = aggregate_method_rows(all_rows)
    aggregate_json_path = args.output_dir / "method_seed_aggregate.json"
    aggregate_csv_path = args.output_dir / "method_seed_aggregate.csv"
    aggregate_json_path.write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(aggregate_csv_path, aggregate_rows)

    main_table_rows = [row for row in aggregate_rows if row["method"] in MAIN_TABLE_METHODS]
    main_table_json_path = args.output_dir / "main_table_repeat_only.json"
    main_table_csv_path = args.output_dir / "main_table_repeat_only.csv"
    main_table_json_path.write_text(json.dumps(main_table_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(main_table_csv_path, main_table_rows)

    aggregate_by_method = {row["method"]: row for row in aggregate_rows}
    budget_sensitivity = {"available": False}
    budget_run_dir = find_latest_budget_ablation_run(args.budget_ablation_root)
    if budget_run_dir is not None:
        budget_sensitivity = load_budget_sensitivity_summary(budget_run_dir, aggregate_by_method)

    budget_summary_path = args.output_dir / "budget_sensitivity_summary.json"
    budget_summary_path.write_text(json.dumps(budget_sensitivity, ensure_ascii=False, indent=2), encoding="utf-8")

    budget_csv_path = args.output_dir / "budget_sensitivity_summary.csv"
    write_csv(budget_csv_path, [budget_sensitivity])

    decision_summary = build_decision_summary(latest_rows, aggregate_rows, budget_sensitivity=budget_sensitivity)
    decision_summary["analysis_filter"] = {
        "repeat_only": args.repeat_only,
        "min_generations": args.min_generations,
        "methods": args.methods,
    }
    decision_summary_path = args.output_dir / "paper_decision_summary.json"
    decision_summary_path.write_text(json.dumps(decision_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {latest_json_path}")
    print(f"Wrote {latest_csv_path}")
    print(f"Wrote {all_json_path}")
    print(f"Wrote {all_csv_path}")
    print(f"Wrote {aggregate_json_path}")
    print(f"Wrote {aggregate_csv_path}")
    print(f"Wrote {main_table_json_path}")
    print(f"Wrote {main_table_csv_path}")
    print(f"Wrote {budget_summary_path}")
    print(f"Wrote {budget_csv_path}")
    print(f"Wrote {decision_summary_path}")
    print()
    for row in sorted(
        latest_rows,
        key=lambda item: (
            item["final_test_score"] is None,
            -(item["final_test_score"] or -1.0),
            -(item["selected_holdout_score"] or -1.0),
        ),
    ):
        print(
            f"{row['method']:>12} | final={fmt_score(row['final_test_score'])} | "
            f"holdout={fmt_score(row['selected_holdout_score'])} | "
            f"best_search={fmt_score(row['best_combined_score'])} | invalid={row['invalid_generation_rate']:.3f} | "
            f"run={row['run_name']}"
        )


if __name__ == "__main__":
    main()
