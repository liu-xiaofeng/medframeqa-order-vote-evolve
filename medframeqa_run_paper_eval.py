#!/home1/xl693/anaconda3/envs/vlm/bin/python
"""为单个 MedFrameQA run 生成论文阶段所需的 paper_eval 产物。

这个脚本把 notebook 里“postrun 论文汇总”的逻辑收口成可复用命令行入口。

输入：
- 某条方法线的 `results_dir`
- 对应的方法名，用来恢复 task 目录、search_mini、大池复评代数与阈值

输出：
- `results_dir/paper_eval/generation_records.json|csv`
- `results_dir/paper_eval/milestone_pool_eval.json|csv`
- `results_dir/paper_eval/top3_holdout_eval.json|csv`
- `results_dir/paper_eval/paper_summary.json`

失败处理：
- 如果 run 目录里缺少 generation 结果，直接报错而不是静默跳过；
- 如果 direct eval 失败，保留异常向上抛出，让上层 pipeline 明确失败；
- 这个脚本只写 `paper_eval` 目录，不会修改任何 generation 代码。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/gluon4/xl693/evolve")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medframeqa_runtime import (
    collect_generation_records,
    parse_generation_set,
    select_best_so_far,
    select_top_k_records,
    write_csv_rows,
    write_json,
)
from run_medframeqa_repeats import METHOD_SPECS, build_env


INVALID_RATE_THRESHOLDS = {
    "fixed": 0.05,
    "reasoning": 0.05,
    "order_vote": 0.05,
    "order_vote_plus": 0.05,
    "order_rerank": 0.10,
}


def normalize_pool_reeval_gens(raw_value) -> list[int]:
    """把 pool_reeval_gens 统一成排序后的 int 列表。"""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return sorted(parse_generation_set(raw_value))
    if isinstance(raw_value, (list, tuple, set)):
        normalized = []
        for item in raw_value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(set(normalized))
    return []


def run_direct_eval(
    method: str,
    results_dir: Path,
    paper_dir: Path,
    program_path: str,
    protocol_mode: str,
    out_name: str,
) -> dict:
    spec = METHOD_SPECS[method]
    out_dir = paper_dir / out_name
    env = build_env(spec)
    cmd = [
        sys.executable,
        str(ROOT / spec["task_dir"] / "evaluate.py"),
        "--program_path",
        program_path,
        "--results_dir",
        str(out_dir),
        "--single_run",
        "--protocol_mode",
        protocol_mode,
        "--search_mini_size",
        str(spec["search_mini_size"]),
    ]
    subprocess.run(cmd, env=env, cwd=str(ROOT), check=True)
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))


def build_paper_summary(method: str, results_dir: Path) -> dict:
    spec = METHOD_SPECS[method]
    invalid_rate_threshold = INVALID_RATE_THRESHOLDS[method]
    paper_dir = results_dir / "paper_eval"
    paper_dir.mkdir(parents=True, exist_ok=True)
    pool_reeval_gens = normalize_pool_reeval_gens(spec.get("pool_reeval_gens"))

    records = collect_generation_records(results_dir)
    if not records:
        raise RuntimeError(f"{results_dir} 没有 generation 结果，无法生成 paper summary")

    write_csv_rows(paper_dir / "generation_records.csv", records)
    write_json(paper_dir / "generation_records.json", records)

    valid_records = [row for row in records if not row.get("invalid_generation")]
    invalid_records = [row for row in records if row.get("invalid_generation")]
    valid_rate = (len(valid_records) / len(records)) if records else 0.0
    invalid_rate = (len(invalid_records) / len(records)) if records else 0.0
    best_record_list = select_top_k_records(records, top_k=1)
    best_record = best_record_list[0] if best_record_list else None
    last_record = records[-1] if records else None

    milestone_rows = []
    max_generation = records[-1]["generation"] if records else None
    milestone_generations = sorted(
        set(pool_reeval_gens + ([max_generation] if max_generation is not None else []))
    )
    for milestone in milestone_generations:
        record = select_best_so_far(records, milestone)
        if record is None:
            continue
        metrics = run_direct_eval(
            method=method,
            results_dir=results_dir,
            paper_dir=paper_dir,
            program_path=record["program_path"],
            protocol_mode="evolution_pool",
            out_name=f"milestone_gen_{milestone}_pool_eval",
        )
        milestone_rows.append(
            {
                "milestone_generation": milestone,
                "selected_generation": record["generation"],
                "selected_program_path": record["program_path"],
                **metrics,
            }
        )
    write_csv_rows(paper_dir / "milestone_pool_eval.csv", milestone_rows)
    write_json(paper_dir / "milestone_pool_eval.json", milestone_rows)

    top_records = select_top_k_records(records, top_k=3)
    if not top_records:
        raise RuntimeError(f"{results_dir} 没有任何有效 generation，无法继续 holdout/final test 评估")
    holdout_rows = []
    for rank, record in enumerate(top_records, 1):
        metrics = run_direct_eval(
            method=method,
            results_dir=results_dir,
            paper_dir=paper_dir,
            program_path=record["program_path"],
            protocol_mode="selection_holdout",
            out_name=f"top{rank}_holdout_eval",
        )
        holdout_rows.append(
            {
                "rank": rank,
                "generation": record["generation"],
                "program_path": record["program_path"],
                **metrics,
            }
        )
    write_csv_rows(paper_dir / "top3_holdout_eval.csv", holdout_rows)
    write_json(paper_dir / "top3_holdout_eval.json", holdout_rows)

    best_holdout = None
    if holdout_rows:
        best_holdout = sorted(
            holdout_rows,
            key=lambda row: (
                -row.get("holdout_score", 0.0),
                row.get("generation", 10**9),
            ),
        )[0]

    summary = {
        "task_name": spec["task_name"],
        "results_dir": str(results_dir),
        "generation_count": len(records),
        "valid_generation_count": len(valid_records),
        "invalid_generation_count": len(invalid_records),
        "valid_generation_rate": round(valid_rate, 4),
        "invalid_generation_rate": round(invalid_rate, 4),
        "invalid_rate_threshold": invalid_rate_threshold,
        "paper_ready_candidate": invalid_rate <= invalid_rate_threshold,
        "best_generation": best_record["generation"] if best_record else None,
        "best_combined_score": best_record.get("combined_score") if best_record else None,
        "last_generation": last_record["generation"] if last_record else None,
        "last_combined_score": last_record.get("combined_score") if last_record else None,
        "top3_generations": [row["generation"] for row in top_records],
        "milestone_pool": milestone_rows,
        "top3_holdout": holdout_rows,
    }
    if best_holdout is not None:
        final_metrics = run_direct_eval(
            method=method,
            results_dir=results_dir,
            paper_dir=paper_dir,
            program_path=best_holdout["program_path"],
            protocol_mode="independent_final_test",
            out_name="final_test_eval",
        )
        summary["selected_generation"] = best_holdout["generation"]
        summary["selected_program_path"] = best_holdout["program_path"]
        summary["final_test"] = final_metrics

    write_json(paper_dir / "paper_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="为单个 MedFrameQA run 生成 paper_eval 目录。")
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHOD_SPECS),
        help="方法名，用于恢复 task 目录和运行配置。",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_paper_summary(args.method, args.results_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
