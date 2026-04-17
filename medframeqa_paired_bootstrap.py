#!/home1/xl693/anaconda3/envs/vlm/bin/python
"""MedFrameQA 配对 bootstrap 显著性分析脚本。

输入：
- 各方法在同一 split、同一 question 集上的逐题预测结果；

输出：
- accuracy 差值
- paired bootstrap 95% CI
- 胜率与近似双侧 p 值

默认比较：
- order_vote vs fixed
- order_vote vs order_rerank
- 如果 order_vote_plus 有 post-hoc 输出，再额外比较 order_vote_plus vs order_vote
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/gluon4/xl693/evolve")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medframeqa_posthoc_eval import find_latest_complete_run, load_json
from medframeqa_runtime import write_csv_rows, write_json


DEFAULT_OUTPUT_DIR = ROOT / "paper_analysis_output"


def load_prediction_rows(method: str, results_root: Path, eval_name: str):
    run_dir = find_latest_complete_run(method, results_root)
    prediction_path = run_dir / "paper_eval" / eval_name / "predictions.json"
    if not prediction_path.exists():
        raise FileNotFoundError(f"缺少 predictions: {prediction_path}")
    rows = load_json(prediction_path)
    rows = sorted(rows, key=lambda row: row["question_id"])
    return run_dir, rows


def paired_accuracy_arrays(rows_a, rows_b):
    by_qid_a = {row["question_id"]: row for row in rows_a}
    by_qid_b = {row["question_id"]: row for row in rows_b}
    qids_a = set(by_qid_a)
    qids_b = set(by_qid_b)
    if qids_a != qids_b:
        raise RuntimeError("paired bootstrap 需要两边 question_id 完全一致")

    qids = sorted(qids_a)
    arr_a = np.array([by_qid_a[qid]["correct"] for qid in qids], dtype=np.float64)
    arr_b = np.array([by_qid_b[qid]["correct"] for qid in qids], dtype=np.float64)
    return qids, arr_a, arr_b


def bootstrap_diff(arr_a, arr_b, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(arr_a)
    if n == 0:
        raise ValueError("empty arrays")

    diff = float(arr_a.mean() - arr_b.mean())
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    sampled_diff = arr_a[indices].mean(axis=1) - arr_b[indices].mean(axis=1)
    ci_low = float(np.percentile(sampled_diff, 2.5))
    ci_high = float(np.percentile(sampled_diff, 97.5))
    prob_le_zero = float((sampled_diff <= 0.0).mean())
    prob_ge_zero = float((sampled_diff >= 0.0).mean())
    p_two_sided = float(min(1.0, 2.0 * min(prob_le_zero, prob_ge_zero)))
    return {
        "accuracy_a": float(arr_a.mean()),
        "accuracy_b": float(arr_b.mean()),
        "diff_a_minus_b": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate_a_gt_b": float((sampled_diff > 0.0).mean()),
        "win_rate_a_ge_b": float((sampled_diff >= 0.0).mean()),
        "p_two_sided": p_two_sided,
        "num_questions": int(n),
        "num_bootstrap": int(n_bootstrap),
    }


def main():
    parser = argparse.ArgumentParser(description="对 MedFrameQA 最终测试结果做 paired bootstrap。")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--eval-name", default="posthoc_selected_final_test")
    parser.add_argument("--pairs", nargs="+", default=["order_vote,fixed", "order_vote,order_rerank"])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260329)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_rows = []
    for pair in args.pairs:
        method_a, method_b = pair.split(",", 1)
        run_dir_a, rows_a = load_prediction_rows(method_a, args.results_root, args.eval_name)
        run_dir_b, rows_b = load_prediction_rows(method_b, args.results_root, args.eval_name)
        _, arr_a, arr_b = paired_accuracy_arrays(rows_a, rows_b)
        result = bootstrap_diff(arr_a, arr_b, args.bootstrap_samples, args.seed)
        output_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "run_dir_a": str(run_dir_a),
                "run_dir_b": str(run_dir_b),
                "eval_name": args.eval_name,
                **result,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bootstrap_{args.eval_name}"
    write_json(args.output_dir / f"{stem}.json", output_rows)
    write_csv_rows(args.output_dir / f"{stem}.csv", output_rows)

    for row in output_rows:
        print(
            f"{row['method_a']} vs {row['method_b']} | "
            f"diff={row['diff_a_minus_b']:.6f} | "
            f"95% CI=({row['ci_low']:.6f}, {row['ci_high']:.6f}) | "
            f"p={row['p_two_sided']:.6f}"
        )


if __name__ == "__main__":
    main()
