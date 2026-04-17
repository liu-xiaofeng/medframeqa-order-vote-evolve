#!/home1/xl693/anaconda3/envs/vlm/bin/python
"""MedFrameQA 论文级自动重跑与分析 pipeline。

设计目标：
- 把 preflight、重跑、paper_eval、post-hoc final validation、paper analysis
  串成一个可恢复的单入口流程；
- 主文 50-generation 结果和 100-generation budget ablation 分开存放，避免互相污染；
- 每个阶段都把状态写进统一的 `pipeline_report.json`，便于中断后恢复。

这个脚本会调用已有脚本：
- run_medframeqa_repeats.py
- medframeqa_run_paper_eval.py
- medframeqa_posthoc_eval.py
- medframeqa_paired_bootstrap.py
- summarize_medframeqa_paper_runs.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/gluon4/xl693/evolve")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medframeqa_runtime import get_image_columns, load_medframeqa_dataset, load_split_manifest
from run_medframeqa_repeats import (
    METHOD_SPECS,
    PROFILE_SPECS,
    RIGOROUS_PROFILE_NAME,
    ORDER_VOTE_BUDGET100_PROFILE,
    build_env,
    build_shinka_command,
    ensure_shinka_bin,
    get_effective_spec,
)
from summarize_medframeqa_paper_runs import METHOD_RUN_PATTERNS


RESULTS_ROOT = ROOT / "results"
OUTPUT_DIR = ROOT / "paper_analysis_output"
PIPELINE_REPORT_PATH = OUTPUT_DIR / "pipeline_report.json"
PAPER_EVAL_SCRIPT = ROOT / "medframeqa_run_paper_eval.py"
POSTHOC_SCRIPT = ROOT / "medframeqa_posthoc_eval.py"
BOOTSTRAP_SCRIPT = ROOT / "medframeqa_paired_bootstrap.py"
SUMMARY_SCRIPT = ROOT / "summarize_medframeqa_paper_runs.py"
REPEATS_SCRIPT = ROOT / "run_medframeqa_repeats.py"
BUDGET_ABLATION_ROOT = ROOT / "results_budget_ablation"
DEFAULT_METHODS = ["fixed", "reasoning", "order_vote", "order_rerank", "order_vote_plus"]
CORE_METHODS = ["fixed", "order_vote", "order_rerank"]
STAGE_ORDER = [
    "preflight",
    "main5_once",
    "recover_main5_paper_eval",
    "posthoc_validate_main5",
    "paper_analyze_main5",
    "core_repeats",
    "posthoc_validate_all",
    "paper_analyze_all",
    "optional_budget_ablation",
]


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def init_report(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "started_at": now_ts(),
        "cwd": os.getcwd(),
        "args": {
            "stage": args.stage,
            "resume_from": args.resume_from,
            "skip_budget_ablation": args.skip_budget_ablation,
            "dry_run": args.dry_run,
        },
        "profiles": {
            "main_pipeline": RIGOROUS_PROFILE_NAME,
            "budget_ablation": ORDER_VOTE_BUDGET100_PROFILE,
            "definitions": PROFILE_SPECS,
        },
        "stage_order": STAGE_ORDER,
        "stages": {},
        "new_run_dirs": {},
        "core_run_targets": {},
        "order_vote_plus_gate": {},
        "budget_ablation": {},
    }
    write_json(PIPELINE_REPORT_PATH, report)
    return report


def save_report(report: dict[str, Any]) -> None:
    write_json(PIPELINE_REPORT_PATH, report)


def begin_stage(report: dict[str, Any], stage_name: str) -> None:
    print(f"\n=== STAGE START: {stage_name} ===")
    report["stages"].setdefault(stage_name, {})
    report["stages"][stage_name].update(
        {
            "status": "running",
            "started_at": now_ts(),
        }
    )
    save_report(report)


def finish_stage(report: dict[str, Any], stage_name: str, details: dict[str, Any] | None = None) -> None:
    print(f"=== STAGE DONE: {stage_name} ===")
    if details:
        try:
            print(json.dumps(details, indent=2, ensure_ascii=False)[:4000])
        except Exception:
            print(details)
    report["stages"].setdefault(stage_name, {})
    report["stages"][stage_name].update(
        {
            "status": "completed",
            "ended_at": now_ts(),
            "details": details or {},
        }
    )
    save_report(report)


def fail_stage(report: dict[str, Any], stage_name: str, exc: Exception) -> None:
    print(f"=== STAGE FAILED: {stage_name} ===")
    print(f"{type(exc).__name__}: {exc}")
    report["stages"].setdefault(stage_name, {})
    report["stages"][stage_name].update(
        {
            "status": "failed",
            "ended_at": now_ts(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )
    save_report(report)


def stage_subset(args: argparse.Namespace) -> list[str]:
    if args.stage != "full":
        return [args.stage]
    if args.resume_from:
        if args.resume_from not in STAGE_ORDER:
            raise ValueError(f"unknown resume stage: {args.resume_from}")
        start = STAGE_ORDER.index(args.resume_from)
        stages = STAGE_ORDER[start:]
    else:
        stages = list(STAGE_ORDER)
    if args.skip_budget_ablation and "optional_budget_ablation" in stages:
        stages = [stage for stage in stages if stage != "optional_budget_ablation"]
    return stages


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("COMMAND:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=True)


def list_run_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.resolve()) for path in root.iterdir() if path.is_dir()}


def match_run_method(run_dir_name: str) -> str:
    for method, pattern in METHOD_RUN_PATTERNS.items():
        if not run_dir_name.startswith(pattern["prefix"]):
            continue
        if any(excluded in run_dir_name for excluded in pattern["exclude"]):
            continue
        return method
    raise ValueError(f"无法从 run 目录名识别方法: {run_dir_name}")


def find_latest_run_for_method(
    method: str,
    *,
    require_repeat01: bool = False,
    results_root: Path = RESULTS_ROOT,
) -> Path | None:
    pattern = METHOD_RUN_PATTERNS[method]
    candidates = []
    if not results_root.exists():
        return None
    for run_dir in results_root.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith(pattern["prefix"]):
            continue
        if any(excluded in run_dir.name for excluded in pattern["exclude"]):
            continue
        if require_repeat01 and not run_dir.name.endswith("_repeat01"):
            continue
        candidates.append(run_dir)
    return sorted(candidates)[-1] if candidates else None


def resolve_run_dir_strings(run_dirs: list[str]) -> list[Path]:
    resolved = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        if path.exists():
            resolved.append(path)
    return resolved


def find_recoverable_main5_run_dirs(report: dict[str, Any]) -> list[Path]:
    report_run_dirs = resolve_run_dir_strings(report.get("new_run_dirs", {}).get("main5_once", []))
    if report_run_dirs:
        return sorted(report_run_dirs)

    recovered = []
    missing_methods = []
    for method in DEFAULT_METHODS:
        run_dir = find_latest_run_for_method(method, require_repeat01=True, results_root=RESULTS_ROOT)
        if run_dir is None:
            missing_methods.append(method)
            continue
        recovered.append(run_dir)

    if missing_methods:
        raise FileNotFoundError(
            f"无法恢复以下方法的 repeat01 run 目录: {', '.join(missing_methods)}"
        )
    return sorted(recovered)


def generate_paper_eval_for_runs(run_dirs: list[Path], dry_run: bool) -> list[dict[str, str]]:
    generated = []
    for run_dir in run_dirs:
        method = match_run_method(run_dir.name)
        print(f"[paper_eval] method={method} results_dir={run_dir}")
        cmd = [
            sys.executable,
            str(PAPER_EVAL_SCRIPT),
            "--method",
            method,
            "--results-dir",
            str(run_dir),
        ]
        run_cmd(cmd, cwd=ROOT, dry_run=dry_run)
        generated.append({"method": method, "results_dir": str(run_dir)})
    return generated


def check_service(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "medframeqa-pipeline"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    model_ids = [row.get("id") for row in data if isinstance(row, dict)]
    return {
        "url": url,
        "ok": True,
        "model_ids": model_ids,
    }


def run_preflight(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image_count_counter = Counter()
    dataset = load_medframeqa_dataset(include_images=True)
    for sample in dataset:
        image_count_counter[len(get_image_columns(sample))] += 1

    if not set(image_count_counter).issubset({2, 3, 4, 5}):
        raise RuntimeError(f"非法 image_count 分布: {dict(sorted(image_count_counter.items()))}")

    manifest = load_split_manifest()
    shinka_bin = ensure_shinka_bin()
    service_results = []
    service_urls = [
        "http://localhost:8000/v1/models",
        "http://localhost:8001/v1/models",
        "http://localhost:8002/v1/models",
    ]
    for url in service_urls:
        try:
            service_results.append(check_service(url))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"模型服务不可用: {url} -> {exc}") from exc

    return {
        "shinka_bin": str(shinka_bin),
        "manifest_path": str(ROOT / "medframeqa_split_manifest_v1.json"),
        "manifest_version": manifest.get("version"),
        "manifest_generator": (manifest.get("generator") or {}).get("strategy"),
        "image_count_counter": dict(sorted(image_count_counter.items())),
        "services": service_results,
    }


def run_main5_once(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    before = list_run_dirs(RESULTS_ROOT)
    cmd = [
        sys.executable,
        str(REPEATS_SCRIPT),
        "--preset",
        "main5_once",
        "--profile",
        RIGOROUS_PROFILE_NAME,
    ]
    run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run)
    after = list_run_dirs(RESULTS_ROOT)
    new_run_dirs = sorted(Path(path) for path in (after - before))
    report["new_run_dirs"]["main5_once"] = [str(path) for path in new_run_dirs]
    save_report(report)
    return {
        "profile": RIGOROUS_PROFILE_NAME,
        "effective_method_specs": {
            method: get_effective_spec(method, RIGOROUS_PROFILE_NAME) for method in DEFAULT_METHODS
        },
        "new_run_dirs": [str(path) for path in new_run_dirs],
        "paper_eval_generated_for": [],
    }


def run_recover_main5_paper_eval(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    run_dirs = find_recoverable_main5_run_dirs(report)
    if len(run_dirs) != len(DEFAULT_METHODS):
        raise RuntimeError(
            f"恢复到的 main5_once run 数量不对: expected={len(DEFAULT_METHODS)} got={len(run_dirs)}"
        )
    report["new_run_dirs"]["recover_main5_paper_eval"] = [str(path) for path in run_dirs]
    save_report(report)
    generated = generate_paper_eval_for_runs(run_dirs, args.dry_run)
    return {
        "recovered_run_dirs": [str(path) for path in run_dirs],
        "paper_eval_generated_for": generated,
    }


def run_posthoc_validation(report: dict[str, Any], args: argparse.Namespace, stage_name: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(POSTHOC_SCRIPT),
        "--results-root",
        str(RESULTS_ROOT),
        "--protocol-mode",
        "independent_final_test",
        "--methods",
        *DEFAULT_METHODS,
        "--targets",
        "selected",
        "gen0",
        "--force",
    ]
    run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run)

    details = {"posthoc_eval_output_dir": str(OUTPUT_DIR / "posthoc_eval")}
    if not args.dry_run:
        rows = load_json(OUTPUT_DIR / "posthoc_eval" / "selected_vs_gen0.json")
        details["selected_vs_gen0"] = rows
    return details


def choose_bootstrap_pairs(latest_rows: list[dict[str, Any]]) -> list[str]:
    latest_by_method = {row["method"]: row for row in latest_rows}
    pairs = ["order_vote,fixed", "order_vote,order_rerank"]
    order_vote_plus = latest_by_method.get("order_vote_plus", {}).get("final_test_score")
    order_rerank = latest_by_method.get("order_rerank", {}).get("final_test_score")
    order_vote = latest_by_method.get("order_vote", {}).get("final_test_score")
    if (
        order_vote_plus is not None
        and order_rerank is not None
        and order_vote is not None
        and order_vote_plus >= order_rerank
        and (order_vote - order_vote_plus) <= 0.01
    ):
        pairs.append("order_vote_plus,order_vote")
    return pairs


def run_paper_analysis(report: dict[str, Any], args: argparse.Namespace, stage_name: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--results-root",
        str(RESULTS_ROOT),
        "--output-dir",
        str(OUTPUT_DIR),
        "--repeat-only",
        "--methods",
        *DEFAULT_METHODS,
    ]
    run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run)

    details: dict[str, Any] = {"output_dir": str(OUTPUT_DIR)}
    if args.dry_run:
        return details

    latest_rows = load_json(OUTPUT_DIR / "method_comparison_latest.json")
    decision_summary = load_json(OUTPUT_DIR / "paper_decision_summary.json")
    pairs = choose_bootstrap_pairs(latest_rows)
    cmd = [
        sys.executable,
        str(BOOTSTRAP_SCRIPT),
        "--results-root",
        str(RESULTS_ROOT),
        "--eval-name",
        "posthoc_selected_final_test",
        "--pairs",
        *pairs,
    ]
    run_cmd(cmd, cwd=ROOT, dry_run=False)
    bootstrap_rows = load_json(OUTPUT_DIR / "bootstrap_posthoc_selected_final_test.json")

    details.update(
        {
            "latest_final_test_ranking": decision_summary["latest_final_test_ranking"],
            "decision_summary": decision_summary,
            "bootstrap_pairs": pairs,
            "bootstrap_rows": bootstrap_rows,
        }
    )
    report["order_vote_plus_gate"] = {
        "qualified": not decision_summary.get("order_vote_plus_below_order_rerank", True),
        "decision_summary_path": str(OUTPUT_DIR / "paper_decision_summary.json"),
    }
    return details


def run_core_repeats(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not report.get("order_vote_plus_gate"):
        # 如果用户从这一阶段恢复，则从现有 summary 推断 gate。
        decision_summary = load_json(OUTPUT_DIR / "paper_decision_summary.json")
        report["order_vote_plus_gate"] = {
            "qualified": not decision_summary.get("order_vote_plus_below_order_rerank", True),
            "decision_summary_path": str(OUTPUT_DIR / "paper_decision_summary.json"),
        }

    methods = list(CORE_METHODS)
    if report["order_vote_plus_gate"]["qualified"]:
        methods.append("order_vote_plus")

    before = list_run_dirs(RESULTS_ROOT)
    cmd = [
        sys.executable,
        str(REPEATS_SCRIPT),
        "--methods",
        *methods,
        "--repeats",
        "2",
        "--profile",
        RIGOROUS_PROFILE_NAME,
    ]
    run_cmd(cmd, cwd=ROOT, dry_run=args.dry_run)
    after = list_run_dirs(RESULTS_ROOT)
    new_run_dirs = sorted(Path(path) for path in (after - before))
    report["new_run_dirs"]["core_repeats"] = [str(path) for path in new_run_dirs]
    report["core_run_targets"] = {
        "methods": methods,
        "target_run_count": {method: (3 if method in CORE_METHODS else 3) for method in methods},
        "profile": RIGOROUS_PROFILE_NAME,
    }
    save_report(report)
    generated = generate_paper_eval_for_runs(new_run_dirs, args.dry_run)
    return {
        "repeated_methods": methods,
        "profile": RIGOROUS_PROFILE_NAME,
        "effective_method_specs": {
            method: get_effective_spec(method, RIGOROUS_PROFILE_NAME) for method in methods
        },
        "new_run_dirs": [str(path) for path in new_run_dirs],
        "paper_eval_generated_for": generated,
    }


def run_budget_ablation(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    spec = get_effective_spec("order_vote", ORDER_VOTE_BUDGET100_PROFILE)
    BUDGET_ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    before = list_run_dirs(BUDGET_ABLATION_ROOT)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    results_dir = BUDGET_ABLATION_ROOT / f"{spec['task_name']}_budget100_{run_tag}"
    env = build_env(spec)
    cmd = build_shinka_command(
        "order_vote",
        results_dir,
        profile=ORDER_VOTE_BUDGET100_PROFILE,
        num_generations_override=100,
    )
    run_cmd(cmd, cwd=ROOT, env=env, dry_run=args.dry_run)
    after = list_run_dirs(BUDGET_ABLATION_ROOT)
    new_run_dirs = sorted(Path(path) for path in (after - before))

    if args.dry_run:
        details = {
            "new_run_dirs": [str(path) for path in new_run_dirs],
            "results_root": str(BUDGET_ABLATION_ROOT),
            "executed": False,
            "profile": ORDER_VOTE_BUDGET100_PROFILE,
            "effective_method_spec": spec,
        }
        report["budget_ablation"] = details
        return details

    generate_paper_eval_for_runs(new_run_dirs, dry_run=False)
    if not new_run_dirs:
        raise RuntimeError("100-generation ablation 没有产生新的 run 目录")

    summary_path = Path(new_run_dirs[-1]) / "paper_eval" / "paper_summary.json"
    ablation_summary = load_json(summary_path)
    aggregate_rows = load_json(OUTPUT_DIR / "method_seed_aggregate.json")
    aggregate_by_method = {row["method"]: row for row in aggregate_rows}
    current_order_vote_mean = aggregate_by_method["order_vote"]["final_test_score_mean"]
    ablation_final = (ablation_summary.get("final_test") or {}).get("final_test_score")
    delta = (ablation_final - current_order_vote_mean) if (ablation_final is not None and current_order_vote_mean is not None) else None
    details = {
        "executed": True,
        "results_root": str(BUDGET_ABLATION_ROOT),
        "new_run_dirs": [str(path) for path in new_run_dirs],
        "ablation_run": str(new_run_dirs[-1]),
        "ablation_final_test_score": ablation_final,
        "current_order_vote_mean_final_test_score": current_order_vote_mean,
        "delta_vs_50gen_mean": delta,
        "appendix_budget_sensitive": bool(delta is not None and delta >= 0.01),
        "keep_main_budget_at_50": bool(delta is None or delta < 0.005),
        "profile": ORDER_VOTE_BUDGET100_PROFILE,
        "effective_method_spec": spec,
    }
    report["budget_ablation"] = details
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MedFrameQA 论文级自动重跑 pipeline。")
    parser.add_argument(
        "--stage",
        default="full",
        choices=["full", *STAGE_ORDER],
        help="`full` 会顺序执行所有阶段，也可以只跑单个阶段。",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="只在 --stage full 时使用，从某个阶段继续执行。",
    )
    parser.add_argument(
        "--skip-budget-ablation",
        action="store_true",
        help="跳过可选的 100-generation order_vote budget ablation。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不真正启动重跑任务。",
    )
    args = parser.parse_args()

    if args.stage != "full" and args.resume_from:
        raise ValueError("--resume-from 只能和 --stage full 一起使用")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = init_report(args)
    stages_to_run = stage_subset(args)

    stage_handlers = {
        "preflight": run_preflight,
        "main5_once": run_main5_once,
        "recover_main5_paper_eval": run_recover_main5_paper_eval,
        "posthoc_validate_main5": lambda report, args: run_posthoc_validation(report, args, "posthoc_validate_main5"),
        "paper_analyze_main5": lambda report, args: run_paper_analysis(report, args, "paper_analyze_main5"),
        "core_repeats": run_core_repeats,
        "posthoc_validate_all": lambda report, args: run_posthoc_validation(report, args, "posthoc_validate_all"),
        "paper_analyze_all": lambda report, args: run_paper_analysis(report, args, "paper_analyze_all"),
        "optional_budget_ablation": run_budget_ablation,
    }

    try:
        for stage_name in stages_to_run:
            begin_stage(report, stage_name)
            details = stage_handlers[stage_name](report, args)
            finish_stage(report, stage_name, details)
        report["ended_at"] = now_ts()
        report["status"] = "completed"
        save_report(report)
        print("\nPipeline completed successfully.")
        print(f"Report: {PIPELINE_REPORT_PATH}")
    except Exception as exc:
        fail_stage(report, stage_name, exc)
        report["ended_at"] = now_ts()
        report["status"] = "failed"
        save_report(report)
        print(f"\nPipeline failed. Report: {PIPELINE_REPORT_PATH}")
        raise


if __name__ == "__main__":
    main()
