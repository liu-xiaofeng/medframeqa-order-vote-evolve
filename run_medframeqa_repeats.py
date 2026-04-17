#!/home1/xl693/anaconda3/envs/vlm/bin/python
"""顺序启动 MedFrameQA 的独立重复运行。

用途：
- 给 fixed / order_vote / order_rerank 等方法补额外独立 run；
- 这些 run 在论文里作为多次独立重复实验使用；
- 不覆盖已有结果目录。

说明：
- 当前 Shinka CLI 没有暴露一个稳定、统一的“evolution 全局 seed”参数；
- 因此这里实现的是“独立重复运行”，而不是严格意义上的可控 seed sweep；
- 论文写作时应明确表述为 independent repeats / repeated runs。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from copy import deepcopy


ROOT = Path("/gluon4/xl693/evolve")
RESULTS_ROOT = ROOT / "results"
SHINKA_BIN = Path(sys.executable).with_name("shinka_run")
RIGOROUS_PROFILE_NAME = "paper_rigorous_local"
ORDER_VOTE_BUDGET100_PROFILE = "order_vote_budget100"

RIGOROUS_SHINKA_SET_OVERRIDES = {
    "db.num_islands": "5",
    "db.archive_size": "40",
    "db.num_archive_inspirations": "4",
    "db.num_top_k_inspirations": "2",
    "db.migration_interval": "10",
    "db.migration_rate": "0.1",
    "db.parent_selection_strategy": "weighted",
    "evo.max_patch_attempts": "3",
    "evo.max_patch_resamples": "3",
    "evo.patch_types": '["diff","full"]',
    "evo.patch_type_probs": "[0.85,0.15]",
    "evo.llm_kwargs": '{"temperatures":[0.0,0.4,0.8],"max_tokens":8192}',
    "evo.use_text_feedback": "true",
}

METHOD_SPECS = {
    "fixed": {
        "task_dir": "advanced_vqa_task_fixed",
        "task_name": "advanced_vqa_task_fixed",
        "search_mini_size": "256",
        "pool_reeval_gens": "10,20,30,40,49",
        "api_timeout": "30",
        "num_generations": 50,
        "max_proposal_jobs": 4,
        "max_evaluation_jobs": 1,
    },
    "reasoning": {
        "task_dir": "advanced_vqa_task_reasoning",
        "task_name": "advanced_vqa_task_reasoning",
        "search_mini_size": "256",
        "pool_reeval_gens": "10,20,30,40,49",
        "api_timeout": "30",
        "num_generations": 50,
        "max_proposal_jobs": 4,
        "max_evaluation_jobs": 1,
    },
    "order_vote": {
        "task_dir": "advanced_vqa_task_order_vote",
        "task_name": "advanced_vqa_task_order_vote",
        "search_mini_size": "256",
        "pool_reeval_gens": "10,20,30,40,49",
        "api_timeout": "30",
        "num_generations": 50,
        "max_proposal_jobs": 4,
        "max_evaluation_jobs": 1,
    },
    "order_rerank": {
        "task_dir": "advanced_vqa_task_order_rerank",
        "task_name": "advanced_vqa_task_order_rerank",
        "search_mini_size": "128",
        "pool_reeval_gens": "10,20,30,40,49",
        "api_timeout": "20",
        "num_generations": 50,
        "max_proposal_jobs": 4,
        "max_evaluation_jobs": 1,
    },
    "order_vote_plus": {
        "task_dir": "advanced_vqa_task_order_vote_plus",
        "task_name": "advanced_vqa_task_order_vote_plus",
        "search_mini_size": "256",
        "pool_reeval_gens": "10,20,30,40,49",
        "api_timeout": "30",
        "num_generations": 50,
        "max_proposal_jobs": 4,
        "max_evaluation_jobs": 1,
    },
}

PROFILE_SPECS = {
    RIGOROUS_PROFILE_NAME: {
        "description": "论文主流程：更大的 search_mini、更密的 pool 复评、5-island 稳定高预算搜索。",
        "method_overrides": {},
        "shinka_set_overrides": deepcopy(RIGOROUS_SHINKA_SET_OVERRIDES),
    },
    ORDER_VOTE_BUDGET100_PROFILE: {
        "description": "只对 order_vote 做 100 generations 的预算敏感性分析。",
        "method_overrides": {
            "order_vote": {
                "num_generations": 100,
                "search_mini_size": "256",
                "pool_reeval_gens": "10,20,30,40,49",
            }
        },
        "shinka_set_overrides": deepcopy(RIGOROUS_SHINKA_SET_OVERRIDES),
    },
}

RERUN_PRESETS = {
    "main5_once": {
        "methods": ["fixed", "reasoning", "order_vote", "order_rerank", "order_vote_plus"],
        "repeats": 1,
        "description": "先按修复后的协议把 5 条主线各重跑 1 次。",
    },
    "core3_repeats": {
        "methods": ["fixed", "order_vote", "order_rerank"],
        "repeats": 2,
        "description": "再给主文三条线各补 2 次独立重复运行，总计 3 runs。",
    },
}


def get_effective_spec(method: str, profile: str = RIGOROUS_PROFILE_NAME) -> dict[str, str]:
    if method not in METHOD_SPECS:
        raise KeyError(f"unknown method: {method}")
    if profile not in PROFILE_SPECS:
        raise KeyError(f"unknown profile: {profile}")
    spec = dict(METHOD_SPECS[method])
    spec.update(PROFILE_SPECS[profile].get("method_overrides", {}).get(method, {}))
    return spec


def build_env(spec: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "MEDFRAMEQA_SPLIT_MANIFEST": str(ROOT / "medframeqa_split_manifest_v1.json"),
            "MEDFRAMEQA_PROTOCOL_MODE": "search_mini",
            "MEDFRAMEQA_SEARCH_MINI_SIZE": spec["search_mini_size"],
            "MEDFRAMEQA_POOL_REEVAL_GENS": spec["pool_reeval_gens"],
            "MEDFRAMEQA_API_TIMEOUT": spec["api_timeout"],
            "MEDFRAMEQA_REQUEST_RETRIES": "3",
            "MEDFRAMEQA_REQUEST_RETRY_SLEEP": "2.0",
            "MEDFRAMEQA_VLM_LOCK_PATH": "/tmp/medframeqa_vlm_8001.lock",
            "MEDFRAMEQA_VLM_LOCK_POLL_SECONDS": "5",
            "MEDFRAMEQA_VLM_LOCK_STALE_SECONDS": "1800",
            "MEDFRAMEQA_IMAGE_MAX_SIDE": "384",
            "MEDFRAMEQA_IMAGE_QUALITY": "60",
        }
    )
    return env


def ensure_shinka_bin() -> Path:
    if SHINKA_BIN.exists():
        return SHINKA_BIN
    raise FileNotFoundError(f"找不到 shinka_run: {SHINKA_BIN}")


def build_shinka_command(
    method: str,
    results_dir: Path,
    profile: str = RIGOROUS_PROFILE_NAME,
    num_generations_override: int | None = None,
) -> list[str]:
    spec = get_effective_spec(method, profile)
    cmd: list[str] = [
        str(ensure_shinka_bin()),
        "--task-dir",
        str(ROOT / spec["task_dir"]),
        "--config-fname",
        "shinka_config.yaml",
        "--results_dir",
        str(results_dir),
        "--num_generations",
        str(num_generations_override or spec["num_generations"]),
        "--max-proposal-jobs",
        str(spec["max_proposal_jobs"]),
        "--max-evaluation-jobs",
        str(spec["max_evaluation_jobs"]),
    ]
    for key, value in PROFILE_SPECS[profile]["shinka_set_overrides"].items():
        cmd.extend(["--set", f"{key}={value}"])
    return cmd


def run_one(method: str, repeat_index: int, dry_run: bool, profile: str = RIGOROUS_PROFILE_NAME):
    spec = get_effective_spec(method, profile)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_ROOT / f"{spec['task_name']}_{run_tag}_repeat{repeat_index:02d}"
    cmd = build_shinka_command(method, results_dir, profile=profile)
    print(f"\n[{method}] repeat={repeat_index}")
    print("profile:", profile)
    print("results_dir:", results_dir)
    print("command:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(
        cmd,
        env=build_env(spec),
        cwd=str(ROOT),
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="顺序启动 MedFrameQA 独立重复运行。")
    parser.add_argument(
        "--preset",
        choices=sorted(RERUN_PRESETS),
        default=None,
        help="按论文重跑方案直接选择一组方法与重复次数。",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fixed", "order_vote", "order_rerank"],
        choices=sorted(METHOD_SPECS),
    )
    parser.add_argument("--repeats", type=int, default=2, help="每个方法额外跑多少次。")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SPECS),
        default=RIGOROUS_PROFILE_NAME,
        help="使用哪套运行预算与 Shinka 搜索 profile。",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.preset is not None:
        preset = RERUN_PRESETS[args.preset]
        args.methods = list(preset["methods"])
        args.repeats = int(preset["repeats"])
        print(f"Using preset: {args.preset}")
        print(f"Description: {preset['description']}")

    print(f"Using profile: {args.profile}")
    print(f"Profile description: {PROFILE_SPECS[args.profile]['description']}")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for repeat_index in range(1, args.repeats + 1):
            run_one(method, repeat_index, args.dry_run, profile=args.profile)


if __name__ == "__main__":
    main()
