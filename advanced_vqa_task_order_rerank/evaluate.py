"""
order_rerank 评测器。

整体思路：
- 先做多个 option order 的直接投票；
- 只在允许时做 top-2 pairwise rerank；
- 空返回只丢弃该次比较，不允许整题崩掉。
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
from collections import Counter
from pathlib import Path

from shinka.core import run_shinka_eval

from medframeqa_runtime import (
    deterministic_letter_fallback,
    extract_option_letter,
    generate_guided_choice_with_retries,
    get_option_orders,
    get_protocol_meta,
    get_protocol_score_key,
    get_protocol_subset,
    load_medframeqa_dataset,
    load_mutated_module,
    load_split_manifest,
    local_to_global,
    make_openai_client,
    make_protocol_metrics,
    parse_generation_index,
    parse_generation_set,
    safe_run_experiment,
    write_json,
)


TASK_NAME = "advanced_vqa_task_order_rerank"
VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "128"))
POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "10,20,30,40,49"))
API_TIMEOUT = float(os.environ.get("MEDFRAMEQA_API_TIMEOUT", "20"))


def rank_letters(valid_letters, direct_votes, rerank_wins, tie_break):
    if tie_break == "reverse_alphabetical":
        return sorted(
            valid_letters,
            key=lambda letter: (-rerank_wins[letter], -direct_votes[letter], -ord(letter)),
        )
    return sorted(
        valid_letters,
        key=lambda letter: (-rerank_wins[letter], -direct_votes[letter], letter),
    )


def should_rerank(runtime_config, direct_votes):
    trigger = runtime_config.get("rerank_trigger", "disagreement_only")
    if runtime_config.get("rerank_topk", 2) == 0:
        return False
    if trigger == "always":
        return True
    active = [letter for letter, count in direct_votes.items() if count > 0]
    return len(active) > 1


def evaluate_sample(module, sample, client, call_stats):
    format_fn = module.format_vqa_payload
    runtime_config = module.get_runtime_config() if hasattr(module, "get_runtime_config") else {}
    valid_letters = list(string.ascii_uppercase[: len(sample["options"])])
    gt = sample["correct_answer"].strip().upper()
    order_views = runtime_config.get("order_views", 2)
    rerank_topk = runtime_config.get("rerank_topk", 2)
    tie_break = runtime_config.get("vote_tie_break", "alphabetical")

    direct_votes = Counter()
    rerank_wins = Counter()
    direct_debug = []
    rerank_debug = []
    missing_vote_count = 0
    pairwise_fallback_count = 0

    for order_name, option_indices in get_option_orders(len(sample["options"]), order_views):
        local_letters = list(string.ascii_uppercase[: len(option_indices)])
        raw = generate_guided_choice_with_retries(
            client,
            lambda option_indices=option_indices, order_name=order_name: format_fn(
                sample,
                mode="direct",
                option_indices=option_indices,
                order_name=order_name,
            ),
            local_letters,
            VLM_MODEL,
            f"[direct] QID={sample.get('question_id', 'UNKNOWN')} order={order_name}",
            call_stats=call_stats,
        )
        local_pred = extract_option_letter(raw, local_letters)
        global_pred = local_to_global(local_pred, option_indices)
        if global_pred:
            direct_votes[global_pred] += 1
        else:
            missing_vote_count += 1
        direct_debug.append(
            {
                "order_name": order_name,
                "option_indices": option_indices,
                "raw": raw,
                "local_pred": local_pred,
                "global_pred": global_pred,
            }
        )

    shortlist = rank_letters(valid_letters, direct_votes, Counter(), tie_break)[: max(rerank_topk, 2)]
    if should_rerank(runtime_config, direct_votes) and len(shortlist) >= 2:
        left = valid_letters.index(shortlist[0])
        right = valid_letters.index(shortlist[1])
        pair_orders = [(left, right), (right, left)]
        for pair_left, pair_right in pair_orders:
            pair_indices = [pair_left, pair_right]
            raw = generate_guided_choice_with_retries(
                client,
                lambda pair_indices=pair_indices: format_fn(
                    sample,
                    mode="pairwise",
                    pair_indices=pair_indices,
                ),
                ["A", "B"],
                VLM_MODEL,
                f"[pairwise] QID={sample.get('question_id', 'UNKNOWN')} pair={pair_indices}",
                call_stats=call_stats,
            )
            local_pred = extract_option_letter(raw, ["A", "B"])
            global_pred = local_to_global(local_pred, pair_indices)
            if global_pred:
                rerank_wins[global_pred] += 1
            else:
                missing_vote_count += 1
                pairwise_fallback_count += 1
            rerank_debug.append(
                {
                    "pair_indices": pair_indices,
                    "raw": raw,
                    "local_pred": local_pred,
                    "global_pred": global_pred,
                }
            )

    ranked = rank_letters(valid_letters, direct_votes, rerank_wins, tie_break)
    pred = ranked[0] if ranked else deterministic_letter_fallback(valid_letters, tie_break)
    if not pred:
        pred = deterministic_letter_fallback(valid_letters, tie_break)

    active_vote_count = sum(1 for count in direct_votes.values() if count > 0)

    return {
        "gt": gt,
        "pred": pred,
        "shortlist": shortlist,
        "direct_votes": dict(direct_votes),
        "rerank_wins": dict(rerank_wins),
        "direct_debug": direct_debug,
        "rerank_debug": rerank_debug,
        "missing_vote_count": missing_vote_count,
        "pairwise_fallback_count": pairwise_fallback_count,
        "all_votes_missing": int(sum(direct_votes.values()) == 0 and sum(rerank_wins.values()) == 0),
        "disagreement_strength": max(0, active_vote_count - 1),
    }


def evaluate_subset(module, subset, split_name, client, call_stats):
    correct = 0
    total = len(subset)
    missing_vote_count = 0
    all_votes_missing_count = 0
    pairwise_fallback_count = 0
    disagreement_total = 0

    for sample in subset:
        result = evaluate_sample(module, sample, client, call_stats)
        missing_vote_count += result["missing_vote_count"]
        all_votes_missing_count += result["all_votes_missing"]
        pairwise_fallback_count += result["pairwise_fallback_count"]
        disagreement_total += result["disagreement_strength"]
        print(
            f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')} "
            f"gt={result['gt']!r} pred={result['pred']!r} "
            f"direct={result['direct_votes']!r} rerank={result['rerank_wins']!r} "
            f"shortlist={result['shortlist']!r}",
            file=sys.stderr,
        )
        if result["pred"] == result["gt"]:
            correct += 1

    metrics = make_protocol_metrics(split_name, correct, total)
    metrics["missing_vote_count"] = missing_vote_count
    metrics["all_votes_missing_count"] = all_votes_missing_count
    metrics["pairwise_fallback_count"] = pairwise_fallback_count
    metrics["disagreement_strength"] = round(disagreement_total / total, 4) if total else 0.0
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
        f"missing_vote_count={int(metrics.get('missing_vote_count', 0) or 0)}",
        f"pairwise_fallback_count={int(metrics.get('pairwise_fallback_count', 0) or 0)}",
        f"disagreement_strength={float(metrics.get('disagreement_strength', 0.0) or 0.0):.4f}",
    ]
    if metrics.get("invalid_generation"):
        lines.append(
            "The candidate failed during evaluation. Preserve the rerank config schema and keep pairwise comparison prompts valid."
        )
        if metrics.get("error_message"):
            lines.append(f"error_message={metrics['error_message']}")
        return "\n".join(lines)

    lines.append(
        "Use reranking only to resolve genuine disagreement; the direct vote should remain strong before pairwise tie-breaking."
    )
    if metrics.get("pairwise_fallback_count", 0) > 0:
        lines.append(
            "Some pairwise comparisons returned no valid local choice. Make the pairwise prompt shorter and more decisive."
        )
    if float(metrics.get("disagreement_strength", 0.0) or 0.0) > 0.5:
        lines.append(
            "Direct votes disagree strongly across option orders. Improve shortlist stability before relying on rerank."
        )
    pool_score = metrics.get("evolution_pool_score")
    search_score = metrics.get("search_score")
    if pool_score is not None and search_score is not None and pool_score + 1e-9 < search_score:
        lines.append(
            "The search-mini score is higher than the pool score. Reduce brittle shortlist behavior and prefer globally consistent choices."
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
