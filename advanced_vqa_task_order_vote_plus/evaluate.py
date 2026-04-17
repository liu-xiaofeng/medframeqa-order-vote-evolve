"""
order_vote_plus 评测器。

整体思路：
- 第一阶段沿用 order_vote 的多顺序直接投票；
- 只有在投票不稳定时，才触发一次 top-2 定向纠错；
- 保持统一 metrics schema，只额外记录 uncertainty_trigger_count。
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


TASK_NAME = "advanced_vqa_task_order_vote_plus"
VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "256"))
POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "10,20,30,40,49"))
API_TIMEOUT = float(os.environ.get("MEDFRAMEQA_API_TIMEOUT", "30"))


def choose_vote_winner(valid_letters, direct_votes, tie_break):
    if tie_break == "reverse_alphabetical":
        ranked = sorted(valid_letters, key=lambda letter: (-direct_votes[letter], -ord(letter)))
    else:
        ranked = sorted(valid_letters, key=lambda letter: (-direct_votes[letter], letter))
    return ranked[0] if ranked else ""


def rank_vote_letters(valid_letters, direct_votes, tie_break):
    if tie_break == "reverse_alphabetical":
        return sorted(valid_letters, key=lambda letter: (-direct_votes[letter], -ord(letter)))
    return sorted(valid_letters, key=lambda letter: (-direct_votes[letter], letter))


def should_trigger_uncertainty(direct_votes, ranked_letters, trigger_rule):
    voted_letters = [letter for letter in ranked_letters if direct_votes.get(letter, 0) > 0]
    total_votes = sum(direct_votes.values())
    if total_votes <= 1 or len(voted_letters) < 2:
        return False

    top1 = direct_votes.get(voted_letters[0], 0)
    top2 = direct_votes.get(voted_letters[1], 0)
    margin = top1 - top2
    highest_tie_count = sum(1 for letter in voted_letters if direct_votes.get(letter, 0) == top1)
    all_disagree = len(voted_letters) == total_votes and all(direct_votes.get(letter, 0) == 1 for letter in voted_letters)

    if trigger_rule == "all_disagree_only":
        return all_disagree
    if trigger_rule == "margin_or_tie":
        return margin <= 1 or highest_tie_count >= 2
    return margin <= 1 or highest_tie_count >= 2 or all_disagree


def apply_top2_fallback(top2_letters, direct_votes, tie_break, fallback_rule, vote_winner):
    if fallback_rule == "top2_tie_break" and top2_letters:
        top2_votes = Counter({letter: direct_votes.get(letter, 0) for letter in top2_letters})
        pred = choose_vote_winner(top2_letters, top2_votes, tie_break)
        if pred:
            return pred
    return vote_winner


def evaluate_sample(module, sample, client, call_stats):
    format_fn = module.format_vqa_payload
    runtime_config = module.get_runtime_config() if hasattr(module, "get_runtime_config") else {}
    valid_letters = list(string.ascii_uppercase[: len(sample["options"])])
    gt = sample["correct_answer"].strip().upper()
    order_views = runtime_config.get("order_views", 2)
    tie_break = runtime_config.get("vote_tie_break", "alphabetical")
    trigger_rule = runtime_config.get("uncertainty_trigger_rule", "margin_or_tie_or_all_disagree")
    fallback_rule = runtime_config.get("fallback_rule", "vote_winner")

    direct_votes = Counter()
    direct_debug = []
    missing_vote_count = 0

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
                "raw": raw,
                "local_pred": local_pred,
                "global_pred": global_pred,
            }
        )

    ranked_letters = rank_vote_letters(valid_letters, direct_votes, tie_break)
    vote_winner = choose_vote_winner(valid_letters, direct_votes, tie_break)
    if not vote_winner:
        vote_winner = deterministic_letter_fallback(valid_letters, tie_break)

    uncertainty_triggered = 0
    top2_rerank_debug = {}
    pred = vote_winner
    rerank_fallback_count = 0

    if should_trigger_uncertainty(direct_votes, ranked_letters, trigger_rule):
        top2_letters = [letter for letter in ranked_letters if direct_votes.get(letter, 0) > 0][:2]
        if len(top2_letters) == 2:
            uncertainty_triggered = 1
            option_indices = [ord(letter) - ord("A") for letter in top2_letters]
            local_letters = list(string.ascii_uppercase[: len(option_indices)])
            raw = generate_guided_choice_with_retries(
                client,
                lambda option_indices=option_indices: format_fn(
                    sample,
                    mode="top2_rerank",
                    option_indices=option_indices,
                    order_name="uncertain_top2",
                ),
                local_letters,
                VLM_MODEL,
                f"[top2_rerank] QID={sample.get('question_id', 'UNKNOWN')}",
                call_stats=call_stats,
            )
            local_pred = extract_option_letter(raw, local_letters)
            rerank_pred = local_to_global(local_pred, option_indices)
            if rerank_pred:
                pred = rerank_pred
            else:
                rerank_fallback_count = 1
                pred = apply_top2_fallback(
                    top2_letters,
                    direct_votes,
                    tie_break,
                    fallback_rule,
                    vote_winner,
                )
            top2_rerank_debug = {
                "top2_letters": top2_letters,
                "raw": raw,
                "local_pred": local_pred,
                "rerank_pred": rerank_pred,
            }

    if not pred:
        pred = deterministic_letter_fallback(valid_letters, tie_break)

    return {
        "gt": gt,
        "pred": pred,
        "direct_votes": dict(direct_votes),
        "direct_debug": direct_debug,
        "missing_vote_count": missing_vote_count,
        "all_votes_missing": int(sum(direct_votes.values()) == 0),
        "uncertainty_triggered": uncertainty_triggered,
        "rerank_fallback_count": rerank_fallback_count,
        "top2_rerank_debug": top2_rerank_debug,
    }


def evaluate_subset(module, subset, split_name, client, call_stats):
    correct = 0
    total = len(subset)
    missing_vote_count = 0
    all_votes_missing_count = 0
    uncertainty_trigger_count = 0
    rerank_fallback_count = 0

    for sample in subset:
        result = evaluate_sample(module, sample, client, call_stats)
        missing_vote_count += result["missing_vote_count"]
        all_votes_missing_count += result["all_votes_missing"]
        uncertainty_trigger_count += result["uncertainty_triggered"]
        rerank_fallback_count += result["rerank_fallback_count"]
        print(
            f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')} "
            f"pred={result['pred']!r} gt={result['gt']!r} direct={result['direct_votes']!r} "
            f"uncertain={result['uncertainty_triggered']}",
            file=sys.stderr,
        )
        if result["pred"] == result["gt"]:
            correct += 1

    metrics = make_protocol_metrics(split_name, correct, total)
    metrics["missing_vote_count"] = missing_vote_count
    metrics["all_votes_missing_count"] = all_votes_missing_count
    metrics["uncertainty_trigger_count"] = uncertainty_trigger_count
    metrics["rerank_fallback_count"] = rerank_fallback_count
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
        f"uncertainty_trigger_count={int(metrics.get('uncertainty_trigger_count', 0) or 0)}",
        f"rerank_fallback_count={int(metrics.get('rerank_fallback_count', 0) or 0)}",
    ]
    if metrics.get("invalid_generation"):
        lines.append(
            "The candidate failed during evaluation. Preserve the config schema and keep the top-2 correction path minimal and valid."
        )
        if metrics.get("error_message"):
            lines.append(f"error_message={metrics['error_message']}")
        return "\n".join(lines)

    lines.append(
        "Use top-2 correction only for genuinely uncertain votes, and keep the first-stage order-vote stable across option permutations."
    )
    if metrics.get("missing_vote_count", 0) > 0:
        lines.append(
            "Some local votes were empty or malformed. Tighten the direct vote prefix and keep output to one local letter."
        )
    if metrics.get("rerank_fallback_count", 0) > 0:
        lines.append(
            "Some triggered top-2 corrections fell back instead of resolving uncertainty. Make the top-2 rerank prompt more decisive."
        )
    pool_score = metrics.get("evolution_pool_score")
    search_score = metrics.get("search_score")
    if pool_score is not None and search_score is not None and pool_score + 1e-9 < search_score:
        lines.append(
            "The search-mini score is higher than the pool score. Reduce overfitting to a narrow subset and simplify the correction trigger."
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
