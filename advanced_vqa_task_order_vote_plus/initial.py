"""
order_vote_plus：顺序投票 + 条件 top-2 纠错。

整体思路：
- 第一阶段沿用多个 option order 的直接投票；
- 只有在投票不稳定时，才触发一次 top-2 定向纠错；
- 可进化空间仍然限制在少量文本槽位和离散开关上。
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

from medframeqa_runtime import (
    build_content_list,
    coerce_choice,
    coerce_int_choice,
    format_options,
    infer_case_metadata,
    merge_text_config,
    parse_json_config_block,
    render_metadata_block,
)


DEFAULT_PROMPT_CONFIG = {
"system_prompts": {
    "direct": "You are a radiologist choosing the best answer to a multi-image medical multiple-choice question. Return only the local choice requested.\n",
    "top2_rerank": "You are a radiologist resolving an uncertain vote between two shortlisted answers. Return only the better local choice requested.\n"
},
"direct_instruction": "Review the full image sequence and choose the best option for this vote. The options may be re-ordered; judge option content rather than fixed letter position.\n",
"sequence_instruction": "Prefer the answer that best explains the full ordered sequence, not a single frame.\n",
"metadata_instruction": "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n",
"top2_rerank_prompt": "The first-stage vote is uncertain. Compare only local option A and local option B, then return the better local choice.\n",
"decision_prefixes": {
    "direct": "Best local option: ",
    "top2_rerank": "Better local option: "
},
"image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact.",
"order_views": 2,
"vote_tie_break": "alphabetical",
"uncertainty_trigger_rule": "margin_or_tie_or_all_disagree",
"fallback_rule": "vote_winner"
}


# EVOLVE-BLOCK-START
# 说明：只编辑下面 JSON 文本中的值。
PROMPT_CONFIG_JSON = r'''
{
  "system_prompts": {
"direct": "You are a radiologist choosing the best answer to a multi-image medical multiple-choice question. Return only the local choice requested.\n",
"top2_rerank": "You are a radiologist resolving an uncertain vote between two shortlisted answers. Return only the better local choice requested.\n"
  },
  "direct_instruction": "Review the full image sequence and choose the best option for this vote. The options may be re-ordered; judge option content rather than fixed letter position.\n",
  "sequence_instruction": "Prefer the answer that best explains the full ordered sequence, not a single frame.\n",
  "metadata_instruction": "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n",
  "top2_rerank_prompt": "The first-stage vote is uncertain. Compare only local option A and local option B, then return the better local choice.\n",
  "decision_prefixes": {
"direct": "Best local option: ",
"top2_rerank": "Better local option: "
  },
  "image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact.",
  "order_views": 2,
  "vote_tie_break": "alphabetical",
  "uncertainty_trigger_rule": "margin_or_tie_or_all_disagree",
  "fallback_rule": "vote_winner"
}
'''
# EVOLVE-BLOCK-END


def generate_prompt_config():
    return parse_json_config_block(PROMPT_CONFIG_JSON)


def get_prompt_config():
    candidate = generate_prompt_config()
    merged = merge_text_config(DEFAULT_PROMPT_CONFIG, candidate)
    merged["order_views"] = coerce_int_choice(
        candidate.get("order_views"),
        {1, 2},
        DEFAULT_PROMPT_CONFIG["order_views"],
    )
    merged["vote_tie_break"] = coerce_choice(
        candidate.get("vote_tie_break"),
        {"alphabetical", "reverse_alphabetical"},
        DEFAULT_PROMPT_CONFIG["vote_tie_break"],
    )
    merged["uncertainty_trigger_rule"] = coerce_choice(
        candidate.get("uncertainty_trigger_rule"),
        {"margin_or_tie_or_all_disagree", "margin_or_tie", "all_disagree_only"},
        DEFAULT_PROMPT_CONFIG["uncertainty_trigger_rule"],
    )
    merged["fallback_rule"] = coerce_choice(
        candidate.get("fallback_rule"),
        {"vote_winner", "top2_tie_break"},
        DEFAULT_PROMPT_CONFIG["fallback_rule"],
    )
    return merged


def get_runtime_config():
    config = get_prompt_config()
    return {
        "task_family": "order_vote_plus",
        "order_views": config["order_views"],
        "vote_tie_break": config["vote_tie_break"],
        "uncertainty_trigger_rule": config["uncertainty_trigger_rule"],
        "fallback_rule": config["fallback_rule"],
    }


def format_vqa_payload(sample, mode="direct", option_indices=None, order_name="identity"):
    config = get_prompt_config()
    metadata_block = render_metadata_block(infer_case_metadata(sample))

    if option_indices is None:
        option_indices = list(range(len(sample["options"])))

    if mode == "direct":
        prompt_text = (
            f"{config['direct_instruction']}"
            f"{config['sequence_instruction']}"
            f"{config['metadata_instruction']}\n"
            f"Current option ordering for this vote: {order_name}. "
            "Treat the displayed letters as local labels only.\n\n"
            f"{metadata_block}\n"
            f"Question:\n{sample['question']}\n\n"
            f"Options for this vote:\n"
            f"{format_options(sample['options'], option_indices=option_indices)}\n"
        )
        system_prompt = config["system_prompts"]["direct"]
        answer_prefix = config["decision_prefixes"]["direct"]
    elif mode == "top2_rerank":
        if len(option_indices) != 2:
            raise ValueError("top2_rerank 模式只允许 2 个候选")
        prompt_text = (
            f"{config['top2_rerank_prompt']}"
            f"{config['sequence_instruction']}"
            f"{config['metadata_instruction']}\n"
            f"Current shortlist order: {order_name}. "
            "Judge only local option A and local option B.\n\n"
            f"{metadata_block}\n"
            f"Question:\n{sample['question']}\n\n"
            f"Shortlisted options:\n"
            f"{format_options(sample['options'], option_indices=option_indices)}\n"
        )
        system_prompt = config["system_prompts"]["top2_rerank"]
        answer_prefix = config["decision_prefixes"]["top2_rerank"]
    else:
        raise ValueError(f"Unsupported mode for order_vote_plus: {mode}")

    content_list, _ = build_content_list(
        sample,
        prompt_text,
        config["image_prompt_template"],
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list},
        {"role": "assistant", "content": answer_prefix},
    ]
