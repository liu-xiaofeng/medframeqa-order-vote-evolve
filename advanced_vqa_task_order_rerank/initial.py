"""
顺序投票 + 条件 rerank 的主方法候选。

整体思路：
- 先做多个 option order 的直接投票；
- 只有在离散开关允许时才进入 pairwise rerank；
- rerank 引擎本身固定，只让 Shinka 改少量短文本和离散超参数。
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
    "pairwise": "You are a radiology tie-break judge comparing two candidate answers. Return only the better local label requested by the prompt.\n"
},
"direct_instruction": "Review the full image sequence and choose the best option for this vote. The options may be re-ordered; judge option content rather than fixed letter position.\n",
"sequence_instruction": "Prefer the answer that best explains the full ordered sequence, not a single frame.\n",
"metadata_instruction": "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n",
"pairwise_instruction": "Tie-break between local option A and local option B using the full image set. Choose the better of only these two candidates.\n",
"decision_prefixes": {
    "direct": "Best local option: ",
    "pairwise": "Better local option: "
},
"image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact.",
"order_views": 2,
"rerank_topk": 2,
"rerank_trigger": "disagreement_only",
"vote_tie_break": "alphabetical"
}


# EVOLVE-BLOCK-START
# 说明：只编辑下面 JSON 文本中的值。
PROMPT_CONFIG_JSON = r'''
{
  "system_prompts": {
"direct": "You are a radiologist choosing the best answer to a multi-image medical multiple-choice question. Return only the local choice requested.\n",
"pairwise": "You are a radiology tie-break judge comparing two candidate answers. Return only the better local label requested by the prompt.\n"
  },
  "direct_instruction": "Review the full image sequence and choose the best option for this vote. The options may be re-ordered; judge option content rather than fixed letter position.\n",
  "sequence_instruction": "Prefer the answer that best explains the full ordered sequence, not a single frame.\n",
  "metadata_instruction": "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n",
  "pairwise_instruction": "Tie-break between local option A and local option B using the full image set. Choose the better of only these two candidates.\n",
  "decision_prefixes": {
"direct": "Best local option: ",
"pairwise": "Better local option: "
  },
  "image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact.",
  "order_views": 2,
  "rerank_topk": 2,
  "rerank_trigger": "disagreement_only",
  "vote_tie_break": "alphabetical"
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
    merged["rerank_topk"] = coerce_int_choice(
        candidate.get("rerank_topk"),
        {0, 2},
        DEFAULT_PROMPT_CONFIG["rerank_topk"],
    )
    merged["rerank_trigger"] = coerce_choice(
        candidate.get("rerank_trigger"),
        {"disagreement_only", "always"},
        DEFAULT_PROMPT_CONFIG["rerank_trigger"],
    )
    merged["vote_tie_break"] = coerce_choice(
        candidate.get("vote_tie_break"),
        {"alphabetical", "reverse_alphabetical"},
        DEFAULT_PROMPT_CONFIG["vote_tie_break"],
    )
    return merged


def get_runtime_config():
    config = get_prompt_config()
    return {
        "task_family": "order_rerank",
        "order_views": config["order_views"],
        "rerank_topk": config["rerank_topk"],
        "rerank_trigger": config["rerank_trigger"],
        "vote_tie_break": config["vote_tie_break"],
    }


def format_vqa_payload(sample, mode="direct", option_indices=None, order_name="identity", pair_indices=None):
    config = get_prompt_config()
    metadata_block = render_metadata_block(infer_case_metadata(sample))

    if mode == "direct":
        if option_indices is None:
            option_indices = list(range(len(sample["options"])))
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
        content_list, _ = build_content_list(
            sample,
            prompt_text,
            config["image_prompt_template"],
        )
        return [
            {"role": "system", "content": config["system_prompts"]["direct"]},
            {"role": "user", "content": content_list},
            {"role": "assistant", "content": config["decision_prefixes"]["direct"]},
        ]

    if mode == "pairwise":
        if pair_indices is None or len(pair_indices) != 2:
            raise ValueError("pair_indices must contain exactly two indices")
        prompt_text = (
            f"{config['pairwise_instruction']}"
            f"{config['metadata_instruction']}\n"
            f"{metadata_block}\n"
            f"Question:\n{sample['question']}\n\n"
            f"Candidate options:\n"
            f"{format_options(sample['options'], option_indices=pair_indices)}\n"
        )
        content_list, _ = build_content_list(
            sample,
            prompt_text,
            config["image_prompt_template"],
        )
        return [
            {"role": "system", "content": config["system_prompts"]["pairwise"]},
            {"role": "user", "content": content_list},
            {"role": "assistant", "content": config["decision_prefixes"]["pairwise"]},
        ]

    raise ValueError(f"Unsupported mode: {mode}")
