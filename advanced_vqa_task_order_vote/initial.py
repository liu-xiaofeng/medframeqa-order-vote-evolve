"""
顺序去偏但不做 pairwise rerank 的基线。

整体思路：
- 通过多个 option order 做直接投票；
- 进化空间被收紧为短文本 + 少量离散开关；
- 如果 JSON 配置损坏，自动回退到默认值，不让整代因为 KeyError 崩掉。
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
    "direct": "You are a radiologist choosing the best answer to a multi-image medical multiple-choice question. Return only the local choice requested.\n"
},
"direct_instruction": "Review the full image sequence and choose the best option for this vote. The options may be re-ordered; judge option content rather than fixed letter position.\n",
"sequence_instruction": "Prefer the answer that best explains the full ordered sequence, not a single frame.\n",
"metadata_instruction": "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n",
"decision_prefixes": {
    "direct": "Best local option: "
},
"image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact.",
"order_views": 2,
"vote_tie_break": "alphabetical"
}


# EVOLVE-BLOCK-START
# 说明：只编辑下面 JSON 文本中的值。
PROMPT_CONFIG_JSON = r'''
{
  "system_prompts": {
"direct": "You are a radiologist choosing the best answer to a multi-image medical multiple-choice question. Return only the local choice requested.\n"
  },
  "direct_instruction": "Review the full image sequence and choose the best option for this vote. The options may be re-ordered; judge option content rather than fixed letter position.\n",
  "sequence_instruction": "Prefer the answer that best explains the full ordered sequence, not a single frame.\n",
  "metadata_instruction": "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n",
  "decision_prefixes": {
"direct": "Best local option: "
  },
  "image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact.",
  "order_views": 2,
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
    merged["vote_tie_break"] = coerce_choice(
        candidate.get("vote_tie_break"),
        {"alphabetical", "reverse_alphabetical"},
        DEFAULT_PROMPT_CONFIG["vote_tie_break"],
    )
    return merged


def get_runtime_config():
    config = get_prompt_config()
    return {
        "task_family": "order_vote",
        "order_views": config["order_views"],
        "vote_tie_break": config["vote_tie_break"],
    }


def format_vqa_payload(sample, mode="direct", option_indices=None, order_name="identity"):
    if mode != "direct":
        raise ValueError(f"Unsupported mode for order_vote: {mode}")

    config = get_prompt_config()
    metadata_block = render_metadata_block(infer_case_metadata(sample))
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
