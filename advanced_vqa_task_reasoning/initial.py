"""
多图临床推理 prompt 空间。

整体思路：
- 只开放少量离散槽位：case/frame/sequence/option/decision；
- 变异程序只改 JSON 文本，不直接改 Python 逻辑；
- 未知字段和坏 JSON 一律回退到默认配置。
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
    infer_case_metadata,
    format_options,
    merge_text_config,
    parse_json_config_block,
    render_metadata_block,
)


DEFAULT_PROMPT_CONFIG = {
"system_prompt": "You are a radiology attending solving a multi-image medical multiple-choice question. Treat image order as clinically meaningful when relevant.\n",
"case_role": "Read the case like a radiologist writing a final impression.\n",
"frame_strategy": "For each image, identify the dominant finding, localize it, and note whether it supports or weakens candidate answers.\n",
"sequence_strategy": "Use the entire sequence. Prefer answers that explain the full ordered set of images rather than a single frame.\n",
"option_rule": "Check the options against modality, anatomy, laterality, and disease progression. Eliminate contradictions aggressively.\n",
"decision_rule": "Reason silently and return exactly one capital letter.\n",
"answer_prefix": "Final answer: ",
"image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact."
}


# EVOLVE-BLOCK-START
# 说明：只修改这段 JSON 的值，不要改函数定义。
PROMPT_CONFIG_JSON = r'''
{
  "system_prompt": "You are a radiology attending solving a multi-image medical multiple-choice question. Treat image order as clinically meaningful when relevant.\n",
  "case_role": "Read the case like a radiologist writing a final impression.\n",
  "frame_strategy": "For each image, identify the dominant finding, localize it, and note whether it supports or weakens candidate answers.\n",
  "sequence_strategy": "Use the entire sequence. Prefer answers that explain the full ordered set of images rather than a single frame.\n",
  "option_rule": "Check the options against modality, anatomy, laterality, and disease progression. Eliminate contradictions aggressively.\n",
  "decision_rule": "Reason silently and return exactly one capital letter.\n",
  "answer_prefix": "Final answer: ",
  "image_prompt_template": "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact."
}
'''
# EVOLVE-BLOCK-END


def generate_prompt_config():
    return parse_json_config_block(PROMPT_CONFIG_JSON)


def get_prompt_config():
    return merge_text_config(DEFAULT_PROMPT_CONFIG, generate_prompt_config())


def get_runtime_config():
    return {"task_family": "reasoning"}


def format_vqa_payload(sample):
    config = get_prompt_config()
    metadata = infer_case_metadata(sample)
    metadata_block = render_metadata_block(metadata)
    question_block = (
        f"{config['case_role']}"
        f"{config['frame_strategy']}"
        f"{config['sequence_strategy']}"
        f"{config['option_rule']}"
        f"{config['decision_rule']}\n"
        f"{metadata_block}\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n"
    )
    content_list, _ = build_content_list(
        sample,
        question_block,
        config["image_prompt_template"],
    )
    return [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": content_list},
        {"role": "assistant", "content": config["answer_prefix"]},
    ]
