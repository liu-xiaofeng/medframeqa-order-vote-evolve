"""
固定基线任务线。

整体思路：
- 这条线只允许进化极少量 prompt wording；
- 运行控制、评测协议、异常兜底都放到共享 runtime；
- 这里不允许引入新的控制流，只负责把样本组织成标准消息。
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
    format_options,
    merge_text_config,
    parse_json_config_block,
)


DEFAULT_PROMPT_CONFIG = {
"system_prompt": "You are a board-certified radiologist answering a multi-image medical multiple-choice question. Inspect all images before deciding.\n",
"task_instruction": "Answer the question using the ordered image set. The choices are labeled with capital letters.\n",
"image_instruction": "Use every image in sequence and reject options that conflict with the overall evidence.\n",
"decision_rule": "Reason silently and return exactly one capital letter.\n",
"answer_prefix": "Answer: ",
"image_prompt_template": "[IMAGE {index}/{total}]"
}


# EVOLVE-BLOCK-START
# 说明：Shinka 只应修改下面这段 JSON 文本里的值，不应该新增 Python 逻辑。
PROMPT_CONFIG_JSON = r'''
{
  "system_prompt": "You are a board-certified radiologist answering a multi-image medical multiple-choice question. Inspect all images before deciding.\n",
  "task_instruction": "Answer the question using the ordered image set. The choices are labeled with capital letters.\n",
  "image_instruction": "Use every image in sequence and reject options that conflict with the overall evidence.\n",
  "decision_rule": "Reason silently and return exactly one capital letter.\n",
  "answer_prefix": "Answer: ",
  "image_prompt_template": "[IMAGE {index}/{total}]"
}
'''
# EVOLVE-BLOCK-END


def generate_prompt_config():
    return parse_json_config_block(PROMPT_CONFIG_JSON)


def get_prompt_config():
    return merge_text_config(DEFAULT_PROMPT_CONFIG, generate_prompt_config())


def get_runtime_config():
    return {"task_family": "fixed"}


def format_vqa_payload(sample):
    config = get_prompt_config()
    question_block = (
        f"{config['task_instruction']}"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n\n"
        f"{config['image_instruction']}"
        f"{config['decision_rule']}"
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
