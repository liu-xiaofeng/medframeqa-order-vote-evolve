"""
统一生成 MedFrameQA 五条实验线的任务文件与 notebook。

整体思路：
- 共享运行逻辑写在 medframeqa_runtime.py；
- 这里负责把 5 条任务线的 initial.py / evaluate.py / shinka_config.yaml
  和 notebook 一次性重生成，保证协议、注释、cell 结构完全一致。

本文件是“生成器”，不是“实验入口”。
运行它会覆盖这 5 条任务线的源文件与 notebook，但不会删除旧结果目录。
"""

import json
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path("/gluon4/xl693/evolve")
sys.path.insert(0, str(ROOT))

from create_medframeqa_split_manifest import ensure_split_manifest
from run_medframeqa_repeats import METHOD_SPECS as PIPELINE_METHOD_SPECS


METADATA = {
    "kernelspec": {
        "display_name": "vlm",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.13",
    },
}

HELPER_IMPORT_BLOCK = """
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
"""


def to_source(text):
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip("\n") + "\n")


def compose_source(body):
    body_text = dedent(body).strip("\n")
    helper_text = dedent(HELPER_IMPORT_BLOCK).strip("\n")

    # 让生成文件真正以中文总览开头，然后再补齐路径修正辅助代码。
    if body_text.startswith('"""'):
        end = body_text.find('"""', 3)
        if end != -1:
            doc_end = end + 3
            head = body_text[:doc_end]
            tail = body_text[doc_end:].lstrip("\n")
            return head + "\n\n" + helper_text + "\n\n" + tail + "\n"
    return helper_text + "\n\n" + body_text + "\n"


def json_literal(payload, indent=4):
    return json.dumps(payload, ensure_ascii=False, indent=indent)


FIXED_DEFAULT_CONFIG = {
    "system_prompt": (
        "You are a board-certified radiologist answering a multi-image medical "
        "multiple-choice question. Inspect all images before deciding.\n"
    ),
    "task_instruction": (
        "Answer the question using the ordered image set. The choices are "
        "labeled with capital letters.\n"
    ),
    "image_instruction": (
        "Use every image in sequence and reject options that conflict with "
        "the overall evidence.\n"
    ),
    "decision_rule": (
        "Reason silently and return exactly one capital letter.\n"
    ),
    "answer_prefix": "Answer: ",
    "image_prompt_template": "[IMAGE {index}/{total}]",
}

REASONING_DEFAULT_CONFIG = {
    "system_prompt": (
        "You are a radiology attending solving a multi-image medical multiple-choice "
        "question. Treat image order as clinically meaningful when relevant.\n"
    ),
    "case_role": (
        "Read the case like a radiologist writing a final impression.\n"
    ),
    "frame_strategy": (
        "For each image, identify the dominant finding, localize it, and note "
        "whether it supports or weakens candidate answers.\n"
    ),
    "sequence_strategy": (
        "Use the entire sequence. Prefer answers that explain the full ordered "
        "set of images rather than a single frame.\n"
    ),
    "option_rule": (
        "Check the options against modality, anatomy, laterality, and disease "
        "progression. Eliminate contradictions aggressively.\n"
    ),
    "decision_rule": (
        "Reason silently and return exactly one capital letter.\n"
    ),
    "answer_prefix": "Final answer: ",
    "image_prompt_template": (
        "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact."
    ),
}

ORDER_VOTE_DEFAULT_CONFIG = {
    "system_prompts": {
        "direct": (
            "You are a radiologist choosing the best answer to a multi-image medical "
            "multiple-choice question. Return only the local choice requested.\n"
        ),
    },
    "direct_instruction": (
        "Review the full image sequence and choose the best option for this vote. "
        "The options may be re-ordered; judge option content rather than fixed letter position.\n"
    ),
    "sequence_instruction": (
        "Prefer the answer that best explains the full ordered sequence, not a single frame.\n"
    ),
    "metadata_instruction": (
        "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n"
    ),
    "decision_prefixes": {
        "direct": "Best local option: ",
    },
    "image_prompt_template": (
        "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact."
    ),
    "order_views": 2,
    "vote_tie_break": "alphabetical",
}

ORDER_VOTE_PLUS_DEFAULT_CONFIG = {
    "system_prompts": {
        "direct": (
            "You are a radiologist choosing the best answer to a multi-image medical "
            "multiple-choice question. Return only the local choice requested.\n"
        ),
        "top2_rerank": (
            "You are a radiologist resolving an uncertain vote between two shortlisted "
            "answers. Return only the better local choice requested.\n"
        ),
    },
    "direct_instruction": (
        "Review the full image sequence and choose the best option for this vote. "
        "The options may be re-ordered; judge option content rather than fixed letter position.\n"
    ),
    "sequence_instruction": (
        "Prefer the answer that best explains the full ordered sequence, not a single frame.\n"
    ),
    "metadata_instruction": (
        "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n"
    ),
    "top2_rerank_prompt": (
        "The first-stage vote is uncertain. Compare only local option A and local option B, "
        "then return the better local choice.\n"
    ),
    "decision_prefixes": {
        "direct": "Best local option: ",
        "top2_rerank": "Better local option: ",
    },
    "image_prompt_template": (
        "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact."
    ),
    "order_views": 2,
    "vote_tie_break": "alphabetical",
    "uncertainty_trigger_rule": "margin_or_tie_or_all_disagree",
    "fallback_rule": "vote_winner",
}

ORDER_RERANK_DEFAULT_CONFIG = {
    "system_prompts": {
        "direct": (
            "You are a radiologist choosing the best answer to a multi-image medical "
            "multiple-choice question. Return only the local choice requested.\n"
        ),
        "pairwise": (
            "You are a radiology tie-break judge comparing two candidate answers. "
            "Return only the better local label requested by the prompt.\n"
        ),
    },
    "direct_instruction": (
        "Review the full image sequence and choose the best option for this vote. "
        "The options may be re-ordered; judge option content rather than fixed letter position.\n"
    ),
    "sequence_instruction": (
        "Prefer the answer that best explains the full ordered sequence, not a single frame.\n"
    ),
    "metadata_instruction": (
        "Use the metadata anchors only to reject options that clearly conflict with modality, organ, or sequence.\n"
    ),
    "pairwise_instruction": (
        "Tie-break between local option A and local option B using the full image set. "
        "Choose the better of only these two candidates.\n"
    ),
    "decision_prefixes": {
        "direct": "Best local option: ",
        "pairwise": "Better local option: ",
    },
    "image_prompt_template": (
        "[IMAGE {index}/{total}] Checklist: anatomy | key finding | temporal role | option impact."
    ),
    "order_views": 2,
    "rerank_topk": 2,
    "rerank_trigger": "disagreement_only",
    "vote_tie_break": "alphabetical",
}


def _fill(template, **kwargs):
    for key, value in kwargs.items():
        template = template.replace(f"<<{key}>>", value)
    normalized_lines = []
    for line in template.splitlines():
        if line.startswith("    "):
            normalized_lines.append(line[4:])
        else:
            normalized_lines.append(line)
    return compose_source("\n".join(normalized_lines))


def make_fixed_initial():
    template = """
    """
    template += '"""\n'
    template += "固定基线任务线。\n\n"
    template += "整体思路：\n"
    template += "- 这条线只允许进化极少量 prompt wording；\n"
    template += "- 运行控制、评测协议、异常兜底都放到共享 runtime；\n"
    template += "- 这里不允许引入新的控制流，只负责把样本组织成标准消息。\n"
    template += '"""\n\n'
    template += """
    from medframeqa_runtime import (
        build_content_list,
        format_options,
        merge_text_config,
        parse_json_config_block,
    )


    DEFAULT_PROMPT_CONFIG = <<DEFAULT_CONFIG>>


    # EVOLVE-BLOCK-START
    # 说明：Shinka 只应修改下面这段 JSON 文本里的值，不应该新增 Python 逻辑。
    PROMPT_CONFIG_JSON = r'''
    <<PROMPT_JSON>>
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
            f"Question:\\n{sample['question']}\\n\\n"
            f"Options:\\n{format_options(sample['options'])}\\n\\n"
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
    """
    return _fill(
        template,
        DEFAULT_CONFIG=json_literal(FIXED_DEFAULT_CONFIG, indent=4),
        PROMPT_JSON=json.dumps(FIXED_DEFAULT_CONFIG, ensure_ascii=False, indent=2),
    )


def make_reasoning_initial():
    template = '"""\n'
    template += "多图临床推理 prompt 空间。\n\n"
    template += "整体思路：\n"
    template += "- 只开放少量离散槽位：case/frame/sequence/option/decision；\n"
    template += "- 变异程序只改 JSON 文本，不直接改 Python 逻辑；\n"
    template += "- 未知字段和坏 JSON 一律回退到默认配置。\n"
    template += '"""\n\n'
    template += """
    from medframeqa_runtime import (
        build_content_list,
        infer_case_metadata,
        format_options,
        merge_text_config,
        parse_json_config_block,
        render_metadata_block,
    )


    DEFAULT_PROMPT_CONFIG = <<DEFAULT_CONFIG>>


    # EVOLVE-BLOCK-START
    # 说明：只修改这段 JSON 的值，不要改函数定义。
    PROMPT_CONFIG_JSON = r'''
    <<PROMPT_JSON>>
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
            f"{config['decision_rule']}\\n"
            f"{metadata_block}\\n"
            f"Question:\\n{sample['question']}\\n\\n"
            f"Options:\\n{format_options(sample['options'])}\\n"
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
    """
    return _fill(
        template,
        DEFAULT_CONFIG=json_literal(REASONING_DEFAULT_CONFIG, indent=4),
        PROMPT_JSON=json.dumps(REASONING_DEFAULT_CONFIG, ensure_ascii=False, indent=2),
    )


def make_order_vote_initial():
    template = '"""\n'
    template += "顺序去偏但不做 pairwise rerank 的基线。\n\n"
    template += "整体思路：\n"
    template += "- 通过多个 option order 做直接投票；\n"
    template += "- 进化空间被收紧为短文本 + 少量离散开关；\n"
    template += "- 如果 JSON 配置损坏，自动回退到默认值，不让整代因为 KeyError 崩掉。\n"
    template += '"""\n\n'
    template += """
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


    DEFAULT_PROMPT_CONFIG = <<DEFAULT_CONFIG>>


    # EVOLVE-BLOCK-START
    # 说明：只编辑下面 JSON 文本中的值。
    PROMPT_CONFIG_JSON = r'''
    <<PROMPT_JSON>>
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
            f"{config['metadata_instruction']}\\n"
            f"Current option ordering for this vote: {order_name}. "
            "Treat the displayed letters as local labels only.\\n\\n"
            f"{metadata_block}\\n"
            f"Question:\\n{sample['question']}\\n\\n"
            f"Options for this vote:\\n"
            f"{format_options(sample['options'], option_indices=option_indices)}\\n"
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
    """
    return _fill(
        template,
        DEFAULT_CONFIG=json_literal(ORDER_VOTE_DEFAULT_CONFIG, indent=4),
        PROMPT_JSON=json.dumps(ORDER_VOTE_DEFAULT_CONFIG, ensure_ascii=False, indent=2),
    )


def make_order_vote_plus_initial():
    template = '"""\n'
    template += "order_vote_plus：顺序投票 + 条件 top-2 纠错。\n\n"
    template += "整体思路：\n"
    template += "- 第一阶段沿用多个 option order 的直接投票；\n"
    template += "- 只有在投票不稳定时，才触发一次 top-2 定向纠错；\n"
    template += "- 可进化空间仍然限制在少量文本槽位和离散开关上。\n"
    template += '"""\n\n'
    template += """
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


    DEFAULT_PROMPT_CONFIG = <<DEFAULT_CONFIG>>


    # EVOLVE-BLOCK-START
    # 说明：只编辑下面 JSON 文本中的值。
    PROMPT_CONFIG_JSON = r'''
    <<PROMPT_JSON>>
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
                f"{config['metadata_instruction']}\\n"
                f"Current option ordering for this vote: {order_name}. "
                "Treat the displayed letters as local labels only.\\n\\n"
                f"{metadata_block}\\n"
                f"Question:\\n{sample['question']}\\n\\n"
                f"Options for this vote:\\n"
                f"{format_options(sample['options'], option_indices=option_indices)}\\n"
            )
            system_prompt = config["system_prompts"]["direct"]
            answer_prefix = config["decision_prefixes"]["direct"]
        elif mode == "top2_rerank":
            if len(option_indices) != 2:
                raise ValueError("top2_rerank 模式只允许 2 个候选")
            prompt_text = (
                f"{config['top2_rerank_prompt']}"
                f"{config['sequence_instruction']}"
                f"{config['metadata_instruction']}\\n"
                f"Current shortlist order: {order_name}. "
                "Judge only local option A and local option B.\\n\\n"
                f"{metadata_block}\\n"
                f"Question:\\n{sample['question']}\\n\\n"
                f"Shortlisted options:\\n"
                f"{format_options(sample['options'], option_indices=option_indices)}\\n"
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
    """
    return _fill(
        template,
        DEFAULT_CONFIG=json_literal(ORDER_VOTE_PLUS_DEFAULT_CONFIG, indent=4),
        PROMPT_JSON=json.dumps(ORDER_VOTE_PLUS_DEFAULT_CONFIG, ensure_ascii=False, indent=2),
    )


def make_order_rerank_initial():
    template = '"""\n'
    template += "顺序投票 + 条件 rerank 的主方法候选。\n\n"
    template += "整体思路：\n"
    template += "- 先做多个 option order 的直接投票；\n"
    template += "- 只有在离散开关允许时才进入 pairwise rerank；\n"
    template += "- rerank 引擎本身固定，只让 Shinka 改少量短文本和离散超参数。\n"
    template += '"""\n\n'
    template += """
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


    DEFAULT_PROMPT_CONFIG = <<DEFAULT_CONFIG>>


    # EVOLVE-BLOCK-START
    # 说明：只编辑下面 JSON 文本中的值。
    PROMPT_CONFIG_JSON = r'''
    <<PROMPT_JSON>>
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
                f"{config['metadata_instruction']}\\n"
                f"Current option ordering for this vote: {order_name}. "
                "Treat the displayed letters as local labels only.\\n\\n"
                f"{metadata_block}\\n"
                f"Question:\\n{sample['question']}\\n\\n"
                f"Options for this vote:\\n"
                f"{format_options(sample['options'], option_indices=option_indices)}\\n"
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
                f"{config['metadata_instruction']}\\n"
                f"{metadata_block}\\n"
                f"Question:\\n{sample['question']}\\n\\n"
                f"Candidate options:\\n"
                f"{format_options(sample['options'], option_indices=pair_indices)}\\n"
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
    """
    return _fill(
        template,
        DEFAULT_CONFIG=json_literal(ORDER_RERANK_DEFAULT_CONFIG, indent=4),
        PROMPT_JSON=json.dumps(ORDER_RERANK_DEFAULT_CONFIG, ensure_ascii=False, indent=2),
    )


def make_simple_evaluate(task_name, default_search_mini, pool_reeval_gens):
    template = '"""\n'
    template += "固定/推理两条纯 prompt 任务线共用的评测器。\n\n"
    template += "整体思路：\n"
    template += "- 统一走共享 runtime；\n"
    template += "- 整个评测过程受 8001 全局锁保护；\n"
    template += "- 输出统一 metrics schema，便于 paper summary 汇总。\n"
    template += '"""\n\n'
    template += """
    import argparse
    import json
    import os
    import string
    import sys
    from pathlib import Path

    from shinka.core import run_shinka_eval

    from medframeqa_runtime import (
        deterministic_letter_fallback,
        extract_option_letter,
        generate_guided_choice_with_retries,
        get_protocol_meta,
        get_protocol_score_key,
        get_protocol_subset,
        load_medframeqa_dataset,
        load_mutated_module,
        load_split_manifest,
        make_openai_client,
        make_protocol_metrics,
        parse_generation_index,
        parse_generation_set,
        safe_run_experiment,
        write_json,
    )


    TASK_NAME = "<<TASK_NAME>>"
    VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
    VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
    DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
    DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "<<DEFAULT_SEARCH_MINI>>"))
    POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "<<POOL_REEVAL_GENS>>"))
    API_TIMEOUT = float(os.environ.get("MEDFRAMEQA_API_TIMEOUT", "30"))


    def evaluate_subset(module, subset, split_name, client, call_stats):
        format_fn = module.format_vqa_payload
        correct = 0
        total = len(subset)
        missing_pred_count = 0

        for sample in subset:
            valid_letters = list(string.ascii_uppercase[: len(sample["options"])])
            gt = sample["correct_answer"].strip().upper()
            raw = generate_guided_choice_with_retries(
                client,
                lambda sample=sample: format_fn(sample),
                valid_letters,
                VLM_MODEL,
                f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')}",
                call_stats=call_stats,
            )
            pred = extract_option_letter(raw, valid_letters)
            if not pred:
                missing_pred_count += 1
                pred = deterministic_letter_fallback(valid_letters, "alphabetical")
            print(
                f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')} "
                f"raw={raw!r} pred={pred!r} gt={gt!r}",
                file=sys.stderr,
            )
            if pred == gt:
                correct += 1

        metrics = make_protocol_metrics(split_name, correct, total)
        metrics["missing_pred_count"] = missing_pred_count
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
        ]
        if metrics.get("invalid_generation"):
            lines.append(
                "The candidate failed during evaluation. Keep the program schema valid and preserve the prompt config keys."
            )
            if metrics.get("error_message"):
                lines.append(f"error_message={metrics['error_message']}")
            return "\\n".join(lines)

        lines.append(
            "Use all images in order before deciding, and reject options that conflict with modality, anatomy, laterality, or temporal sequence."
        )
        if metrics.get("missing_pred_count", 0) > 0:
            lines.append(
                "Some answers were empty or malformed. Keep the answer prefix tight and force a single capital-letter output."
            )
        search_score = metrics.get("search_score")
        pool_score = metrics.get("evolution_pool_score")
        if pool_score is not None and search_score is not None and pool_score + 1e-9 < search_score:
            lines.append(
                "The search-mini score is higher than the pool score. Make the rule less brittle and more consistent across the full evolution pool."
            )
        lines.append(
            "Prefer options supported by the whole sequence, not a single frame-level match."
        )
        return "\\n".join(lines)


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
    """
    return _fill(
        template,
        TASK_NAME=task_name,
        DEFAULT_SEARCH_MINI=str(default_search_mini),
        POOL_REEVAL_GENS=pool_reeval_gens,
    )


def make_order_vote_evaluate(task_name, default_search_mini, pool_reeval_gens):
    template = '"""\n'
    template += "order_vote 评测器。\n\n"
    template += "整体思路：\n"
    template += "- 对多个 option order 做直接投票；\n"
    template += "- 空字母只记无效票，不允许整题崩掉；\n"
    template += "- 输出统一 metrics schema。\n"
    template += '"""\n\n'
    template += """
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


    TASK_NAME = "<<TASK_NAME>>"
    VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
    VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
    DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
    DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "<<DEFAULT_SEARCH_MINI>>"))
    POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "<<POOL_REEVAL_GENS>>"))
    API_TIMEOUT = float(os.environ.get("MEDFRAMEQA_API_TIMEOUT", "30"))


    def choose_vote_winner(valid_letters, direct_votes, tie_break):
        if tie_break == "reverse_alphabetical":
            ranked = sorted(valid_letters, key=lambda letter: (-direct_votes[letter], -ord(letter)))
        else:
            ranked = sorted(valid_letters, key=lambda letter: (-direct_votes[letter], letter))
        return ranked[0] if ranked else ""


    def evaluate_sample(module, sample, client, call_stats):
        format_fn = module.format_vqa_payload
        runtime_config = module.get_runtime_config() if hasattr(module, "get_runtime_config") else {}
        valid_letters = list(string.ascii_uppercase[: len(sample["options"])])
        gt = sample["correct_answer"].strip().upper()
        order_views = runtime_config.get("order_views", 2)
        tie_break = runtime_config.get("vote_tie_break", "alphabetical")

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

        pred = choose_vote_winner(valid_letters, direct_votes, tie_break)
        if not pred:
            pred = deterministic_letter_fallback(valid_letters, tie_break)

        active_vote_count = sum(1 for count in direct_votes.values() if count > 0)

        return {
            "gt": gt,
            "pred": pred,
            "direct_votes": dict(direct_votes),
            "direct_debug": direct_debug,
            "missing_vote_count": missing_vote_count,
            "all_votes_missing": int(sum(direct_votes.values()) == 0),
            "disagreement_strength": max(0, active_vote_count - 1),
        }


    def evaluate_subset(module, subset, split_name, client, call_stats):
        correct = 0
        total = len(subset)
        missing_vote_count = 0
        all_votes_missing_count = 0
        disagreement_total = 0

        for sample in subset:
            result = evaluate_sample(module, sample, client, call_stats)
            missing_vote_count += result["missing_vote_count"]
            all_votes_missing_count += result["all_votes_missing"]
            disagreement_total += result["disagreement_strength"]
            print(
                f"[{split_name}] QID={sample.get('question_id', 'UNKNOWN')} "
                f"pred={result['pred']!r} gt={result['gt']!r} direct={result['direct_votes']!r}",
                file=sys.stderr,
            )
            if result["pred"] == result["gt"]:
                correct += 1

        metrics = make_protocol_metrics(split_name, correct, total)
        metrics["missing_vote_count"] = missing_vote_count
        metrics["all_votes_missing_count"] = all_votes_missing_count
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
            f"disagreement_strength={float(metrics.get('disagreement_strength', 0.0) or 0.0):.4f}",
        ]
        if metrics.get("invalid_generation"):
            lines.append(
                "The candidate failed during evaluation. Preserve the prompt config schema and keep local-label voting logic intact."
            )
            if metrics.get("error_message"):
                lines.append(f"error_message={metrics['error_message']}")
            return "\\n".join(lines)

        lines.append(
            "Make the vote robust to option-order changes and prefer answers supported by the full multi-image sequence."
        )
        if metrics.get("missing_vote_count", 0) > 0:
            lines.append(
                "Some local votes were empty or malformed. Tighten the decision prefix so every vote returns exactly one local letter."
            )
        if float(metrics.get("disagreement_strength", 0.0) or 0.0) > 0.5:
            lines.append(
                "Direct votes disagree strongly across option orders. Reduce sensitivity to local label permutations and emphasize global clinical consistency."
            )
        pool_score = metrics.get("evolution_pool_score")
        search_score = metrics.get("search_score")
        if pool_score is not None and search_score is not None and pool_score + 1e-9 < search_score:
            lines.append(
                "The search-mini score is higher than the pool score. Prefer more stable elimination rules that transfer beyond the mini subset."
            )
        return "\\n".join(lines)


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
    """
    return _fill(
        template,
        TASK_NAME=task_name,
        DEFAULT_SEARCH_MINI=str(default_search_mini),
        POOL_REEVAL_GENS=pool_reeval_gens,
    )


def make_order_vote_plus_evaluate(task_name, default_search_mini, pool_reeval_gens):
    template = '"""\n'
    template += "order_vote_plus 评测器。\n\n"
    template += "整体思路：\n"
    template += "- 第一阶段沿用 order_vote 的多顺序直接投票；\n"
    template += "- 只有在投票不稳定时，才触发一次 top-2 定向纠错；\n"
    template += "- 保持统一 metrics schema，只额外记录 uncertainty_trigger_count。\n"
    template += '"""\n\n'
    template += """
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


    TASK_NAME = "<<TASK_NAME>>"
    VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
    VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
    DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
    DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "<<DEFAULT_SEARCH_MINI>>"))
    POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "<<POOL_REEVAL_GENS>>"))
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
            return "\\n".join(lines)

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
        return "\\n".join(lines)


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
    """
    return _fill(
        template,
        TASK_NAME=task_name,
        DEFAULT_SEARCH_MINI=str(default_search_mini),
        POOL_REEVAL_GENS=pool_reeval_gens,
    )


def make_order_rerank_evaluate(task_name, default_search_mini, pool_reeval_gens):
    template = '"""\n'
    template += "order_rerank 评测器。\n\n"
    template += "整体思路：\n"
    template += "- 先做多个 option order 的直接投票；\n"
    template += "- 只在允许时做 top-2 pairwise rerank；\n"
    template += "- 空返回只丢弃该次比较，不允许整题崩掉。\n"
    template += '"""\n\n'
    template += """
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


    TASK_NAME = "<<TASK_NAME>>"
    VLM_URL = os.environ.get("MEDFRAMEQA_VLM_URL", "http://localhost:8001/v1")
    VLM_MODEL = os.environ.get("MEDFRAMEQA_VLM_MODEL", "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit")
    DEFAULT_PROTOCOL_MODE = os.environ.get("MEDFRAMEQA_PROTOCOL_MODE", "search_mini")
    DEFAULT_SEARCH_MINI_SIZE = int(os.environ.get("MEDFRAMEQA_SEARCH_MINI_SIZE", "<<DEFAULT_SEARCH_MINI>>"))
    POOL_REEVAL_GENS = parse_generation_set(os.environ.get("MEDFRAMEQA_POOL_REEVAL_GENS", "<<POOL_REEVAL_GENS>>"))
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
            return "\\n".join(lines)

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
        return "\\n".join(lines)


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
    """
    return _fill(
        template,
        TASK_NAME=task_name,
        DEFAULT_SEARCH_MINI=str(default_search_mini),
        POOL_REEVAL_GENS=pool_reeval_gens,
    )


SHINKA_CONFIG_YAML = """
# ShinkaEvolve 配置：
# - proposal 模型走 8000 上的 coder；
# - embedding 模型走 8002；
# - 本轮不改模型拓扑，只共享一个 8001 VLM 给评测使用；
# - 这里显式覆盖成高预算但稳定的本地搜索设置，不再吃 package 默认 medium_budget。
db:
  num_islands: 5
  archive_size: 40
  num_archive_inspirations: 4
  num_top_k_inspirations: 2
  migration_interval: 10
  migration_rate: 0.1
  parent_selection_strategy: weighted
evo:
  num_generations: 50
  llm_models:
    - "local/cyankiwi/Qwen3-Coder-Next-AWQ-4bit@http://localhost:8000/v1"
  embedding_model: "local/BAAI/bge-m3@http://localhost:8002/v1"
  max_patch_attempts: 3
  max_patch_resamples: 3
  patch_types:
    - diff
    - full
  patch_type_probs:
    - 0.85
    - 0.15
  llm_kwargs:
    temperatures:
      - 0.0
      - 0.4
      - 0.8
    max_tokens: 8192
  use_text_feedback: true
"""


TASK_SPECS = [
    {
        "task_dir": "advanced_vqa_task_fixed",
        "notebook": "sakana_medframeqa_fixed.ipynb",
        "title": "MedFrameQA 固定基线",
        "initial": make_fixed_initial,
        "evaluate": lambda: make_simple_evaluate(
            task_name="advanced_vqa_task_fixed",
            default_search_mini=int(PIPELINE_METHOD_SPECS["fixed"]["search_mini_size"]),
            pool_reeval_gens=PIPELINE_METHOD_SPECS["fixed"]["pool_reeval_gens"],
        ),
        "num_generations": PIPELINE_METHOD_SPECS["fixed"]["num_generations"],
        "search_mini_size": int(PIPELINE_METHOD_SPECS["fixed"]["search_mini_size"]),
        "pool_reeval_gens": [int(value) for value in PIPELINE_METHOD_SPECS["fixed"]["pool_reeval_gens"].split(",")],
        "api_timeout": int(PIPELINE_METHOD_SPECS["fixed"]["api_timeout"]),
        "invalid_rate_threshold": 0.05,
        "extra_env": {
            "MEDFRAMEQA_IMAGE_MAX_SIDE": "384",
            "MEDFRAMEQA_IMAGE_QUALITY": "60",
        },
    },
    {
        "task_dir": "advanced_vqa_task_reasoning",
        "notebook": "sakana_medframeqa_reasoning.ipynb",
        "title": "MedFrameQA 多图临床推理",
        "initial": make_reasoning_initial,
        "evaluate": lambda: make_simple_evaluate(
            task_name="advanced_vqa_task_reasoning",
            default_search_mini=int(PIPELINE_METHOD_SPECS["reasoning"]["search_mini_size"]),
            pool_reeval_gens=PIPELINE_METHOD_SPECS["reasoning"]["pool_reeval_gens"],
        ),
        "num_generations": PIPELINE_METHOD_SPECS["reasoning"]["num_generations"],
        "search_mini_size": int(PIPELINE_METHOD_SPECS["reasoning"]["search_mini_size"]),
        "pool_reeval_gens": [int(value) for value in PIPELINE_METHOD_SPECS["reasoning"]["pool_reeval_gens"].split(",")],
        "api_timeout": int(PIPELINE_METHOD_SPECS["reasoning"]["api_timeout"]),
        "invalid_rate_threshold": 0.05,
        "extra_env": {
            "MEDFRAMEQA_IMAGE_MAX_SIDE": "384",
            "MEDFRAMEQA_IMAGE_QUALITY": "60",
        },
    },
    {
        "task_dir": "advanced_vqa_task_order_vote",
        "notebook": "sakana_medframeqa_order_vote.ipynb",
        "title": "MedFrameQA 顺序投票基线",
        "initial": make_order_vote_initial,
        "evaluate": lambda: make_order_vote_evaluate(
            task_name="advanced_vqa_task_order_vote",
            default_search_mini=int(PIPELINE_METHOD_SPECS["order_vote"]["search_mini_size"]),
            pool_reeval_gens=PIPELINE_METHOD_SPECS["order_vote"]["pool_reeval_gens"],
        ),
        "num_generations": PIPELINE_METHOD_SPECS["order_vote"]["num_generations"],
        "search_mini_size": int(PIPELINE_METHOD_SPECS["order_vote"]["search_mini_size"]),
        "pool_reeval_gens": [int(value) for value in PIPELINE_METHOD_SPECS["order_vote"]["pool_reeval_gens"].split(",")],
        "api_timeout": int(PIPELINE_METHOD_SPECS["order_vote"]["api_timeout"]),
        "invalid_rate_threshold": 0.05,
        "extra_env": {
            "MEDFRAMEQA_IMAGE_MAX_SIDE": "384",
            "MEDFRAMEQA_IMAGE_QUALITY": "60",
        },
    },
    {
        "task_dir": "advanced_vqa_task_order_vote_plus",
        "notebook": "sakana_medframeqa_order_vote_plus.ipynb",
        "title": "MedFrameQA 顺序投票 + 条件 top-2 纠错",
        "initial": make_order_vote_plus_initial,
        "evaluate": lambda: make_order_vote_plus_evaluate(
            task_name="advanced_vqa_task_order_vote_plus",
            default_search_mini=int(PIPELINE_METHOD_SPECS["order_vote_plus"]["search_mini_size"]),
            pool_reeval_gens=PIPELINE_METHOD_SPECS["order_vote_plus"]["pool_reeval_gens"],
        ),
        "num_generations": PIPELINE_METHOD_SPECS["order_vote_plus"]["num_generations"],
        "search_mini_size": int(PIPELINE_METHOD_SPECS["order_vote_plus"]["search_mini_size"]),
        "pool_reeval_gens": [int(value) for value in PIPELINE_METHOD_SPECS["order_vote_plus"]["pool_reeval_gens"].split(",")],
        "api_timeout": int(PIPELINE_METHOD_SPECS["order_vote_plus"]["api_timeout"]),
        "invalid_rate_threshold": 0.05,
        "extra_env": {
            "MEDFRAMEQA_IMAGE_MAX_SIDE": "384",
            "MEDFRAMEQA_IMAGE_QUALITY": "60",
        },
    },
    {
        "task_dir": "advanced_vqa_task_order_rerank",
        "notebook": "sakana_medframeqa_order_rerank.ipynb",
        "title": "MedFrameQA 顺序投票 + 条件重排",
        "initial": make_order_rerank_initial,
        "evaluate": lambda: make_order_rerank_evaluate(
            task_name="advanced_vqa_task_order_rerank",
            default_search_mini=int(PIPELINE_METHOD_SPECS["order_rerank"]["search_mini_size"]),
            pool_reeval_gens=PIPELINE_METHOD_SPECS["order_rerank"]["pool_reeval_gens"],
        ),
        "num_generations": PIPELINE_METHOD_SPECS["order_rerank"]["num_generations"],
        "search_mini_size": int(PIPELINE_METHOD_SPECS["order_rerank"]["search_mini_size"]),
        "pool_reeval_gens": [int(value) for value in PIPELINE_METHOD_SPECS["order_rerank"]["pool_reeval_gens"].split(",")],
        "api_timeout": int(PIPELINE_METHOD_SPECS["order_rerank"]["api_timeout"]),
        "invalid_rate_threshold": 0.10,
        "extra_env": {
            "MEDFRAMEQA_IMAGE_MAX_SIDE": "384",
            "MEDFRAMEQA_IMAGE_QUALITY": "60",
        },
    },
]


def make_runtime_env_cell(spec):
    env_lines = [
        "DEFAULT_ENV = {",
        '    "MEDFRAMEQA_SPLIT_MANIFEST": SPLIT_MANIFEST_PATH,',
        '    "MEDFRAMEQA_PROTOCOL_MODE": "search_mini",',
        f'    "MEDFRAMEQA_SEARCH_MINI_SIZE": "{spec["search_mini_size"]}",',
        f'    "MEDFRAMEQA_POOL_REEVAL_GENS": "{",".join(str(value) for value in spec["pool_reeval_gens"])}",',
        f'    "MEDFRAMEQA_API_TIMEOUT": "{spec["api_timeout"]}",',
        '    "MEDFRAMEQA_REQUEST_RETRIES": "3",',
        '    "MEDFRAMEQA_REQUEST_RETRY_SLEEP": "2.0",',
        '    "MEDFRAMEQA_VLM_LOCK_PATH": "/tmp/medframeqa_vlm_8001.lock",',
        '    "MEDFRAMEQA_VLM_LOCK_POLL_SECONDS": "5",',
        '    "MEDFRAMEQA_VLM_LOCK_STALE_SECONDS": "1800",',
    ]
    for key, value in spec["extra_env"].items():
        env_lines.append(f'    "{key}": "{value}",')
    env_lines.append("}")
    lines = [
        "import os",
        "import time",
        "",
        "# 这格只负责声明本 notebook 的运行预算与公共环境变量。",
        f'TASK_DIR = os.path.abspath("{spec["task_dir"]}")',
        f'TASK_NAME = "{spec["task_dir"]}"',
        'RUN_TAG = time.strftime("%Y%m%d_%H%M%S")',
        f'RESULTS_DIR = os.path.abspath(os.path.join("results", f"{spec["task_dir"]}_{{RUN_TAG}}"))',
        f'SESSION_NAME = f"{spec["task_dir"]}_{{RUN_TAG}}"',
        'SPLIT_MANIFEST_PATH = os.path.abspath("medframeqa_split_manifest_v1.json")',
        f'NUM_GENERATIONS = {spec["num_generations"]}',
        f'SEARCH_MINI_SIZE = {spec["search_mini_size"]}',
        f'POOL_REEVAL_GENS = {spec["pool_reeval_gens"]}',
        f'INVALID_RATE_THRESHOLD = {spec["invalid_rate_threshold"]}',
        "MAX_PROPOSAL_JOBS = 4",
        "MAX_EVALUATION_JOBS = 1",
        *env_lines,
        "",
        "for key, value in DEFAULT_ENV.items():",
        "    os.environ[key] = value",
        "",
        'print("TASK_DIR:", TASK_DIR)',
        'print("RESULTS_DIR:", RESULTS_DIR)',
        'print("SESSION_NAME:", SESSION_NAME)',
        'print("SPLIT_MANIFEST_PATH:", SPLIT_MANIFEST_PATH)',
        'print("NUM_GENERATIONS:", NUM_GENERATIONS)',
        'print("SEARCH_MINI_SIZE:", SEARCH_MINI_SIZE)',
        'print("POOL_REEVAL_GENS:", POOL_REEVAL_GENS)',
        'print("INVALID_RATE_THRESHOLD:", INVALID_RATE_THRESHOLD)',
        'print("DEFAULT_ENV:", DEFAULT_ENV)',
    ]
    return "\n".join(lines)


def make_healthcheck_cell():
    return """
    # 检查 8000 / 8001 / 8002 三个服务是否可用。
    from openai import OpenAI

    endpoints = {
        "coder": "http://localhost:8000/v1",
        "vlm": "http://localhost:8001/v1",
        "embed": "http://localhost:8002/v1",
    }

    for name, base_url in endpoints.items():
        try:
            client = OpenAI(api_key="local", base_url=base_url, timeout=10)
            models = client.models.list()
            model_ids = [model.id for model in models.data[:5]]
            print(name, "OK", model_ids)
        except Exception as exc:
            print(name, "ERROR", exc)
    """


def make_manifest_cell():
    return """
    # 重新确认当前 frozen manifest 已写到磁盘，并打印 actual vs expected 的分布差异。
    import json
    from create_medframeqa_split_manifest import ensure_split_manifest

    manifest_path = ensure_split_manifest(SPLIT_MANIFEST_PATH)
    manifest = json.load(open(manifest_path))
    print("manifest_path:", manifest_path)
    print("manifest_version:", manifest["version"])
    print("targets:", manifest["targets"])
    print("generator:", manifest["generator"]["strategy"])
    print("restart_count:", manifest["generator"].get("restart_count"))
    print("objective_score:", manifest.get("objective_score"))
    for split_name, stats in manifest["stats"].items():
        print("\\n==", split_name, "==")
        print("size:", stats["size"], "groups:", stats["group_count"], "objective:", stats["objective_score"])
        print("actual modality:", stats["modality_distribution"])
        print("expected modality:", stats["expected_modality_distribution"])
        print("modality dev summary:", stats["relative_deviation_summary"]["modality"])
        print("actual answers:", stats["answer_distribution"])
        print("expected answers:", stats["expected_answer_distribution"])
        print("answer dev summary:", stats["relative_deviation_summary"]["answer"])
    """


def make_sanity_cell():
    return """
    # 1-sample sanity check：直接用当前 initial.py 试一题。
    import importlib.util
    import os
    from datasets import load_dataset
    from medframeqa_runtime import (
        acquire_vlm_lock,
        get_protocol_subset,
        load_split_manifest,
        make_openai_client,
    )

    initial_path = os.path.join(TASK_DIR, "initial.py")
    spec = importlib.util.spec_from_file_location(f"{TASK_NAME}_initial", initial_path)
    initial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(initial)

    dataset = load_dataset(
        "SuhaoYu1020/MedFrameQA",
        split="test",
        cache_dir=os.environ.get("MEDFRAMEQA_DATASET_CACHE", "/tmp/medframeqa_hf_cache"),
    )
    manifest = load_split_manifest(SPLIT_MANIFEST_PATH)
    subset, _ = get_protocol_subset(dataset, manifest, "search_mini", 1)
    sample = subset[0]
    messages = initial.format_vqa_payload(sample)
    valid_letters = [chr(65 + index) for index in range(len(sample["options"]))]
    client = make_openai_client("http://localhost:8001/v1", timeout=float(DEFAULT_ENV["MEDFRAMEQA_API_TIMEOUT"]))

    with acquire_vlm_lock(task_name=TASK_NAME, results_dir=RESULTS_DIR, mode="sanity_check"):
        response = client.chat.completions.create(
            model="cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit",
            messages=messages,
            temperature=0.0,
            max_tokens=4,
            extra_body={
                "guided_choice": valid_letters,
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )

    print("question_id:", sample["question_id"])
    print("raw:", repr(response.choices[0].message.content))
    print("gt:", sample["correct_answer"])
    """


def make_smoke_eval_cell(spec):
    return f"""
    # 5-sample smoke test：确认 evaluator、锁、metrics schema 都能跑通。
    import json
    import os
    import signal
    import subprocess
    import sys
    import time

    smoke_results = os.path.join("/tmp", f"{spec["task_dir"]}_smoketest_{{RUN_TAG}}")
    env = dict(os.environ)
    env.update(DEFAULT_ENV)
    env["MEDFRAMEQA_PROTOCOL_MODE"] = "search_mini"
    env["MEDFRAMEQA_SEARCH_MINI_SIZE"] = "5"
    evaluate_path = os.path.join(TASK_DIR, "evaluate.py")
    initial_path = os.path.join(TASK_DIR, "initial.py")

    cmd = [
        sys.executable,
        evaluate_path,
        "--program_path",
        initial_path,
        "--results_dir",
        smoke_results,
        "--single_run",
        "--protocol_mode",
        "search_mini",
        "--search_mini_size",
        "5",
    ]

    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    print("Smoke test PGID:", proc.pid)
    try:
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
    except KeyboardInterrupt:
        print("Interrupt received; terminating smoke test process group:", proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(2)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise

    print("smoke_results:", smoke_results)
    print(json.load(open(os.path.join(smoke_results, "metrics.json"))))
    """


def make_launch_cell():
    return """
    # 正式运行：阻塞等待 shinka_run 完成，并把输出直接显示在 notebook 里。
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    env = dict(os.environ)
    env.update(DEFAULT_ENV)

    shinka_bin = Path(sys.executable).with_name("shinka_run")
    if not shinka_bin.exists():
        fallback = None
        for prefix in env.get("PATH", "").split(os.pathsep):
            candidate = Path(prefix) / "shinka_run"
            if candidate.exists():
                fallback = candidate
                break
        if fallback is None:
            raise RuntimeError(f"找不到 shinka_run: {shinka_bin}")
        shinka_bin = fallback

    cmd = [
        str(shinka_bin),
        "--task-dir",
        TASK_DIR,
        "--config-fname",
        "shinka_config.yaml",
        "--results_dir",
        RESULTS_DIR,
        "--num_generations",
        str(NUM_GENERATIONS),
        "--max-proposal-jobs",
        str(MAX_PROPOSAL_JOBS),
        "--max-evaluation-jobs",
        str(MAX_EVALUATION_JOBS),
    ]

    print("Blocking run command:")
    print(" ".join(cmd))
    print("RESULTS_DIR:", RESULTS_DIR)

    proc = subprocess.Popen(cmd, env=env, cwd=os.path.abspath("."), start_new_session=True)
    print("Run PGID:", proc.pid)
    try:
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
    except KeyboardInterrupt:
        print("Interrupt received; terminating run process group:", proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(2)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise

    print("Run finished. RESULTS_DIR:", RESULTS_DIR)
    """


def make_plot_cell():
    return """
    # 结果可视化：画每一代 combined_score，并标出 invalid generation。
    import math
    from medframeqa_runtime import collect_generation_records, select_top_k_records

    records = collect_generation_records(RESULTS_DIR)
    print("generation_count:", len(records))
    if not records:
        raise RuntimeError("当前 RESULTS_DIR 没有 generation 结果。请先完成正式运行。")

    top_records = select_top_k_records(records, top_k=3)
    print("top3 generations:", [row.get("generation") for row in top_records])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 不可用，跳过绘图。")
        for row in records:
            print(row.get("generation"), row.get("combined_score"), row.get("invalid_generation"))
    else:
        generations = [row.get("generation") for row in records]
        scores = [row.get("combined_score", 0.0) for row in records]
        invalid_generations = [row.get("generation") for row in records if row.get("invalid_generation")]
        invalid_scores = [row.get("combined_score", 0.0) for row in records if row.get("invalid_generation")]

        best_so_far = []
        current_best = -math.inf
        for score in scores:
            current_best = max(current_best, score)
            best_so_far.append(current_best)

        plt.figure(figsize=(10, 4.5))
        plt.plot(generations, scores, color="#1565c0", marker="o", linewidth=1.6, label="combined_score")
        plt.plot(generations, best_so_far, color="#2e7d32", linestyle="--", linewidth=1.4, label="best_so_far")
        if invalid_generations:
            plt.scatter(invalid_generations, invalid_scores, color="#c62828", marker="x", s=80, label="invalid_generation")
        plt.xlabel("Generation")
        plt.ylabel("Combined Score")
        plt.title(TASK_NAME + " evolution curve")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()
    """


def make_postrun_cell(spec):
    milestone_list = spec["pool_reeval_gens"]
    return f"""
    # 论文汇总：统计 valid/invalid rate，并对 top-3 跑 holdout，最后选一个跑 final test。
    import json
    import os
    import subprocess
    import sys

    from medframeqa_runtime import (
        collect_generation_records,
        select_best_so_far,
        select_top_k_records,
        write_csv_rows,
        write_json,
    )

    PAPER_DIR = os.path.join(RESULTS_DIR, "paper_eval")
    os.makedirs(PAPER_DIR, exist_ok=True)

    def run_direct_eval(program_path, protocol_mode, out_name):
        out_dir = os.path.join(PAPER_DIR, out_name)
        env = dict(os.environ)
        env.update(DEFAULT_ENV)
        cmd = [
            sys.executable,
            os.path.join("{spec["task_dir"]}", "evaluate.py"),
            "--program_path",
            program_path,
            "--results_dir",
            out_dir,
            "--single_run",
            "--protocol_mode",
            protocol_mode,
            "--search_mini_size",
            str(SEARCH_MINI_SIZE),
        ]
        subprocess.run(cmd, env=env, check=True)
        return json.load(open(os.path.join(out_dir, "metrics.json")))

    records = collect_generation_records(RESULTS_DIR)
    write_csv_rows(os.path.join(PAPER_DIR, "generation_records.csv"), records)
    write_json(os.path.join(PAPER_DIR, "generation_records.json"), records)

    valid_records = [row for row in records if not row.get("invalid_generation")]
    invalid_records = [row for row in records if row.get("invalid_generation")]
    valid_rate = (len(valid_records) / len(records)) if records else 0.0
    invalid_rate = (len(invalid_records) / len(records)) if records else 0.0
    best_record = select_top_k_records(records, top_k=1)
    best_record = best_record[0] if best_record else None
    last_record = records[-1] if records else None

    milestone_rows = []
    max_generation = records[-1]["generation"] if records else None
    milestone_generations = sorted(
        set(
            [value for value in {milestone_list} if isinstance(value, int)]
            + ([max_generation] if max_generation is not None else [])
        )
    )
    for milestone in milestone_generations:
        record = select_best_so_far(records, milestone)
        if record is None:
            continue
        metrics = run_direct_eval(
            record["program_path"],
            "evolution_pool",
            f"milestone_gen_{{milestone}}_pool_eval",
        )
        milestone_rows.append({{
            "milestone_generation": milestone,
            "selected_generation": record["generation"],
            "selected_program_path": record["program_path"],
            **metrics,
        }})
    write_csv_rows(os.path.join(PAPER_DIR, "milestone_pool_eval.csv"), milestone_rows)
    write_json(os.path.join(PAPER_DIR, "milestone_pool_eval.json"), milestone_rows)

    top_records = select_top_k_records(records, top_k=3)
    holdout_rows = []
    for rank, record in enumerate(top_records, 1):
        metrics = run_direct_eval(
            record["program_path"],
            "selection_holdout",
            f"top{{rank}}_holdout_eval",
        )
        holdout_rows.append({{
            "rank": rank,
            "generation": record["generation"],
            "program_path": record["program_path"],
            **metrics,
        }})
    write_csv_rows(os.path.join(PAPER_DIR, "top3_holdout_eval.csv"), holdout_rows)
    write_json(os.path.join(PAPER_DIR, "top3_holdout_eval.json"), holdout_rows)

    best_holdout = None
    if holdout_rows:
        best_holdout = sorted(
            holdout_rows,
            key=lambda row: (
                -row.get("holdout_score", 0.0),
                row.get("generation", 10**9),
            ),
        )[0]

    summary = {{
        "task_name": TASK_NAME,
        "results_dir": RESULTS_DIR,
        "generation_count": len(records),
        "valid_generation_count": len(valid_records),
        "invalid_generation_count": len(invalid_records),
        "valid_generation_rate": round(valid_rate, 4),
        "invalid_generation_rate": round(invalid_rate, 4),
        "invalid_rate_threshold": INVALID_RATE_THRESHOLD,
        "paper_ready_candidate": invalid_rate <= INVALID_RATE_THRESHOLD,
        "best_generation": best_record["generation"] if best_record else None,
        "best_combined_score": best_record.get("combined_score") if best_record else None,
        "last_generation": last_record["generation"] if last_record else None,
        "last_combined_score": last_record.get("combined_score") if last_record else None,
        "top3_generations": [row["generation"] for row in top_records],
        "milestone_pool": milestone_rows,
        "top3_holdout": holdout_rows,
    }}
    if best_holdout is not None:
        final_metrics = run_direct_eval(
            best_holdout["program_path"],
            "independent_final_test",
            "final_test_eval",
        )
        summary["selected_generation"] = best_holdout["generation"]
        summary["selected_program_path"] = best_holdout["program_path"]
        summary["final_test"] = final_metrics

    write_json(os.path.join(PAPER_DIR, "paper_summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    """


def make_notebook(spec, sources, helper_sources):
    task_dir = spec["task_dir"]
    notebook = {
        "cells": [
            md_cell(
                f"""
                # {spec["title"]}

                这份 notebook 使用统一论文协议：

                - 先确认模型服务和 frozen manifest；
                - 再做 1-sample sanity 和 5-sample smoke；
                - 正式实验直接在当前 cell 内阻塞运行，不再依赖 `tmux`；
                - 跑完后直接本地画出 generation 曲线；
                - 最后导出 valid/invalid rate、top-3、holdout、final test 的 paper summary。
                """
            ),
            code_cell("!nvidia-smi"),
            md_cell(
                """
                `8000/8001/8002` 模型服务建议通过 `/gluon4/xl693/start_models.sh` 在 `tmux` 里常驻。  
                本 notebook 不会去停止模型服务；如果要中断实验，直接中断当前运行 cell 即可。
                """
            ),
            code_cell(make_runtime_env_cell(spec)),
            code_cell(make_healthcheck_cell()),
            code_cell(f"%%writefile medframeqa_runtime.py\n{helper_sources['medframeqa_runtime.py']}"),
            code_cell(
                f"%%writefile create_medframeqa_split_manifest.py\n{helper_sources['create_medframeqa_split_manifest.py']}"
            ),
            code_cell(f'import os\nos.makedirs("{task_dir}", exist_ok=True)'),
            code_cell(f"%%writefile {task_dir}/initial.py\n{sources['initial.py']}"),
            code_cell(f"%%writefile {task_dir}/evaluate.py\n{sources['evaluate.py']}"),
            code_cell(f"%%writefile {task_dir}/shinka_config.yaml\n{sources['shinka_config.yaml']}"),
            code_cell(make_manifest_cell()),
            code_cell(make_sanity_cell()),
            code_cell(make_smoke_eval_cell(spec)),
            code_cell(make_launch_cell()),
            code_cell(make_plot_cell()),
            code_cell(make_postrun_cell(spec)),
        ],
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def make_paper_analysis_notebook():
    notebook = {
        "cells": [
            md_cell(
                """
                # MedFrameQA 论文结果分析

                这份 notebook 不跑新实验，只做 5 条方法线的结果汇总：

                - `fixed`
                - `reasoning`
                - `order_vote`
                - `order_rerank`
                - `order_vote_plus`

                它会先调用统一汇总脚本，再读入：

                - 最新 high-budget repeat run 对比
                - 同协议 repeat-only complete runs
                - 按方法聚合的重复运行统计（publication-clean）
                """
            ),
            code_cell(
                """
                import json
                import os
                import subprocess
                import sys
                from pathlib import Path

                ROOT = Path("/gluon4/xl693/evolve")
                RESULTS_ROOT = ROOT / "results"
                OUTPUT_DIR = ROOT / "paper_analysis_output"
                BUDGET_ABLATION_ROOT = ROOT / "results_budget_ablation"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                MIN_GENERATIONS = 50
                METHODS = ["fixed", "reasoning", "order_vote", "order_rerank", "order_vote_plus"]
                REPEAT_ONLY = True
                SUMMARY_SCRIPT = ROOT / "summarize_medframeqa_paper_runs.py"

                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                print("ROOT:", ROOT)
                print("RESULTS_ROOT:", RESULTS_ROOT)
                print("BUDGET_ABLATION_ROOT:", BUDGET_ABLATION_ROOT)
                print("OUTPUT_DIR:", OUTPUT_DIR)
                print("MIN_GENERATIONS:", MIN_GENERATIONS)
                print("METHODS:", METHODS)
                print("REPEAT_ONLY:", REPEAT_ONLY)
                print("SUMMARY_SCRIPT:", SUMMARY_SCRIPT)
                """
            ),
            code_cell(
                """
                cmd = [
                    sys.executable,
                    str(SUMMARY_SCRIPT),
                    "--results-root",
                    str(RESULTS_ROOT),
                    "--output-dir",
                    str(OUTPUT_DIR),
                    "--budget-ablation-root",
                    str(BUDGET_ABLATION_ROOT),
                    "--repeat-only",
                    "--methods",
                    *METHODS,
                ]
                print("Running summary command:")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                from medframeqa_runtime import collect_generation_records

                latest_rows = json.loads((OUTPUT_DIR / "method_comparison_latest.json").read_text())
                all_rows = json.loads((OUTPUT_DIR / "method_comparison_all_runs.json").read_text())
                aggregate_rows = json.loads((OUTPUT_DIR / "method_seed_aggregate.json").read_text())
                main_table_rows = json.loads((OUTPUT_DIR / "main_table_repeat_only.json").read_text())
                budget_sensitivity = json.loads((OUTPUT_DIR / "budget_sensitivity_summary.json").read_text())
                decision_summary = json.loads((OUTPUT_DIR / "paper_decision_summary.json").read_text())

                latest_by_method = {row["method"]: row for row in latest_rows}
                artifacts = []
                for row in latest_rows:
                    run_dir = Path(row["run_dir"])
                    generation_records = collect_generation_records(run_dir)
                    artifacts.append(
                        {
                            "method": row["method"],
                            "row": row,
                            "run_dir": run_dir,
                            "generation_records": generation_records,
                        }
                    )

                print("latest_rows:")
                for row in latest_rows:
                    print(row)

                print("\\ndecision_summary:")
                print(json.dumps(decision_summary, indent=2, ensure_ascii=False))
                print("\\nbudget_sensitivity:")
                print(json.dumps(budget_sensitivity, indent=2, ensure_ascii=False))
                """
            ),
            code_cell(
                """
                comparison_rows = latest_rows
                print("main_table_rows:")
                for row in main_table_rows:
                    print(row)

                print("\\nall aggregate_rows:")
                for row in aggregate_rows:
                    print(row)
                """
            ),
            code_cell(
                """
                print("paper decision checks:")
                print("- current_main_candidate:", decision_summary["current_main_candidate"])
                print("- core_methods_seed_complete:", decision_summary["core_methods_seed_complete"])
                print("- reasoning_not_better_than_fixed:", decision_summary["reasoning_not_better_than_fixed"])
                print("- order_vote_plus_below_order_rerank:", decision_summary["order_vote_plus_below_order_rerank"])
                print("- keep_main_budget_at_50:", decision_summary["keep_main_budget_at_50"])

                if not decision_summary["core_methods_seed_complete"]:
                    print("\\nWARNING: fixed / order_vote / order_rerank 还没有都补到 3 runs，不能作为最终 NeurIPS 主表。")

                if budget_sensitivity["available"]:
                    print("\\nBudget sensitivity summary:")
                    print("- 100-gen final_test_score:", budget_sensitivity["final_test_score"])
                    print("- 50-gen repeat-only mean final_test_score:", budget_sensitivity["fifty_gen_repeat_only_mean_final_test_score"])
                    print("- delta_vs_50gen_mean_final_test:", budget_sensitivity["delta_vs_50gen_mean_final_test"])
                    print("- interpretation:", budget_sensitivity["interpretation"])
                """
            ),
            code_cell(
                """
                import matplotlib.pyplot as plt

                plt.figure(figsize=(11, 5))
                for artifact in artifacts:
                    valid_rows = [
                        row
                        for row in artifact["generation_records"]
                        if row.get("generation") is not None and row.get("combined_score") is not None
                    ]
                    generations = [row["generation"] for row in valid_rows]
                    scores = [row["combined_score"] for row in valid_rows]
                    plt.plot(generations, scores, marker="o", linewidth=1.5, label=artifact["method"])

                plt.xlabel("Generation")
                plt.ylabel("Combined Score")
                plt.title("MedFrameQA evolution curves")
                plt.grid(alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.show()
                """
            ),
            code_cell(
                """
                final_rows = sorted(comparison_rows, key=lambda row: (-row.get("final_test_score", 0.0), row.get("method")))
                print("final test ranking:")
                for rank, row in enumerate(final_rows, 1):
                    print(
                        rank,
                        row["method"],
                        "| final_test_score =", row["final_test_score"],
                        "| selected_holdout_score =", row["selected_holdout_score"],
                        "| invalid_rate =", row["invalid_generation_rate"],
                    )
                """
            ),
            code_cell(
                """
                import matplotlib.pyplot as plt

                order_vote_main = next(row for row in aggregate_rows if row["method"] == "order_vote")
                if budget_sensitivity["available"]:
                    labels = ["order_vote 50-gen mean", "order_vote 100-gen single"]
                    values = [
                        order_vote_main["final_test_score_mean"],
                        budget_sensitivity["final_test_score"],
                    ]
                    plt.figure(figsize=(7, 4))
                    bars = plt.bar(labels, values, color=["#2a9d8f", "#e76f51"])
                    plt.ylabel("Final Test Score")
                    plt.title("Order-Vote Budget Sensitivity")
                    plt.ylim(min(values) - 0.03, max(values) + 0.03)
                    for bar, value in zip(bars, values):
                        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center")
                    plt.tight_layout()
                    plt.show()
                else:
                    print("No budget sensitivity result found.")
                """
            ),
        ],
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def make_final_validation_notebook():
    notebook = {
        "cells": [
            md_cell(
                """
                # MedFrameQA 最终验证

                这份 notebook 专门服务于论文补实验：

                - 对 `selected_generation` 与 `gen_0` 跑 post-hoc final test
                - 导出逐题预测
                - 生成 `modality / image_count` breakdown
                - 计算 paired bootstrap
                """
            ),
            code_cell(
                """
                import json
                import subprocess
                import sys
                from pathlib import Path

                ROOT = Path("/gluon4/xl693/evolve")
                RESULTS_ROOT = ROOT / "results"
                OUTPUT_DIR = ROOT / "paper_analysis_output"
                METHODS = ["fixed", "reasoning", "order_vote", "order_rerank", "order_vote_plus"]
                POSTHOC_SCRIPT = ROOT / "medframeqa_posthoc_eval.py"
                BOOTSTRAP_SCRIPT = ROOT / "medframeqa_paired_bootstrap.py"

                print("ROOT:", ROOT)
                print("RESULTS_ROOT:", RESULTS_ROOT)
                print("OUTPUT_DIR:", OUTPUT_DIR)
                print("METHODS:", METHODS)
                """
            ),
            code_cell(
                """
                from collections import Counter

                from medframeqa_runtime import get_image_columns, load_medframeqa_dataset

                dataset = load_medframeqa_dataset(include_images=True)
                image_count_counter = Counter(len(get_image_columns(sample)) for sample in dataset)
                print("dataset image_count_counter:", dict(sorted(image_count_counter.items())))
                assert set(image_count_counter).issubset({2, 3, 4, 5}), image_count_counter
                """
            ),
            code_cell(
                """
                # 这格会真正运行 post-hoc final test，成本较高。
                cmd = [
                    sys.executable,
                    str(POSTHOC_SCRIPT),
                    "--results-root",
                    str(RESULTS_ROOT),
                    "--protocol-mode",
                    "independent_final_test",
                    "--methods",
                    *METHODS,
                    "--targets",
                    "selected",
                    "gen0",
                ]
                print("Running posthoc command:")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                posthoc_rows = json.loads((OUTPUT_DIR / "posthoc_eval" / "posthoc_eval_rows.json").read_text())
                selected_vs_gen0 = json.loads((OUTPUT_DIR / "posthoc_eval" / "selected_vs_gen0.json").read_text())

                print("selected_vs_gen0:")
                for row in selected_vs_gen0:
                    print(row)
                """
            ),
            code_cell(
                """
                protocol_checks = {}
                for row in posthoc_rows:
                    if row["target"] != "selected":
                        continue
                    check_path = Path(row["output_dir"]) / "protocol_checks.json"
                    protocol_checks[row["method"]] = json.loads(check_path.read_text())

                print("protocol_checks:")
                for method, payload in protocol_checks.items():
                    print(method, payload)
                """
            ),
            code_cell(
                """
                import matplotlib.pyplot as plt

                delta_rows = [row for row in selected_vs_gen0 if row.get("delta_selected_minus_gen0") is not None]
                methods = [row["method"] for row in delta_rows]
                deltas = [row["delta_selected_minus_gen0"] for row in delta_rows]

                plt.figure(figsize=(9, 4))
                plt.bar(methods, deltas)
                plt.axhline(0.0, color="black", linewidth=1)
                plt.ylabel("selected - gen0 final test score")
                plt.title("Evolution gain on independent final test")
                plt.tight_layout()
                plt.show()
                """
            ),
            code_cell(
                """
                selected_scores = {row["method"]: row["selected_final_test_score"] for row in selected_vs_gen0}
                pairs = ["order_vote,fixed", "order_vote,order_rerank"]
                if (
                    selected_scores.get("order_vote_plus") is not None
                    and selected_scores.get("order_rerank") is not None
                    and selected_scores.get("order_vote") is not None
                    and selected_scores["order_vote_plus"] >= selected_scores["order_rerank"]
                    and (selected_scores["order_vote"] - selected_scores["order_vote_plus"]) <= 0.01
                ):
                    pairs.append("order_vote_plus,order_vote")

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
                print("Running bootstrap command:")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                bootstrap_rows = json.loads((OUTPUT_DIR / "bootstrap_posthoc_selected_final_test.json").read_text())
                print("bootstrap_rows:")
                for row in bootstrap_rows:
                    print(row)
                """
            ),
            code_cell(
                """
                from collections import defaultdict

                breakdown_dir_map = {}
                for row in posthoc_rows:
                    if row["target"] == "selected":
                        breakdown_dir_map[row["method"]] = Path(row["output_dir"])

                modality_tables = {}
                image_count_tables = {}
                for method in ["fixed", "order_vote", "order_rerank", "order_vote_plus"]:
                    out_dir = breakdown_dir_map.get(method)
                    if out_dir is None:
                        continue
                    modality_tables[method] = json.loads((out_dir / "breakdown_modality.json").read_text())
                    image_count_tables[method] = json.loads((out_dir / "breakdown_image_count.json").read_text())

                print("modality_tables:")
                for method, rows in modality_tables.items():
                    print(method, rows)

                print("\\nimage_count_tables:")
                for method, rows in image_count_tables.items():
                    print(method, rows)
                """
            ),
        ],
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def make_rerun_protocol_notebook():
    notebook = {
        "cells": [
            md_cell(
                """
                # MedFrameQA 论文级自动 pipeline

                这份 notebook 是单入口包装器，直接调用 `run_medframeqa_paper_pipeline.py`。

                默认完整流程：

                1. `preflight`
                2. `main5_once`
                3. `recover_main5_paper_eval`
                4. `posthoc_validate_main5`
                5. `paper_analyze_main5`
                6. `core_repeats`
                7. `posthoc_validate_all`
                8. `paper_analyze_all`
                9. `optional_budget_ablation`
                """
            ),
            code_cell(
                """
                import json
                import subprocess
                import sys
                from pathlib import Path

                ROOT = Path("/gluon4/xl693/evolve")
                PIPELINE_SCRIPT = ROOT / "run_medframeqa_paper_pipeline.py"
                OUTPUT_DIR = ROOT / "paper_analysis_output"
                REPORT_PATH = OUTPUT_DIR / "pipeline_report.json"

                print("ROOT:", ROOT)
                print("PIPELINE_SCRIPT:", PIPELINE_SCRIPT)
                print("REPORT_PATH:", REPORT_PATH)
                """
            ),
            code_cell(
                """
                # 只跑 preflight，确认协议和服务都正常。
                cmd = [sys.executable, str(PIPELINE_SCRIPT), "--stage", "preflight"]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                # 完整自动流程：默认会包含可选的 budget ablation。
                cmd = [sys.executable, str(PIPELINE_SCRIPT), "--stage", "full"]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                # 如果 main5_once 已经跑完，但 paper_eval 中途失败，用这格只恢复当前 5 条 repeat01 结果。
                cmd = [sys.executable, str(PIPELINE_SCRIPT), "--stage", "recover_main5_paper_eval"]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                # 如果你不想当前就跑 100-generation budget ablation，用这格。
                cmd = [sys.executable, str(PIPELINE_SCRIPT), "--stage", "full", "--skip-budget-ablation"]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                # 断点恢复示例：先从 recover_main5_paper_eval 接着恢复现有 batch。
                cmd = [sys.executable, str(PIPELINE_SCRIPT), "--stage", "full", "--resume-from", "recover_main5_paper_eval"]
                print("Running:", " ".join(cmd))
                print("如果要真正执行，把下面一行取消注释。")
                # subprocess.run(cmd, check=True)
                """
            ),
            code_cell(
                """
                if REPORT_PATH.exists():
                    report = json.loads(REPORT_PATH.read_text())
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                else:
                    print("尚未发现 pipeline_report.json")
                """
            ),
        ],
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def materialize_tasks():
    task_sources = {}
    for spec in TASK_SPECS:
        task_dir = ROOT / spec["task_dir"]
        sources = {
            "initial.py": spec["initial"](),
            "evaluate.py": spec["evaluate"](),
            "shinka_config.yaml": dedent(SHINKA_CONFIG_YAML).strip("\n") + "\n",
        }
        for relative_path, content in sources.items():
            write_text(task_dir / relative_path, content)
        task_sources[spec["task_dir"]] = sources
    return task_sources


def materialize_notebooks(task_sources):
    helper_sources = {
        "medframeqa_runtime.py": (ROOT / "medframeqa_runtime.py").read_text(),
        "create_medframeqa_split_manifest.py": (ROOT / "create_medframeqa_split_manifest.py").read_text(),
    }
    for spec in TASK_SPECS:
        notebook = make_notebook(spec, task_sources[spec["task_dir"]], helper_sources)
        (ROOT / spec["notebook"]).write_text(json.dumps(notebook, indent=2, ensure_ascii=False))
    analysis_notebook = make_paper_analysis_notebook()
    (ROOT / "sakana_medframeqa_paper_analysis.ipynb").write_text(
        json.dumps(analysis_notebook, indent=2, ensure_ascii=False)
    )
    final_validation_notebook = make_final_validation_notebook()
    (ROOT / "sakana_medframeqa_final_validation.ipynb").write_text(
        json.dumps(final_validation_notebook, indent=2, ensure_ascii=False)
    )
    rerun_protocol_notebook = make_rerun_protocol_notebook()
    (ROOT / "sakana_medframeqa_rerun_protocol.ipynb").write_text(
        json.dumps(rerun_protocol_notebook, indent=2, ensure_ascii=False)
    )
    (ROOT / "sakana_medframeqa_paper_pipeline.ipynb").write_text(
        json.dumps(rerun_protocol_notebook, indent=2, ensure_ascii=False)
    )


def main():
    ensure_split_manifest(ROOT / "medframeqa_split_manifest_v1.json")
    task_sources = materialize_tasks()
    materialize_notebooks(task_sources)
    print("Generated task files, manifest, and notebooks.")


if __name__ == "__main__":
    main()
