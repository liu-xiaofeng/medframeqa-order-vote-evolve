"""
MedFrameQA 共享运行时工具。

这个文件负责 5 类共用能力：
1. 数据集与 frozen split manifest 的加载。
2. 多图样本的图像压缩、消息构造、选项字母解析。
3. 单一 8001 VLM 服务下的全局串行锁，避免多个实验同时打同一个服务。
4. 统一的实验指标字段、结果汇总、异常兜底。
5. notebook 与 evaluate.py 共用的小工具，尽量把脆弱逻辑收口到这里。

输入：
- MedFrameQA 数据集
- 每条任务线生成的 initial.py / evaluate.py
- notebook 的 sanity / smoke / paper eval 请求

输出：
- 规范化后的模型消息
- 统一 schema 的 metrics
- VLM 锁状态与运行控制信息

失败处理：
- 如果变异程序返回了非法配置，会尽量回退到默认值而不是直接崩。
- 如果请求 8001 失败，会按统一重试策略重试。
- 如果实验内部抛异常，会转成 invalid_generation 指标，而不是让整代静默失败。
"""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import signal
import string
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from PIL import Image
from datasets import load_dataset
from openai import OpenAI


DEFAULT_PROJECT_ROOT = Path("/gluon4/xl693/evolve")
DEFAULT_DATASET_ID = os.environ.get("MEDFRAMEQA_DATASET", "SuhaoYu1020/MedFrameQA")
DEFAULT_DATASET_CACHE_DIR = os.environ.get(
    "MEDFRAMEQA_DATASET_CACHE",
    "/tmp/medframeqa_hf_cache",
)
DEFAULT_SPLIT_MANIFEST = os.environ.get(
    "MEDFRAMEQA_SPLIT_MANIFEST",
    str(DEFAULT_PROJECT_ROOT / "medframeqa_split_manifest_v1.json"),
)
DEFAULT_IMAGE_MAX_SIDE = int(os.environ.get("MEDFRAMEQA_IMAGE_MAX_SIDE", "384"))
DEFAULT_IMAGE_QUALITY = int(os.environ.get("MEDFRAMEQA_IMAGE_QUALITY", "60"))
DEFAULT_MAX_IMAGES = int(os.environ.get("MEDFRAMEQA_MAX_IMAGES", "0"))
DEFAULT_API_TIMEOUT = float(os.environ.get("MEDFRAMEQA_API_TIMEOUT", "30"))
DEFAULT_REQUEST_RETRIES = int(os.environ.get("MEDFRAMEQA_REQUEST_RETRIES", "3"))
DEFAULT_REQUEST_RETRY_SLEEP = float(
    os.environ.get("MEDFRAMEQA_REQUEST_RETRY_SLEEP", "2.0")
)
DEFAULT_VLM_LOCK_PATH = Path(
    os.environ.get("MEDFRAMEQA_VLM_LOCK_PATH", "/tmp/medframeqa_vlm_8001.lock")
)
DEFAULT_VLM_LOCK_POLL_SECONDS = float(
    os.environ.get("MEDFRAMEQA_VLM_LOCK_POLL_SECONDS", "5")
)
DEFAULT_VLM_LOCK_STALE_SECONDS = float(
    os.environ.get("MEDFRAMEQA_VLM_LOCK_STALE_SECONDS", "1800")
)
RESAMPLING = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
IMAGE_FRAME_PATTERN = re.compile(r"^image_(\d+)$")

PROTOCOL_ALIAS = {
    "search_mini": "search",
    "selection_holdout": "holdout",
    "independent_final_test": "final_test",
    "evolution_pool": "evolution_pool",
}


def stable_hash(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def resolve_project_root(start_path=None):
    if start_path is None:
        path = Path.cwd()
    else:
        path = Path(start_path).resolve()

    candidates = [path] if path.is_dir() else [path.parent]
    candidates.extend(path.parents)
    for candidate in candidates:
        if (candidate / "medframeqa_runtime.py").exists():
            return candidate
        if (candidate / "build_medframeqa_notebooks.py").exists():
            return candidate
    return DEFAULT_PROJECT_ROOT


def ensure_project_root_on_path(start_path=None):
    root = resolve_project_root(start_path)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_mutated_module(path):
    spec = importlib.util.spec_from_file_location("mutated_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merge_text_config(defaults, candidate):
    """只把白名单里的字符串/嵌套字符串字段从 candidate 投影回默认配置。"""
    merged = json.loads(json.dumps(defaults))
    if not isinstance(candidate, dict):
        return merged

    for key, fallback in defaults.items():
        value = candidate.get(key)
        if isinstance(fallback, dict):
            if isinstance(value, dict):
                for subkey, subfallback in fallback.items():
                    subvalue = value.get(subkey)
                    if isinstance(subfallback, str):
                        if isinstance(subvalue, str) and subvalue:
                            merged[key][subkey] = subvalue
                    else:
                        merged[key][subkey] = subfallback
        elif isinstance(fallback, str):
            if isinstance(value, str) and value:
                merged[key] = value
    return merged


def parse_json_config_block(raw_text):
    """
    把可进化 JSON 文本解析成 dict。
    解析失败时返回空 dict，这样上层会自动回退到默认配置。
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {}
    try:
        value = json.loads(raw_text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def coerce_choice(value, allowed_values, default_value):
    return value if value in allowed_values else default_value


def coerce_int_choice(value, allowed_values, default_value):
    return value if value in allowed_values else default_value


def prepare_image(image):
    rgb = image.convert("RGB")
    max_side = max(
        128,
        int(os.environ.get("MEDFRAMEQA_IMAGE_MAX_SIDE", str(DEFAULT_IMAGE_MAX_SIDE))),
    )
    quality = max(
        20,
        min(
            95,
            int(
                os.environ.get(
                    "MEDFRAMEQA_IMAGE_QUALITY",
                    str(DEFAULT_IMAGE_QUALITY),
                )
            ),
        ),
    )

    if max(rgb.size) > max_side:
        scale = max_side / max(rgb.size)
        new_size = (
            max(1, int(round(rgb.size[0] * scale))),
            max(1, int(round(rgb.size[1] * scale))),
        )
        rgb = rgb.resize(new_size, RESAMPLING)
    return rgb, quality


def encode_image(image):
    if not isinstance(image, Image.Image):
        return None
    rgb, quality = prepare_image(image)
    buffered = io.BytesIO()
    rgb.save(buffered, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_frame_image_column(column_name):
    """
    只把真正的逐帧图像列当成图片输入。

    MedFrameQA 的样本里同时存在:
    - image_url: 原始 URL / 辅助字符串
    - image_1 ... image_5: 真正的图像帧

    这里必须显式排除 image_url，否则会把题目的图像数错误地整体 +1，
    进而污染 prompt 元信息、图像编号和按 frame count 的统计分析。
    """
    return bool(IMAGE_FRAME_PATTERN.match(str(column_name)))


def image_column_sort_key(column_name):
    match = IMAGE_FRAME_PATTERN.match(str(column_name))
    return int(match.group(1)) if match else math.inf


def get_image_columns(sample):
    image_columns = [
        key
        for key in sample.keys()
        if is_frame_image_column(key) and sample[key]
    ]
    image_columns.sort(key=image_column_sort_key)
    max_images = int(os.environ.get("MEDFRAMEQA_MAX_IMAGES", str(DEFAULT_MAX_IMAGES)))
    if max_images > 0:
        image_columns = image_columns[:max_images]
    return image_columns


def format_options(options, option_indices=None, letters=None):
    if option_indices is None:
        option_indices = list(range(len(options)))
    labels = letters or string.ascii_uppercase
    return "\n".join(
        f"{labels[local_idx]}. {options[option_idx]}"
        for local_idx, option_idx in enumerate(option_indices)
    )


def infer_case_metadata(sample):
    combined_text = " ".join(
        [
            sample.get("system", ""),
            sample.get("organ", ""),
            sample.get("keyword", ""),
            sample.get("modality", ""),
            sample.get("question", ""),
            *sample.get("options", []),
        ]
    ).lower()
    image_columns = get_image_columns(sample)

    sequence_tokens = [
        "before",
        "after",
        "follow-up",
        "progression",
        "sequence",
        "change",
        "stable",
        "improved",
        "worse",
        "postoperative",
    ]

    present_sequence_tokens = [token for token in sequence_tokens if token in combined_text]
    return {
        "image_count": len(image_columns),
        "system": sample.get("system", "unspecified"),
        "organ": sample.get("organ", "unspecified"),
        "keyword": sample.get("keyword", "unspecified"),
        "modality": sample.get("modality", "unspecified"),
        "video_id": sample.get("video_id", "unknown"),
        "sequence_hints": present_sequence_tokens[:5] or ["none"],
    }


def render_metadata_block(metadata):
    return (
        "Structured case metadata:\n"
        f"- System: {metadata['system']}\n"
        f"- Organ: {metadata['organ']}\n"
        f"- Keyword: {metadata['keyword']}\n"
        f"- Modality: {metadata['modality']}\n"
        f"- Video ID: {metadata['video_id']}\n"
        f"- Ordered image count: {metadata['image_count']}\n"
        f"- Sequence hints: {', '.join(metadata['sequence_hints'])}\n"
    )


def build_content_list(sample, prompt_text, image_prompt_template):
    content = [{"type": "text", "text": prompt_text}]
    image_columns = get_image_columns(sample)
    for index, column in enumerate(image_columns, 1):
        encoded = encode_image(sample[column])
        if encoded:
            if image_prompt_template:
                content.append(
                    {
                        "type": "text",
                        "text": "\n" + image_prompt_template.format(index=index, total=len(image_columns)),
                    }
                )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
    return content, image_columns


def get_manifest_path(manifest_path=None):
    if manifest_path:
        return Path(manifest_path)
    return Path(os.environ.get("MEDFRAMEQA_SPLIT_MANIFEST", DEFAULT_SPLIT_MANIFEST))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_medframeqa_dataset(include_images=True):
    dataset = load_dataset(
        DEFAULT_DATASET_ID,
        split="test",
        cache_dir=DEFAULT_DATASET_CACHE_DIR,
    )
    if not include_images:
        drop_columns = [column for column in dataset.column_names if column.startswith("image_")]
        dataset = dataset.remove_columns(drop_columns)
    return dataset


def load_split_manifest(manifest_path=None):
    path = get_manifest_path(manifest_path)
    if not path.exists():
        ensure_split_manifest(path)
    else:
        ensure_project_root_on_path(__file__)
        from create_medframeqa_split_manifest import GENERATOR_STRATEGY, MANIFEST_VERSION

        try:
            payload = json.loads(path.read_text())
        except Exception:
            ensure_split_manifest(path)
        else:
            generator = payload.get("generator", {})
            if payload.get("version") != MANIFEST_VERSION or generator.get("strategy") != GENERATOR_STRATEGY:
                ensure_split_manifest(path)
    return json.loads(path.read_text())


def ensure_split_manifest(manifest_path=None):
    path = get_manifest_path(manifest_path)
    ensure_project_root_on_path(__file__)
    from create_medframeqa_split_manifest import (
        GENERATOR_STRATEGY,
        MANIFEST_VERSION,
        ensure_split_manifest as _ensure,
    )

    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = None
        if payload:
            generator = payload.get("generator", {})
            if payload.get("version") == MANIFEST_VERSION and generator.get("strategy") == GENERATOR_STRATEGY:
                return path

    return Path(_ensure(output_path=path))


def build_question_index(dataset):
    return {question_id: index for index, question_id in enumerate(dataset["question_id"])}


def select_stratified_search_mini(question_ids, meta_by_qid, size):
    if size <= 0 or size >= len(question_ids):
        return list(question_ids)

    strata = defaultdict(list)
    for question_id in question_ids:
        answer, modality = meta_by_qid[question_id]
        strata[(answer, modality)].append(question_id)

    sorted_items = sorted(strata.items(), key=lambda item: stable_hash(f"{item[0][0]}::{item[0][1]}"))
    quotas = {}
    remainders = []
    allocated = 0
    total = len(question_ids)

    for key, members in sorted_items:
        members.sort(key=stable_hash)
        ideal = size * len(members) / total
        quota = min(len(members), int(math.floor(ideal)))
        quotas[key] = quota
        allocated += quota
        remainders.append((ideal - quota, stable_hash(str(key)), key))

    remaining = size - allocated
    for _, _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            remaining -= 1

    if remaining > 0:
        leftovers = []
        for key, members in sorted_items:
            leftovers.extend(members[quotas[key] :])
        for question_id in sorted(leftovers, key=stable_hash)[:remaining]:
            answer, modality = meta_by_qid[question_id]
            quotas[(answer, modality)] += 1

    selected = []
    for key, members in sorted_items:
        selected.extend(members[: quotas[key]])
    return sorted(selected, key=stable_hash)


def get_protocol_subset(dataset, manifest, protocol_mode, search_mini_size):
    question_index = build_question_index(dataset)
    meta_by_qid = {
        qid: (answer, modality)
        for qid, answer, modality in zip(
            dataset["question_id"],
            dataset["correct_answer"],
            dataset["modality"],
        )
    }

    if protocol_mode == "search_mini":
        pool_ids = manifest["splits"]["evolution_pool"]
        selected_ids = select_stratified_search_mini(pool_ids, meta_by_qid, search_mini_size)
    else:
        selected_ids = manifest["splits"][protocol_mode]

    indices = [question_index[question_id] for question_id in selected_ids]
    subset = dataset.select(indices)
    return subset, selected_ids


def get_protocol_meta(manifest, protocol_mode, selected_ids, search_mini_size):
    return {
        "protocol_mode": protocol_mode,
        "split_manifest_version": manifest["version"],
        "selected_subset_size": len(selected_ids),
        "search_mini_size": search_mini_size,
        "selected_subset_preview": selected_ids[:5],
    }


def protocol_alias(protocol_mode):
    return PROTOCOL_ALIAS.get(protocol_mode, protocol_mode)


def protocol_metric_prefix(protocol_mode):
    return protocol_alias(protocol_mode)


def make_protocol_metrics(protocol_mode, correct, total):
    prefix = protocol_metric_prefix(protocol_mode)
    return {
        f"{prefix}_score": correct / total if total else 0.0,
        f"{prefix}_correct": correct,
        f"{prefix}_size": total,
    }


def get_protocol_score_key(protocol_mode):
    return f"{protocol_metric_prefix(protocol_mode)}_score"


def get_protocol_size_key(protocol_mode):
    return f"{protocol_metric_prefix(protocol_mode)}_size"


def normalize_messages(payload):
    if not isinstance(payload, list) or not payload:
        raise TypeError("format_vqa_payload must return a non-empty list")
    if isinstance(payload[0], dict) and "role" in payload[0]:
        return payload
    return [{"role": "user", "content": payload}]


def extract_option_letter(text, valid_letters):
    raw = (text or "").strip().upper()
    if raw in valid_letters:
        return raw
    for letter in valid_letters:
        if re.search(rf"\b{re.escape(letter)}\b", raw):
            return letter
    if raw and raw[0] in valid_letters:
        return raw[0]
    return ""


def get_image_budget_schedule():
    schedule = [
        (DEFAULT_IMAGE_MAX_SIDE, DEFAULT_IMAGE_QUALITY),
        (320, 50),
        (256, 40),
    ]
    ordered = []
    seen = set()
    for item in schedule:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def is_context_length_error(exc):
    message = str(exc).lower()
    return "maximum context length" in message or "input length" in message


def set_image_budget(max_side, quality):
    os.environ["MEDFRAMEQA_IMAGE_MAX_SIDE"] = str(max_side)
    os.environ["MEDFRAMEQA_IMAGE_QUALITY"] = str(quality)


def is_retryable_request_error(exc):
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    retryable_names = (
        "apiconnectionerror",
        "apitimeouterror",
        "connecterror",
        "connecttimeout",
        "readtimeout",
        "pooltimeout",
        "remoteprotocolerror",
        "serverdisconnectederror",
        "readerror",
    )
    retryable_message_parts = (
        "connection error",
        "connection refused",
        "connection reset",
        "server disconnected",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "502",
        "503",
        "504",
    )
    return any(token in name for token in retryable_names) or any(
        token in message for token in retryable_message_parts
    )


def make_openai_client(base_url, timeout=None):
    return OpenAI(
        api_key="local",
        base_url=base_url,
        timeout=timeout if timeout is not None else DEFAULT_API_TIMEOUT,
    )


def _lock_meta_path(lock_path):
    return Path(str(lock_path) + ".json")


def _pid_exists(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def read_vlm_lock_info(lock_path=None):
    lock_path = Path(lock_path or DEFAULT_VLM_LOCK_PATH)
    meta_path = _lock_meta_path(lock_path)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _write_vlm_lock_info(lock_path, info):
    meta_path = _lock_meta_path(lock_path)
    meta_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


def _remove_vlm_lock_files(lock_path):
    lock_path = Path(lock_path)
    meta_path = _lock_meta_path(lock_path)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    try:
        meta_path.unlink()
    except FileNotFoundError:
        pass


def _is_vlm_lock_stale(lock_path, stale_seconds):
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return False
    info = read_vlm_lock_info(lock_path)
    pid = info.get("pid")
    if pid is not None and not _pid_exists(pid):
        return True
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    if age > stale_seconds and pid is not None and not _pid_exists(pid):
        return True
    return False


@contextmanager
def acquire_vlm_lock(task_name, results_dir=None, mode="eval", lock_path=None):
    """
    在单个 8001 VLM 服务前面加一把全局文件锁。

    设计目标：
    - notebook / smoke / evaluate / paper eval 全部共用同一把锁；
    - 后来的实验不假死，而是明确显示自己在等待谁；
    - 旧 owner 已死时，锁可以自动回收。
    """
    lock_path = Path(lock_path or DEFAULT_VLM_LOCK_PATH)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    poll_seconds = DEFAULT_VLM_LOCK_POLL_SECONDS
    stale_seconds = DEFAULT_VLM_LOCK_STALE_SECONDS
    token = uuid.uuid4().hex
    info = {
        "token": token,
        "pid": os.getpid(),
        "task_name": task_name,
        "results_dir": str(results_dir) if results_dir else "",
        "mode": mode,
        "started_at": time.time(),
    }
    wait_started = time.time()
    last_report = 0.0

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(token + "\n")
            _write_vlm_lock_info(lock_path, info)
            break
        except FileExistsError:
            if _is_vlm_lock_stale(lock_path, stale_seconds):
                print(f"[VLM LOCK] 清理陈旧锁: {lock_path}", file=sys.stderr)
                _remove_vlm_lock_files(lock_path)
                continue

            owner = read_vlm_lock_info(lock_path)
            now = time.time()
            if now - last_report >= poll_seconds:
                owner_task = owner.get("task_name", "unknown")
                owner_mode = owner.get("mode", "unknown")
                owner_pid = owner.get("pid", "unknown")
                owner_dir = owner.get("results_dir", "")
                print(
                    f"[VLM LOCK] 等待 8001: owner_task={owner_task} owner_mode={owner_mode} "
                    f"owner_pid={owner_pid} owner_results={owner_dir}",
                    file=sys.stderr,
                )
                last_report = now
            time.sleep(poll_seconds)

    wait_seconds = round(time.time() - wait_started, 3)
    info["waited_for_lock_sec"] = wait_seconds
    _write_vlm_lock_info(lock_path, info)
    try:
        yield info
    finally:
        current = read_vlm_lock_info(lock_path)
        if current.get("token") == token:
            _remove_vlm_lock_files(lock_path)


def generate_guided_choice(
    client,
    messages,
    valid_choices,
    model,
    call_stats=None,
):
    if call_stats is not None:
        call_stats["vlm_call_count"] = call_stats.get("vlm_call_count", 0) + 1

    extra_body = {"guided_choice": valid_choices}
    if messages[-1]["role"] == "assistant":
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=4,
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


def generate_guided_choice_with_retries(
    client,
    message_builder,
    valid_choices,
    model,
    debug_label,
    call_stats=None,
    request_retries=None,
    request_retry_sleep=None,
):
    request_retries = (
        DEFAULT_REQUEST_RETRIES if request_retries is None else request_retries
    )
    request_retry_sleep = (
        DEFAULT_REQUEST_RETRY_SLEEP
        if request_retry_sleep is None
        else request_retry_sleep
    )
    last_exc = None

    for max_side, quality in get_image_budget_schedule():
        set_image_budget(max_side, quality)
        attempt = 0
        while True:
            try:
                messages = normalize_messages(message_builder())
                return generate_guided_choice(
                    client,
                    messages,
                    valid_choices,
                    model,
                    call_stats=call_stats,
                )
            except Exception as exc:
                last_exc = exc
                if is_context_length_error(exc):
                    print(
                        f"{debug_label} retry image_max_side={max_side} "
                        f"image_quality={quality} error={exc}",
                        file=sys.stderr,
                    )
                    break
                if is_retryable_request_error(exc) and attempt < request_retries:
                    attempt += 1
                    sleep_s = request_retry_sleep * attempt
                    print(
                        f"{debug_label} request_retry={attempt}/{request_retries} "
                        f"sleep={sleep_s:.1f}s error={exc}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
    raise last_exc


def get_option_orders(num_options, order_views):
    base = list(range(num_options))
    candidates = [
        ("identity", base),
        ("reverse", list(reversed(base))),
        ("rotate_left", base[1:] + base[:1] if num_options > 1 else base),
        ("outside_in", base[::2] + base[1::2] if num_options > 2 else base),
    ]
    orders = []
    seen = set()
    for order_name, order in candidates:
        key = tuple(order)
        if key not in seen:
            orders.append((order_name, order))
            seen.add(key)
        if len(orders) >= order_views:
            break
    return orders


def local_to_global(local_choice, option_indices):
    if not isinstance(local_choice, str) or len(local_choice) != 1:
        return ""
    if local_choice not in string.ascii_uppercase:
        return ""
    local_index = ord(local_choice) - ord("A")
    if local_index < 0 or local_index >= len(option_indices):
        return ""
    return chr(ord("A") + option_indices[local_index])


def deterministic_letter_fallback(valid_letters, tie_break):
    if not valid_letters:
        return ""
    if tie_break == "reverse_alphabetical":
        return sorted(valid_letters, reverse=True)[0]
    return sorted(valid_letters)[0]


def parse_generation_index(path_value):
    if not path_value:
        return None
    match = re.search(r"gen_(\d+)", str(path_value))
    if match:
        return int(match.group(1))
    return None


def parse_generation_set(raw_value):
    if not raw_value:
        return set()
    generations = set()
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.isdigit():
            generations.add(int(chunk))
    return generations


def make_invalid_metrics(protocol_mode, search_mini_size, error_type, error_message):
    metrics = {
        "search_score": 0.0,
        "search_correct": 0,
        "search_size": 0,
        "holdout_score": 0.0,
        "holdout_correct": 0,
        "holdout_size": 0,
        "combined_score": 0.0,
        "invalid_generation": 1,
        "error_type": error_type,
        "error_message": error_message,
        "search_mini_size": search_mini_size,
        "vlm_call_count": 0,
        "protocol_mode": protocol_mode,
        "split_manifest_version": "",
    }
    active_prefix = protocol_metric_prefix(protocol_mode)
    metrics[f"{active_prefix}_score"] = 0.0
    metrics[f"{active_prefix}_correct"] = 0
    metrics[f"{active_prefix}_size"] = 0
    return metrics


def safe_run_experiment(
    experiment_fn,
    protocol_mode,
    search_mini_size,
    task_name="unknown_task",
    results_dir=None,
    lock_mode=None,
):
    start_time = time.time()
    lock_info = {}
    try:
        with acquire_vlm_lock(
            task_name=task_name,
            results_dir=results_dir,
            mode=lock_mode or protocol_mode,
        ) as owner_info:
            lock_info = owner_info
            metrics = experiment_fn()
        metrics.setdefault("invalid_generation", 0)
        metrics.setdefault("error_type", "")
        metrics.setdefault("error_message", "")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        metrics = make_invalid_metrics(
            protocol_mode=protocol_mode,
            search_mini_size=search_mini_size,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    metrics["wall_time_sec"] = round(time.time() - start_time, 3)
    metrics["vlm_lock_wait_sec"] = round(lock_info.get("waited_for_lock_sec", 0.0), 3)
    return metrics


def collect_generation_records(run_dir):
    run_dir = Path(run_dir)
    records = []
    for generation_dir in sorted(run_dir.glob("gen_*"), key=lambda path: parse_generation_index(path) or -1):
        metrics_path = generation_dir / "results" / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception as exc:
                metrics = {
                    "combined_score": 0.0,
                    "invalid_generation": 1,
                    "error_type": "UnreadableMetrics",
                    "error_message": str(exc),
                    "metrics_missing": 1,
                    "incomplete_generation": 1,
                }
        else:
            metrics = {
                "combined_score": 0.0,
                "invalid_generation": 1,
                "error_type": "MissingMetrics",
                "error_message": "metrics.json missing",
                "metrics_missing": 1,
                "incomplete_generation": 1,
            }
        records.append(
            {
                "generation": parse_generation_index(generation_dir),
                "generation_dir": str(generation_dir),
                "program_path": str(generation_dir / "main.py"),
                "metrics_path": str(metrics_path),
                **metrics,
            }
        )
    return records


def select_top_k_records(records, top_k=3):
    valid_records = [record for record in records if not record.get("invalid_generation")]
    ordered = sorted(
        valid_records,
        key=lambda record: (
            -record.get("combined_score", 0.0),
            record.get("generation", 10**9),
        ),
    )
    return ordered[:top_k]


def select_best_so_far(records, milestone_generation):
    eligible = [
        record
        for record in records
        if record.get("generation") is not None
        and record["generation"] <= milestone_generation
        and not record.get("invalid_generation")
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda record: (
            -record.get("combined_score", 0.0),
            record.get("generation", 10**9),
        ),
    )[0]
