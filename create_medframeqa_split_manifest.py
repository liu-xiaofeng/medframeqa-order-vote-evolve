import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path("/gluon4/xl693/evolve")
DATASET_ID = os.environ.get("MEDFRAMEQA_DATASET", "SuhaoYu1020/MedFrameQA")
SOURCE_SPLIT = "test"
GROUP_KEY = "video_id"
MANIFEST_VERSION = "medframeqa_split_manifest_v2"
OUTPUT_PATH = PROJECT_ROOT / "medframeqa_split_manifest_v1.json"
DATASET_CACHE_DIR = os.environ.get("MEDFRAMEQA_DATASET_CACHE", "/tmp/medframeqa_hf_cache")

TARGET_SPLITS = {
    "evolution_pool": 1331,
    "selection_holdout": 665,
    "independent_final_test": 855,
}
SPLIT_NAMES = list(TARGET_SPLITS.keys())

MANDATORY_MODALITIES = ("CT", "MRI", "ultrasound", "X-ray")
HOLDOUT_OTHER_REQUIRED = ("selection_holdout", "independent_final_test")

OBJECTIVE_WEIGHTS = {
    "modality": 4.0,
    "answer": 1.5,
    "organ": 0.25,
}
HARD_PENALTIES = {
    "missing_primary_modality": 100.0,
    "missing_other": 50.0,
}

RESTART_COUNT = int(os.environ.get("MEDFRAMEQA_SPLIT_RESTARTS", "128"))
LOCAL_SEARCH_MAX_STEPS = int(os.environ.get("MEDFRAMEQA_SPLIT_LOCAL_STEPS", "24"))
PAIR_SIZE_LIMIT = int(os.environ.get("MEDFRAMEQA_SPLIT_BUNDLE_CAP", "12"))
GROUP_SHORTLIST = int(os.environ.get("MEDFRAMEQA_SPLIT_GROUP_SHORTLIST", "24"))
BASE_SEED = int(os.environ.get("MEDFRAMEQA_SPLIT_BASE_SEED", "135"))
GENERATOR_STRATEGY = "video_group_multistart_bundle_exchange_v2"


def _is_frame_image_column(column_name: str) -> bool:
    if not column_name.startswith("image_"):
        return False
    suffix = column_name.split("_", 1)[1]
    return suffix.isdigit()


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    question_ids: tuple
    size: int
    modality: tuple
    answer: tuple
    organ: tuple


@dataclass(frozen=True)
class BundleRecord:
    members: tuple
    size: int
    modality: tuple
    answer: tuple
    organ: tuple


def _stable_hash(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _load_rows():
    dataset = load_dataset(
        DATASET_ID,
        split=SOURCE_SPLIT,
        cache_dir=DATASET_CACHE_DIR,
    )
    image_columns = [
        column
        for column in dataset.column_names
        if column == "image_url" or _is_frame_image_column(column)
    ]
    if image_columns:
        dataset = dataset.remove_columns(image_columns)
    return [dict(row) for row in dataset]


def _build_key_spaces(rows):
    modality_keys = sorted(Counter(row["modality"] for row in rows))
    answer_keys = sorted(Counter(row["correct_answer"] for row in rows))
    organ_counts = Counter(row["organ"] for row in rows)
    organ_keys = [organ for organ, _ in organ_counts.most_common(10)]
    return modality_keys, answer_keys, organ_keys


def _group_rows(rows, modality_keys, answer_keys, organ_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[GROUP_KEY]].append(row)

    groups = []
    for group_id, members in grouped.items():
        question_ids = tuple(row["question_id"] for row in members)
        modality_counter = Counter(row["modality"] for row in members)
        answer_counter = Counter(row["correct_answer"] for row in members)
        organ_counter = Counter(row["organ"] for row in members)
        groups.append(
            GroupRecord(
                group_id=str(group_id),
                question_ids=question_ids,
                size=len(members),
                modality=tuple(modality_counter.get(key, 0) for key in modality_keys),
                answer=tuple(answer_counter.get(key, 0) for key in answer_keys),
                organ=tuple(organ_counter.get(key, 0) for key in organ_keys),
            )
        )
    groups.sort(key=lambda group: (_stable_hash(group.group_id), group.group_id))
    return groups


def _sum_vectors(groups, attribute, count):
    total = [0] * count
    for group in groups:
        vector = getattr(group, attribute)
        for index, value in enumerate(vector):
            total[index] += value
    return total


def _vector_from_counter(counter, keys):
    return [counter.get(key, 0) for key in keys]


def _expected_counts(total_vector, target_size, total_size):
    return [target_size * value / total_size for value in total_vector]


def _relative_deviation(actual, expected):
    deviations = []
    for actual_value, expected_value in zip(actual, expected):
        baseline = max(expected_value, 1e-9)
        deviations.append(abs(actual_value - expected_value) / baseline)
    return deviations


def _objective_from_vectors(split_name, modality, answer, organ, expected):
    modality_dev = _relative_deviation(modality, expected[split_name]["modality"])
    answer_dev = _relative_deviation(answer, expected[split_name]["answer"])
    organ_dev = _relative_deviation(organ, expected[split_name]["organ"])

    objective = 0.0
    objective += OBJECTIVE_WEIGHTS["modality"] * sum(modality_dev)
    objective += OBJECTIVE_WEIGHTS["answer"] * sum(answer_dev)
    objective += OBJECTIVE_WEIGHTS["organ"] * sum(organ_dev)

    modality_keys = expected["_modality_keys"]
    modality_index = {key: index for index, key in enumerate(modality_keys)}
    for key in MANDATORY_MODALITIES:
        index = modality_index.get(key)
        if index is not None and modality[index] <= 0:
            objective += HARD_PENALTIES["missing_primary_modality"]

    other_index = modality_index.get("other")
    if split_name in HOLDOUT_OTHER_REQUIRED and other_index is not None and modality[other_index] <= 0:
        objective += HARD_PENALTIES["missing_other"]
    return objective


def _assignment_objective(assignment, groups_by_id, expected):
    score = 0.0
    for split_name, group_ids in assignment.items():
        groups = [groups_by_id[group_id] for group_id in group_ids]
        modality = _sum_vectors(groups, "modality", len(expected["_modality_keys"]))
        answer = _sum_vectors(groups, "answer", len(expected["_answer_keys"]))
        organ = _sum_vectors(groups, "organ", len(expected["_organ_keys"]))
        score += _objective_from_vectors(split_name, modality, answer, organ, expected)
    return score


def _split_distribution_score(group, modality_target, answer_target, organ_target, seed_value):
    size = max(group.size, 1)
    modality_prop = [value / size for value in group.modality]
    answer_prop = [value / size for value in group.answer]
    organ_prop = [value / size for value in group.organ]

    modality_delta = sum(
        abs(modality_prop[index] - modality_target[index]) for index in range(len(modality_prop))
    )
    answer_delta = sum(abs(answer_prop[index] - answer_target[index]) for index in range(len(answer_prop)))
    organ_delta = sum(abs(organ_prop[index] - organ_target[index]) for index in range(len(organ_prop)))

    score = 0.0
    score += OBJECTIVE_WEIGHTS["modality"] * (2.0 - modality_delta)
    score += OBJECTIVE_WEIGHTS["answer"] * (2.0 - answer_delta)
    score += OBJECTIVE_WEIGHTS["organ"] * (2.0 - organ_delta)
    score *= size
    score += ((_stable_hash(f"{seed_value}:{group.group_id}") % 1_000_003) / 1_000_003.0) * 0.01
    return score


def _exact_knapsack_select(groups, target_size, values):
    neg_inf = -10**30
    dp = [(neg_inf, None)] * (target_size + 1)
    dp[0] = (0.0, tuple())

    for index, group in enumerate(groups):
        size = group.size
        value = values[index]
        next_dp = list(dp)
        for total in range(target_size, size - 1, -1):
            previous_score, previous_choice = dp[total - size]
            if previous_score <= neg_inf / 2:
                continue
            candidate = previous_score + value
            if candidate > next_dp[total][0]:
                next_dp[total] = (candidate, previous_choice + (index,))
        dp = next_dp

    score, chosen = dp[target_size]
    if chosen is None:
        raise ValueError(f"Could not solve exact knapsack for target size {target_size}")
    return set(chosen)


def _build_initial_assignment(groups, expected, seed_value):
    modality_target = {
        split_name: [
            value / TARGET_SPLITS[split_name] if TARGET_SPLITS[split_name] else 0.0
            for value in expected[split_name]["modality"]
        ]
        for split_name in SPLIT_NAMES
    }
    answer_target = {
        split_name: [
            value / TARGET_SPLITS[split_name] if TARGET_SPLITS[split_name] else 0.0
            for value in expected[split_name]["answer"]
        ]
        for split_name in SPLIT_NAMES
    }
    organ_target = {
        split_name: [
            value / TARGET_SPLITS[split_name] if TARGET_SPLITS[split_name] else 0.0
            for value in expected[split_name]["organ"]
        ]
        for split_name in SPLIT_NAMES
    }

    split_order = ["selection_holdout", "independent_final_test"]
    if seed_value % 2:
        split_order = ["independent_final_test", "selection_holdout"]

    for attempt in range(64):
        rng = random.Random(seed_value * 1009 + attempt)
        remaining = list(groups)
        rng.shuffle(remaining)
        assignment = {}

        try:
            for split_name in split_order:
                values = [
                    _split_distribution_score(
                        group,
                        modality_target[split_name],
                        answer_target[split_name],
                        organ_target[split_name],
                        f"{seed_value}:{attempt}:{split_name}",
                    )
                    for group in remaining
                ]
                chosen_indices = _exact_knapsack_select(remaining, TARGET_SPLITS[split_name], values)
                chosen_groups = [group for index, group in enumerate(remaining) if index in chosen_indices]
                assignment[split_name] = tuple(group.group_id for group in chosen_groups)
                remaining = [group for index, group in enumerate(remaining) if index not in chosen_indices]

            assignment["evolution_pool"] = tuple(group.group_id for group in remaining)
            return {split_name: tuple(sorted(group_ids)) for split_name, group_ids in assignment.items()}
        except ValueError:
            continue

    raise ValueError(f"Could not build an exact initial assignment for seed {seed_value}")


def _vector_add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _vector_sub(left, right):
    return tuple(a - b for a, b in zip(left, right))


def _build_split_state(assignment, groups_by_id, expected):
    states = {}
    for split_name, group_ids in assignment.items():
        groups = [groups_by_id[group_id] for group_id in group_ids]
        modality = tuple(_sum_vectors(groups, "modality", len(expected["_modality_keys"])))
        answer = tuple(_sum_vectors(groups, "answer", len(expected["_answer_keys"])))
        organ = tuple(_sum_vectors(groups, "organ", len(expected["_organ_keys"])))
        states[split_name] = {
            "group_ids": list(group_ids),
            "group_set": set(group_ids),
            "size": sum(group.size for group in groups),
            "modality": modality,
            "answer": answer,
            "organ": organ,
            "objective": _objective_from_vectors(split_name, modality, answer, organ, expected),
        }
    return states


def _make_bundle(group_ids, groups_by_id):
    groups = [groups_by_id[group_id] for group_id in group_ids]
    return BundleRecord(
        members=tuple(sorted(group_ids)),
        size=sum(group.size for group in groups),
        modality=tuple(_sum_vectors(groups, "modality", len(groups[0].modality))),
        answer=tuple(_sum_vectors(groups, "answer", len(groups[0].answer))),
        organ=tuple(_sum_vectors(groups, "organ", len(groups[0].organ))),
    )


def _bundle_direction_score(bundle, source_state, target_state, source_name, target_name, expected):
    score = 0.0
    for weight_name, attribute in (("modality", "modality"), ("answer", "answer"), ("organ", "organ")):
        weight = OBJECTIVE_WEIGHTS[weight_name]
        bundle_vector = getattr(bundle, attribute)
        source_actual = source_state[attribute]
        target_actual = target_state[attribute]
        source_expected = expected[source_name][weight_name]
        target_expected = expected[target_name][weight_name]
        for index, value in enumerate(bundle_vector):
            if value <= 0:
                continue
            source_baseline = max(source_expected[index], 1e-9)
            target_baseline = max(target_expected[index], 1e-9)
            source_surplus = max(0.0, source_actual[index] - source_expected[index]) / source_baseline
            target_deficit = max(0.0, target_expected[index] - target_actual[index]) / target_baseline
            score += weight * value * (source_surplus + target_deficit)
    return score


def _candidate_bundles(group_ids, groups_by_id):
    indexed = defaultdict(list)
    sorted_ids = sorted(group_ids)
    for group_id in sorted_ids:
        bundle = _make_bundle((group_id,), groups_by_id)
        indexed[bundle.size].append(bundle)
    for left_index, left_group in enumerate(sorted_ids):
        left_size = groups_by_id[left_group].size
        for right_group in sorted_ids[left_index + 1 :]:
            bundle = _make_bundle((left_group, right_group), groups_by_id)
            indexed[bundle.size].append(bundle)
    return indexed


def _shortlist_group_ids(group_ids, groups_by_id, source_state, target_state, source_name, target_name, expected):
    scored = []
    for group_id in group_ids:
        bundle = _make_bundle((group_id,), groups_by_id)
        score = _bundle_direction_score(bundle, source_state, target_state, source_name, target_name, expected)
        scored.append((score, groups_by_id[group_id].size, group_id))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [group_id for _, _, group_id in scored[:GROUP_SHORTLIST]]


def _rank_bundles(bundle_index, source_state, target_state, source_name, target_name, expected, bundle_cap):
    ranked = {}
    for size, bundles in bundle_index.items():
        scored = [
            (
                _bundle_direction_score(bundle, source_state, target_state, source_name, target_name, expected),
                bundle,
            )
            for bundle in bundles
        ]
        scored.sort(key=lambda item: (-item[0], item[1].members))
        ranked[size] = [bundle for _, bundle in scored[:bundle_cap]]
    return ranked


def _apply_exchange(states, split_a, split_b, bundle_a, bundle_b, expected):
    state_a = states[split_a]
    state_b = states[split_b]

    new_modality_a = _vector_add(_vector_sub(state_a["modality"], bundle_a.modality), bundle_b.modality)
    new_answer_a = _vector_add(_vector_sub(state_a["answer"], bundle_a.answer), bundle_b.answer)
    new_organ_a = _vector_add(_vector_sub(state_a["organ"], bundle_a.organ), bundle_b.organ)

    new_modality_b = _vector_add(_vector_sub(state_b["modality"], bundle_b.modality), bundle_a.modality)
    new_answer_b = _vector_add(_vector_sub(state_b["answer"], bundle_b.answer), bundle_a.answer)
    new_organ_b = _vector_add(_vector_sub(state_b["organ"], bundle_b.organ), bundle_a.organ)

    state_a["modality"] = new_modality_a
    state_a["answer"] = new_answer_a
    state_a["organ"] = new_organ_a
    state_b["modality"] = new_modality_b
    state_b["answer"] = new_answer_b
    state_b["organ"] = new_organ_b
    state_a["objective"] = _objective_from_vectors(split_a, new_modality_a, new_answer_a, new_organ_a, expected)
    state_b["objective"] = _objective_from_vectors(split_b, new_modality_b, new_answer_b, new_organ_b, expected)

    for member in bundle_a.members:
        state_a["group_set"].remove(member)
        state_b["group_set"].add(member)
    for member in bundle_b.members:
        state_b["group_set"].remove(member)
        state_a["group_set"].add(member)
    state_a["group_ids"] = sorted(state_a["group_set"])
    state_b["group_ids"] = sorted(state_b["group_set"])


def _local_search(assignment, groups_by_id, expected):
    states = _build_split_state(assignment, groups_by_id, expected)
    for _ in range(LOCAL_SEARCH_MAX_STEPS):
        split_order = sorted(SPLIT_NAMES, key=lambda name: states[name]["objective"], reverse=True)
        best_move = None
        best_delta = 0.0

        for left_index, split_a in enumerate(split_order):
            for split_b in split_order[left_index + 1 :]:
                state_a = states[split_a]
                state_b = states[split_b]
                shortlist_a = _shortlist_group_ids(
                    state_a["group_ids"], groups_by_id, state_a, state_b, split_a, split_b, expected
                )
                shortlist_b = _shortlist_group_ids(
                    state_b["group_ids"], groups_by_id, state_b, state_a, split_b, split_a, expected
                )
                bundle_index_a = _candidate_bundles(shortlist_a, groups_by_id)
                bundle_index_b = _candidate_bundles(shortlist_b, groups_by_id)
                ranked_a = _rank_bundles(
                    bundle_index_a, state_a, state_b, split_a, split_b, expected, PAIR_SIZE_LIMIT
                )
                ranked_b = _rank_bundles(
                    bundle_index_b, state_b, state_a, split_b, split_a, expected, PAIR_SIZE_LIMIT
                )

                common_sizes = set(ranked_a).intersection(ranked_b)
                pair_before = state_a["objective"] + state_b["objective"]
                for size in sorted(common_sizes):
                    for bundle_a in ranked_a[size]:
                        for bundle_b in ranked_b[size]:
                            if set(bundle_a.members) & set(bundle_b.members):
                                continue
                            modality_a = _vector_add(_vector_sub(state_a["modality"], bundle_a.modality), bundle_b.modality)
                            answer_a = _vector_add(_vector_sub(state_a["answer"], bundle_a.answer), bundle_b.answer)
                            organ_a = _vector_add(_vector_sub(state_a["organ"], bundle_a.organ), bundle_b.organ)
                            modality_b = _vector_add(_vector_sub(state_b["modality"], bundle_b.modality), bundle_a.modality)
                            answer_b = _vector_add(_vector_sub(state_b["answer"], bundle_b.answer), bundle_a.answer)
                            organ_b = _vector_add(_vector_sub(state_b["organ"], bundle_b.organ), bundle_a.organ)
                            pair_after = _objective_from_vectors(
                                split_a, modality_a, answer_a, organ_a, expected
                            ) + _objective_from_vectors(split_b, modality_b, answer_b, organ_b, expected)
                            delta = pair_before - pair_after
                            if delta > best_delta + 1e-9:
                                best_delta = delta
                                best_move = (split_a, split_b, bundle_a, bundle_b)

        if not best_move:
            break
        _apply_exchange(states, *best_move, expected)

    return {split_name: tuple(states[split_name]["group_ids"]) for split_name in SPLIT_NAMES}


def _compute_expected(rows, modality_keys, answer_keys, organ_keys):
    total_rows = len(rows)
    modality_total = _vector_from_counter(Counter(row["modality"] for row in rows), modality_keys)
    answer_total = _vector_from_counter(Counter(row["correct_answer"] for row in rows), answer_keys)
    organ_total = _vector_from_counter(Counter(row["organ"] for row in rows), organ_keys)

    expected = {
        "_modality_keys": list(modality_keys),
        "_answer_keys": list(answer_keys),
        "_organ_keys": list(organ_keys),
    }
    for split_name, target_size in TARGET_SPLITS.items():
        expected[split_name] = {
            "modality": _expected_counts(modality_total, target_size, total_rows),
            "answer": _expected_counts(answer_total, target_size, total_rows),
            "organ": _expected_counts(organ_total, target_size, total_rows),
        }
    return expected


def _split_stats(split_name, group_ids, groups_by_id, expected):
    groups = [groups_by_id[group_id] for group_id in group_ids]
    size = sum(group.size for group in groups)
    modality_actual = _sum_vectors(groups, "modality", len(expected["_modality_keys"]))
    answer_actual = _sum_vectors(groups, "answer", len(expected["_answer_keys"]))
    organ_actual = _sum_vectors(groups, "organ", len(expected["_organ_keys"]))
    modality_expected = expected[split_name]["modality"]
    answer_expected = expected[split_name]["answer"]
    organ_expected = expected[split_name]["organ"]
    modality_deviation = _relative_deviation(modality_actual, modality_expected)
    answer_deviation = _relative_deviation(answer_actual, answer_expected)
    organ_deviation = _relative_deviation(organ_actual, organ_expected)

    modality_map = {key: modality_actual[index] for index, key in enumerate(expected["_modality_keys"])}
    answer_map = {key: answer_actual[index] for index, key in enumerate(expected["_answer_keys"])}
    organ_map = {key: organ_actual[index] for index, key in enumerate(expected["_organ_keys"])}
    expected_modality_map = {
        key: round(modality_expected[index], 3) for index, key in enumerate(expected["_modality_keys"])
    }
    expected_answer_map = {
        key: round(answer_expected[index], 3) for index, key in enumerate(expected["_answer_keys"])
    }
    expected_organ_map = {
        key: round(organ_expected[index], 3) for index, key in enumerate(expected["_organ_keys"])
    }
    modality_dev_map = {
        key: round(modality_deviation[index], 4) for index, key in enumerate(expected["_modality_keys"])
    }
    answer_dev_map = {
        key: round(answer_deviation[index], 4) for index, key in enumerate(expected["_answer_keys"])
    }
    organ_dev_map = {key: round(organ_deviation[index], 4) for index, key in enumerate(expected["_organ_keys"])}

    def _summary(values):
        return {
            "mean": round(sum(values) / len(values), 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
        }

    return {
        "size": size,
        "group_count": len(group_ids),
        "answer_distribution": answer_map,
        "modality_distribution": modality_map,
        "top_organs": organ_map,
        "expected_answer_distribution": expected_answer_map,
        "expected_modality_distribution": expected_modality_map,
        "expected_top_organs": expected_organ_map,
        "answer_relative_deviation": answer_dev_map,
        "modality_relative_deviation": modality_dev_map,
        "top_organ_relative_deviation": organ_dev_map,
        "relative_deviation_summary": {
            "answer": _summary(answer_deviation),
            "modality": _summary(modality_deviation),
            "organ": _summary(organ_deviation),
        },
        "objective_score": round(
            _objective_from_vectors(split_name, tuple(modality_actual), tuple(answer_actual), tuple(organ_actual), expected),
            6,
        ),
    }


def _validate_assignment(assignment, groups_by_id):
    all_question_ids = []
    seen_groups = set()
    for split_name, target_size in TARGET_SPLITS.items():
        group_ids = assignment[split_name]
        groups = [groups_by_id[group_id] for group_id in group_ids]
        split_size = sum(group.size for group in groups)
        if split_size != target_size:
            raise ValueError(f"{split_name} size mismatch: {split_size} != {target_size}")
        for group_id in group_ids:
            if group_id in seen_groups:
                raise ValueError(f"group leakage detected for {group_id}")
            seen_groups.add(group_id)
        for group in groups:
            all_question_ids.extend(group.question_ids)
    if len(all_question_ids) != len(set(all_question_ids)):
        raise ValueError("question_id overlap detected across splits")


def _manifest_payload(rows, groups, assignment, expected, seed_value, objective_score):
    groups_by_id = {group.group_id: group for group in groups}
    stats = {
        split_name: _split_stats(split_name, assignment[split_name], groups_by_id, expected)
        for split_name in SPLIT_NAMES
    }

    payload = {
        "version": MANIFEST_VERSION,
        "dataset_id": DATASET_ID,
        "source_split": SOURCE_SPLIT,
        "group_key": GROUP_KEY,
        "targets": TARGET_SPLITS,
        "generator": {
            "seed": seed_value,
            "base_seed": BASE_SEED,
            "strategy": GENERATOR_STRATEGY,
            "restart_count": RESTART_COUNT,
            "local_search_max_steps": LOCAL_SEARCH_MAX_STEPS,
            "group_shortlist": GROUP_SHORTLIST,
            "bundle_cap": PAIR_SIZE_LIMIT,
            "objective_weights": OBJECTIVE_WEIGHTS,
            "hard_penalties": HARD_PENALTIES,
            "bundle_exchange": ["1<->1", "1<->2", "2<->1", "2<->2"],
        },
        "total_examples": len(rows),
        "splits": {
            split_name: [
                question_id
                for group_id in assignment[split_name]
                for question_id in groups_by_id[group_id].question_ids
            ]
            for split_name in SPLIT_NAMES
        },
        "stats": stats,
        "objective_score": round(objective_score, 6),
    }
    manifest_text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    payload["manifest_sha256"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    return payload


def _run_search(rows):
    modality_keys, answer_keys, organ_keys = _build_key_spaces(rows)
    groups = _group_rows(rows, modality_keys, answer_keys, organ_keys)
    groups_by_id = {group.group_id: group for group in groups}
    expected = _compute_expected(rows, modality_keys, answer_keys, organ_keys)

    best_assignment = None
    best_score = math.inf
    best_seed = None

    for restart_index in range(RESTART_COUNT):
        seed_value = BASE_SEED + restart_index
        assignment = _build_initial_assignment(groups, expected, seed_value)
        assignment = _local_search(assignment, groups_by_id, expected)
        _validate_assignment(assignment, groups_by_id)
        score = _assignment_objective(assignment, groups_by_id, expected)
        if score < best_score:
            best_score = score
            best_assignment = assignment
            best_seed = seed_value

    return groups, best_assignment, expected, best_seed, best_score


def _needs_regeneration(path):
    if not path.exists():
        return True
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return True
    if payload.get("version") != MANIFEST_VERSION:
        return True
    generator = payload.get("generator", {})
    if generator.get("strategy") != GENERATOR_STRATEGY:
        return True
    return False


def ensure_split_manifest(output_path=None, force=False):
    path = Path(output_path) if output_path else OUTPUT_PATH
    if not force and not _needs_regeneration(path):
        return path

    rows = _load_rows()
    groups, assignment, expected, best_seed, best_score = _run_search(rows)
    payload = _manifest_payload(rows, groups, assignment, expected, best_seed, best_score)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    output_path = ensure_split_manifest(force=True)
    print(f"Wrote manifest to {output_path}")
