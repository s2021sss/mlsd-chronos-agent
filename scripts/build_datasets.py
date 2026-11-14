"""Собирает наборы для оценки в единой схеме, проверяет их и записывает в JSONL.

uv run python scripts/build_datasets.py
"""

import json
import random
import re
from collections import Counter
from datetime import timedelta
from pathlib import Path

import yaml
from dataset_schema import parse_datetime, validate

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "source"
PROCESSED = DATA / "processed"
INTERIM = DATA / "interim"
MASSIVE = DATA / "raw" / "massive" / "ru-RU.jsonl"
SPLIT_SEED = 20261114
TEST_SHARE = 0.6

SLOT = re.compile(r"\[(\w+) : ([^\]]+)\]")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
RELATIVE_DAYS = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2}
# Как действие из seed-набора выражается в нашей схеме: (op, type) или исход без плана.
SEED_ACTIONS = {
    "create_event": ("create", "event"),
    "create_task": ("create", "task"),
    "move_event": ("update", "event"),
    "complete_task": None,
    "unknown": None,
}


def make_example(raw: dict, source: str, defaults: dict) -> dict:
    items = [
        {
            "op": item["op"],
            "type": item["type"],
            "title": item["title"],
            "start": item.get("start"),
            "end": item.get("end"),
            "remind": item.get("remind"),
            "target": item.get("target"),
            "span": item["span"],
        }
        for item in raw.get("items", [])
    ]
    return {
        "id": raw["id"],
        "source": source,
        "label_status": "full",
        "text": raw["text"],
        "now": raw.get("now", defaults["now"]),
        "timezone": raw.get("timezone", defaults["timezone"]),
        "outcome": raw["outcome"],
        "items": items,
        "missing": raw.get("missing", []),
        "answer_window": raw.get("answer_window"),
        "calendar": raw.get("calendar", []),
        "tags": raw.get("tags", []),
        "meta": raw.get("meta", {}),
    }


def seed_mismatches(example: dict, expected: dict) -> list[str]:
    """Сравнивает эталон с частичными ожиданиями seed-набора и возвращает расхождения."""
    actions = [expected["action"], *expected.get("action_accepts", [])]
    items = example["items"]
    if example["outcome"] == "plan":
        pairs = {(item["op"], item["type"]) for item in items}
        if not pairs & {SEED_ACTIONS[action] for action in actions}:
            return [f"действие {actions} против {sorted(pairs)}"]
    elif "unknown" not in actions:
        return [f"действие {actions} против исхода {example['outcome']}"]
    if not items:
        return []

    problems = []
    item = items[0]
    for key in ("title_contains", "target_title_contains"):
        if key in expected and expected[key].lower() not in item["title"].lower():
            problems.append(f"{key}: {expected[key]} против {item['title']}")
    if item["type"] == "task":
        return problems

    start = parse_datetime(item["start"])
    if "start_hour" in expected and start.hour != expected["start_hour"]:
        problems.append(f"start_hour: {expected['start_hour']} против {start.hour}")
    if "start_hour_min" in expected and start.hour < expected["start_hour_min"]:
        problems.append(f"start_hour_min: {expected['start_hour_min']} против {start.hour}")
    if "start_hour_max" in expected and start.hour > expected["start_hour_max"]:
        problems.append(f"start_hour_max: {expected['start_hour_max']} против {start.hour}")
    if "start_weekday" in expected and WEEKDAYS[start.weekday()] != expected["start_weekday"]:
        problems.append(f"start_weekday: {expected['start_weekday']} против {start:%A}")
    if "start_date" in expected:
        now = parse_datetime(example["now"])
        wanted = now.date() + timedelta(days=RELATIVE_DAYS[expected["start_date"]])
        if start.date() != wanted:
            problems.append(f"start_date: {wanted} против {start.date()}")
    if "duration_minutes" in expected and item["end"]:
        minutes = (parse_datetime(item["end"]) - start) / timedelta(minutes=1)
        if abs(minutes - expected["duration_minutes"]) > 15:
            problems.append(f"duration_minutes: {expected['duration_minutes']} против {minutes}")
    return problems


def build_seed() -> list[dict]:
    seed = {case["id"]: case for case in yaml.safe_load((SOURCE / "nlu_dataset.yaml").read_text())}
    gold = yaml.safe_load((SOURCE / "seed_gold.yaml").read_text())
    if set(seed) != {raw["id"] for raw in gold["examples"]}:
        raise SystemExit("seed_gold.yaml не покрывает seed-набор один к одному")

    examples = []
    for raw in gold["examples"]:
        case = seed[raw["id"]]
        reason = raw.get("changed_from_seed")
        meta = {"category": case["category"], "seed_expected": case["expected"]}
        if reason:
            meta["changed_from_seed"] = reason
        example = make_example(
            {**raw, "text": case["input"].strip(), "meta": meta}, "seed_llm", gold["defaults"]
        )
        mismatches = seed_mismatches(example, case["expected"])
        if mismatches and not reason:
            raise SystemExit(f"{raw['id']}: расхождение без объяснения: {mismatches}")
        if reason and not mismatches:
            raise SystemExit(f"{raw['id']}: указана причина, но расхождения нет")
        examples.append(example)
    return examples


def build_imitated() -> list[dict]:
    data = yaml.safe_load((SOURCE / "imitated_user.yaml").read_text())
    return [make_example(raw, "imitated_user", data["defaults"]) for raw in data["examples"]]


def massive_outcome(intent: str, slots: set[str]) -> str:
    action = intent.split("_")[1]
    if action == "query":
        return "answer"
    if action == "set" and not slots & {"date", "time", "timeofday"}:
        return "clarify"
    return "plan"


def massive_item(row: dict, slots: dict[str, str]) -> dict:
    text, action = row["utt"], row["intent"].split("_")[1]
    is_reminder = row["scenario"] == "alarm" or "напомин" in text or "напомни" in text
    op = "update" if "перенес" in text else {"set": "create", "remove": "delete"}[action]
    return {
        "op": op,
        "type": "reminder" if is_reminder else "event",
        "title": slots.get("event_name"),
        "start": None,
        "end": None,
        "remind": None,
        "target": None,
        "span": None,
    }


def build_massive() -> list[dict]:
    examples = []
    for line in MASSIVE.read_text().splitlines():
        row = json.loads(line)
        if row["scenario"] not in {"calendar", "alarm"}:
            continue
        pairs = SLOT.findall(row["annot_utt"])
        slots = {name: value.strip() for name, value in pairs}
        outcome = massive_outcome(row["intent"], set(slots))
        judgments = row["judgments"]
        agree = sum(j["intent_score"] == 1 and j["slots_score"] == 1 for j in judgments)
        examples.append(
            {
                "id": f"MASSIVE-{row['id']}",
                "source": "massive",
                "label_status": "weak",
                "text": row["utt"],
                "now": None,
                "timezone": None,
                "outcome": outcome,
                "items": [massive_item(row, slots)] if outcome == "plan" else [],
                "missing": [],
                "answer_window": None,
                "calendar": [],
                "tags": [],
                "meta": {
                    "intent": row["intent"],
                    "partition": row["partition"],
                    "slot_method": row["slot_method"],
                    "slots": [{"name": n, "value": v.strip()} for n, v in pairs],
                    "annotators_agree": agree,
                    "annotators_total": len(judgments),
                },
            }
        )
    return examples


def assign_splits(examples: list[dict], seed: int = SPLIT_SEED) -> None:
    """Делит на dev и test со стратификацией по источнику, исходу и наличию списка.

    Seed-набор целиком идёт в dev: он командный и проще остальных.
    """
    groups: dict[tuple, list[dict]] = {}
    for example in examples:
        if example["source"] == "seed_llm":
            example["split"] = "dev"
            continue
        key = (example["source"], example["outcome"], len(example["items"]) > 1)
        groups.setdefault(key, []).append(example)
    rng = random.Random(seed)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda example: example["id"])
        rng.shuffle(group)
        test_size = round(len(group) * TEST_SHARE)
        for position, example in enumerate(group):
            example["split"] = "test" if position < test_size else "dev"


def write(examples: list[dict], path: Path) -> None:
    errors = [error for example in examples for error in validate(example)]
    ids = Counter(example["id"] for example in examples)
    errors += [f"{key}: повторяющийся id" for key, count in ids.items() if count > 1]
    if errors:
        raise SystemExit("\n".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(example, ensure_ascii=False) for example in examples)
    path.write_text("\n".join(lines) + "\n")
    outcomes = dict(Counter(example["outcome"] for example in examples))
    print(f"{path.relative_to(DATA.parent)}: {len(examples)} примеров, {outcomes}")


def main() -> None:
    sets = {"seed": build_seed(), "imitated_user": build_imitated()}
    assign_splits([example for examples in sets.values() for example in examples])
    for name, examples in sets.items():
        if examples:
            write(examples, PROCESSED / f"{name}.jsonl")
    labeled = [example for examples in sets.values() for example in examples]
    print("split:", dict(Counter((e["split"], e["source"]) for e in labeled)))
    if MASSIVE.exists():
        write(build_massive(), INTERIM / "massive_calendar_alarm.jsonl")
    else:
        print("MASSIVE пропущен: сначала запустите scripts/download_massive.py")


if __name__ == "__main__":
    main()
