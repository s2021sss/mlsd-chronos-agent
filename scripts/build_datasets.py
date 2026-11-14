"""Собирает наборы для оценки в единой схеме, проверяет их и записывает в JSONL.

uv run python scripts/build_datasets.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from dataset_schema import validate

DATA = Path(__file__).resolve().parents[1] / "data"
INTERIM = DATA / "interim"
MASSIVE = DATA / "raw" / "massive" / "ru-RU.jsonl"

SLOT = re.compile(r"\[(\w+) : ([^\]]+)\]")


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
    if MASSIVE.exists():
        write(build_massive(), INTERIM / "massive_calendar_alarm.jsonl")
    else:
        print("MASSIVE пропущен: сначала запустите scripts/download_massive.py")


if __name__ == "__main__":
    main()
