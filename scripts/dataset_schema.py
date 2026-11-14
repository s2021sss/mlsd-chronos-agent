"""Схема примера для оценки и её проверка, описание полей в data/README.md."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

OUTCOMES = {"plan", "answer", "clarify", "unsupported"}
OPS = {"create", "update", "delete"}
TYPES = {"event", "task", "reminder"}
SOURCES = {"seed_llm", "imitated_user", "synthetic_llm", "massive"}
LABEL_STATUSES = {"full", "weak"}


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"нет часового пояса: {value}")
    return parsed


def _check_interval(start: str, end: str, errors: list[str], label: str) -> None:
    try:
        if parse_datetime(end) <= parse_datetime(start):
            errors.append(f"{label}: end не позже start")
    except (TypeError, ValueError) as error:
        errors.append(f"{label}: {error}")


def _check_when(item: dict, now: datetime, errors: list[str], label: str) -> None:
    kind, start, end = item["type"], item.get("start"), item.get("end")
    try:
        if kind == "event":
            _check_interval(start, end, errors, label)
        elif kind == "reminder":
            if end is not None:
                errors.append(f"{label}: у напоминания нет end")
            parse_datetime(start)
        else:
            if end is not None:
                errors.append(f"{label}: у задачи нет end")
            if date.fromisoformat(start) < now.date():
                errors.append(f"{label}: срок задачи в прошлом")
    except (TypeError, ValueError) as error:
        errors.append(f"{label}: {error}")
        return

    if kind != "task" and parse_datetime(start) <= now:
        errors.append(f"{label}: start в прошлом")
    remind = item.get("remind")
    if remind is not None:
        if kind != "event":
            errors.append(f"{label}: remind только у встречи")
        elif not now < parse_datetime(remind) < parse_datetime(start):
            errors.append(f"{label}: remind должен быть между now и start")


def _check_item(item: dict, example: dict, now: datetime, errors: list[str], label: str) -> None:
    op, kind = item.get("op"), item.get("type")
    if op not in OPS:
        errors.append(f"{label}: неизвестная операция {op}")
        return
    if kind not in TYPES:
        errors.append(f"{label}: неизвестный тип {kind}")
        return
    if not item.get("title"):
        errors.append(f"{label}: нет title")
    if not item.get("span") or item["span"] not in example["text"]:
        errors.append(f"{label}: span не найден в тексте: {item.get('span')!r}")

    target = item.get("target")
    if op == "create":
        if target is not None:
            errors.append(f"{label}: у create нет target")
    else:
        entry = next((e for e in example["calendar"] if e["id"] == target), None)
        if entry is None:
            errors.append(f"{label}: target {target!r} нет в календаре")
        elif entry["type"] != kind or entry["title"] != item.get("title"):
            errors.append(f"{label}: тип или title расходятся с записью {target}")

    if op == "delete":
        if any(item.get(field) is not None for field in ("start", "end", "remind")):
            errors.append(f"{label}: у delete нет времени")
    else:
        _check_when(item, now, errors, label)


def validate(example: dict) -> list[str]:
    """Возвращает список ошибок, пустой список значит, что пример корректен."""
    errors: list[str] = []
    label = example.get("id", "?")
    if example.get("source") not in SOURCES:
        errors.append(f"{label}: неизвестный source {example.get('source')}")
    if example.get("label_status") not in LABEL_STATUSES:
        errors.append(f"{label}: неизвестный label_status {example.get('label_status')}")
    if example.get("outcome") not in OUTCOMES:
        errors.append(f"{label}: неизвестный outcome {example.get('outcome')}")
    if not example.get("text"):
        errors.append(f"{label}: пустой text")
    if errors:
        return errors

    items = example["items"]
    if example["label_status"] == "weak":
        for item in items:
            if item["op"] not in OPS or item["type"] not in TYPES:
                errors.append(f"{label}: неизвестная операция или тип")
        return errors

    try:
        now = parse_datetime(example["now"])
        if now.utcoffset() != now.astimezone(ZoneInfo(example["timezone"])).utcoffset():
            errors.append(f"{label}: смещение now не совпадает с timezone")
    except (KeyError, TypeError, ValueError) as error:
        return [f"{label}: {error}"]

    outcome = example["outcome"]
    if (outcome == "plan") != bool(items):
        errors.append(f"{label}: элементы плана должны быть только у исхода plan")
    if (outcome == "clarify") != bool(example["missing"]):
        errors.append(f"{label}: missing должен быть только у исхода clarify")
    if outcome != "answer" and example["answer_window"] is not None:
        errors.append(f"{label}: answer_window только у исхода answer")
    if example["answer_window"] is not None:
        window = example["answer_window"]
        _check_interval(window["start"], window["end"], errors, f"{label} answer_window")

    for entry in example["calendar"]:
        if entry["type"] == "event":
            _check_interval(entry["start"], entry["end"], errors, f"{label} {entry['id']}")
    for position, item in enumerate(items):
        _check_item(item, example, now, errors, f"{label} item {position}")
    return errors
