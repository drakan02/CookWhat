import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm


INPUT_PATH = "data/recipes.jsonl"
OUTPUT_PATH = "data/chunks.jsonl"
DESC_MAX_CHARS = 200


@dataclass
class ChunkRecord:
    id: str
    document: str
    metadata: dict


def _clean(value) -> str:
    return str(value or "").strip()


def _truncate_words(text: str, max_chars: int) -> str:
    text = _clean(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _to_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _detect_format(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return "empty"
            if not ch.isspace():
                return "json-array" if ch == "[" else "jsonl"


def iter_recipes(path: Path):
    fmt = _detect_format(path)
    if fmt == "empty":
        return
    if fmt == "json-array":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for recipe in payload:
            yield recipe
        return

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            try:
                yield json.loads(row)
            except json.JSONDecodeError as exc:
                print(f"  [!] JSON lỗi ở dòng {line_no}: {exc}")


def build_document(recipe: dict) -> str:
    sections = []

    title = _clean(recipe.get("title"))
    if title:
        sections.append(f"Tên món: {title}")

    description = _truncate_words(recipe.get("description"), DESC_MAX_CHARS)
    if description:
        sections.append(f"Mô tả: {description}")

    cook_time = _clean(recipe.get("cook_time"))
    servings = _clean(recipe.get("servings"))
    if cook_time or servings:
        sections.append(f"Thời gian nấu: {cook_time} | Khẩu phần: {servings}")

    ner_values = _to_list(recipe.get("ner"))
    if ner_values:
        sections.append(f"Nguyên liệu chính: {', '.join(ner_values)}")

    ingredient_values = _to_list(recipe.get("ingredients"))
    if ingredient_values:
        lines = "\n".join(f"- {item}" for item in ingredient_values)
        sections.append(f"Nguyên liệu chi tiết:\n{lines}")

    steps_raw = recipe.get("steps") if isinstance(recipe.get("steps"), list) else []
    step_lines = []
    for step in steps_raw:
        if isinstance(step, dict) and _clean(step.get("text")):
            step_lines.append(_clean(step.get("text")))
    if step_lines:
        sections.append("Cách làm:\n" + "\n".join(step_lines))

    return "\n\n".join(sections)


def build_metadata(recipe: dict) -> dict:
    ner_values = _to_list(recipe.get("ner"))
    return {
        "recipe_id": _clean(recipe.get("id")),
        "title": _clean(recipe.get("title")),
        "url": _clean(recipe.get("url")),
        "cook_time": _clean(recipe.get("cook_time")),
        "servings": _clean(recipe.get("servings")),
        "ner": ", ".join(ner_values),
        "author": _clean(recipe.get("author")),
        "author_location": _clean(recipe.get("author_location")),
    }


def make_chunk(recipe: dict) -> ChunkRecord | None:
    recipe_id = _clean(recipe.get("id"))
    if not recipe_id:
        return None

    document = build_document(recipe)
    if not document.strip():
        return None

    return ChunkRecord(id=recipe_id, document=document, metadata=build_metadata(recipe))


def run_chunking(input_path: str, output_path: str) -> int:
    source = Path(input_path)
    target = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(target, "w", encoding="utf-8") as fout:
        for recipe in tqdm(iter_recipes(source), desc="Chunking"):
            chunk = make_chunk(recipe)
            if chunk is None:
                skipped += 1
                continue
            fout.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            written += 1

    print("\n[chunking] Hoàn thành")
    print(f"  Chunks hợp lệ : {written}")
    print(f"  Bỏ qua        : {skipped}")
    print(f"  Output        : {target}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk recipe JSONL/JSON thành text chuẩn cho embedding")
    parser.add_argument("--input", default=INPUT_PATH, help="File đầu vào JSONL hoặc JSON array")
    parser.add_argument("--output", default=OUTPUT_PATH, help="File chunks JSONL đầu ra")
    cli_args = parser.parse_args()
    run_chunking(cli_args.input, cli_args.output)
