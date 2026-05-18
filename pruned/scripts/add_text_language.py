"""
기존 jsonl 에 text_language 필드 추가 (한국어 학습용).

Usage:
  python scripts/add_text_language.py /deepet/jh/sensevoice/finetune/jsonl/*.jsonl
  python scripts/add_text_language.py --lang "<|ko|>" <files>...
"""
import argparse
import json
from pathlib import Path


def patch(path: Path, lang: str) -> tuple[int, int]:
    n_added = n_kept = 0
    out_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text_language" in obj:
                n_kept += 1
            else:
                obj["text_language"] = lang
                n_added += 1
            out_lines.append(json.dumps(obj, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return n_added, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="<|ko|>", help="text_language 값 (기본: <|ko|>)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    for fp in args.files:
        p = Path(fp)
        added, kept = patch(p, args.lang)
        print(f"  {p.name}: added={added}, already_had={kept}")


if __name__ == "__main__":
    main()
