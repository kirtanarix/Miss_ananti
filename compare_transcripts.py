from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def transcript_body(path):
    text = path.read_text(encoding="utf-8")
    header, separator, body = text.partition("\n\n")
    if separator and header.startswith("Model:"):
        return body
    return text


def main():
    for folder in sorted((ROOT / "outputs").iterdir()):
        if not folder.is_dir():
            continue
        try:
            first = transcript_body(folder / "transcript_gemini_v2.txt")
            second = transcript_body(folder / "transcript_gemini_v2_run2.txt")
            print(f"{folder.name}: identical = {first == second}")
            if first == second:
                continue
            first_words, second_words = first.split(), second.split()
            if first_words == second_words:
                print("  Whitespace differs; words are identical.")
                continue
            for operation, i, j, k, l in SequenceMatcher(
                None, first_words, second_words, autojunk=False
            ).get_opcodes():
                if operation != "equal":
                    print(f"  v2:   {' '.join(first_words[i:j])!r}")
                    print(f"  run2: {' '.join(second_words[k:l])!r}")
        except OSError as exc:
            print(f"FAILED {folder.name}: {exc}")


if __name__ == "__main__":
    main()
