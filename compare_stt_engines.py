"""Compare saved STT transcripts locally using only the Python standard library."""

from difflib import SequenceMatcher
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parent
ENGINES = ("gemini_v2", "elevenlabs_v1", "elevenlabs_v2")
PAIRS = ((ENGINES[0], ENGINES[1]), (ENGINES[0], ENGINES[2]),
         (ENGINES[1], ENGINES[2]))


def transcript_body(path):
    text = path.read_text(encoding="utf-8-sig").lstrip()
    # Both saved formats use a Model: metadata block and a blank separator.
    lines = text.splitlines()
    if lines and lines[0].startswith("Model:"):
        for index, line in enumerate(lines):
            if not line.strip():
                return "\n".join(lines[index + 1:]).strip()
        raise ValueError(f"Missing blank line after header: {path}")
    return text.strip()


def normalized_words(text):
    return "".join(
        char for char in text.lower()
        if not unicodedata.category(char).startswith("P")
    ).split()


def script_counts(text):
    devanagari = sum("DEVANAGARI" in unicodedata.name(char, "")
                     and unicodedata.category(char)[0] in "LM" for char in text)
    latin = sum("LATIN" in unicodedata.name(char, "") and char.isalpha()
                for char in text)
    if devanagari == latin:
        main_script = "mixed (tie)" if latin else "neither"
    else:
        main_script = "Devanagari" if devanagari > latin else "Latin"
    return main_script, devanagari, latin


def differing_places(first, second):
    lines = []
    for operation, i, j, k, l in SequenceMatcher(
        None, first, second, autojunk=False
    ).get_opcodes():
        if operation == "equal":
            continue
        a = " ".join(first[max(0, i - 4):min(len(first), j + 4)])
        b = " ".join(second[max(0, k - 4):min(len(second), l + 4)])
        lines.append(f"A said: {a} | B said: {b}")
    return lines


def main():
    candidates = []
    for folder in sorted((ROOT / "outputs").iterdir()):
        if folder.is_dir() and all(
            (folder / f"transcript_{engine}.txt").is_file() for engine in ENGINES
        ):
            candidates.append(folder)

    # Refuse to overwrite any existing report, including on subsequent runs.
    for folder in candidates:
        target = folder / "compare_stt_engines.txt"
        if target.exists():
            raise FileExistsError(f"Existing file will not be modified: {target}")

    summary = []
    for folder in candidates:
        words = {}
        scripts = {}
        print(f"\n{folder.name} (word counts use normalized transcript text)")
        for engine in ENGINES:
            body = transcript_body(folder / f"transcript_{engine}.txt")
            words[engine] = normalized_words(body)
            scripts[engine], devanagari, latin = script_counts(body)
            print(f"  {engine}: words={len(words[engine])}, "
                  f"ananti={body.lower().count('ananti')}, "
                  f"main script={scripts[engine]} "
                  f"(Devanagari={devanagari}, Latin={latin})")

        report = []
        for first, second in PAIRS:
            if scripts[first] != scripts[second]:
                print(f"  WARNING: {first} vs {second}: the diff will be noisy "
                      "because of the script difference.")
            differences = differing_places(words[first], words[second])
            report.append(f"A = {first} | B = {second}")
            report.extend(differences)
            report.append("")
            summary.append((folder.name, f"{first} vs {second}", len(differences)))

        with (folder / "compare_stt_engines.txt").open(
            "x", encoding="utf-8"
        ) as output:
            output.write("\n".join(report))

    print("\nCandidate  Pair                                    Differing places")
    print("---------  --------------------------------------  ----------------")
    for candidate, pair, count in summary:
        print(f"{candidate:<9}  {pair:<38}  {count}")


if __name__ == "__main__":
    main()
