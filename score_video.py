"""Calculate video marks from supplied integer levels; no model or network calls."""

import argparse
import json
from decimal import Decimal
from pathlib import Path


RUBRIC_PATH = Path(__file__).resolve().parent / "config" / "rubric_video_v1.0.yaml"


def load_rubric():
    # JSON-compatible YAML with full-line comments, like the written calculator.
    text = RUBRIC_PATH.read_text(encoding="utf-8")
    return json.loads("\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ))


def score_video(levels):
    """Return (parameter marks, total), using Decimal without display rounding."""
    parameters = load_rubric()["parameters"]
    if not isinstance(levels, dict):
        raise ValueError("Input must be a JSON object mapping parameters to levels.")
    unknown = set(levels) - set(parameters)
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}.")
    missing = set(parameters) - set(levels)
    if missing:
        raise ValueError(f"Missing parameter(s): {', '.join(sorted(missing))}.")
    marks = {}
    for parameter, details in parameters.items():
        level = levels[parameter]
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError(f"{parameter}: level must be an integer from 0 to 8; got {level!r}.")
        if not 0 <= level <= 8:
            raise ValueError(f"{parameter}: level must be from 0 to 8; got {level}.")
        marks[parameter] = Decimal(level) / 8 * Decimal(str(details["maximum_marks"]))
    return marks, sum(marks.values(), Decimal(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="JSON object with SA, Goals, Purpose, Clarity, Voice, Composure integer levels")
    args = parser.parse_args()
    try:
        levels = json.loads(args.input.read_text(encoding="utf-8-sig"), parse_float=Decimal)
        marks, total = score_video(levels)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    for parameter, value in marks.items():
        print(f"{parameter}: {value:.2f}")
    maximum = sum(Decimal(str(p["maximum_marks"])) for p in load_rubric()["parameters"].values())
    print(f"Video total: {total:.2f} / {maximum:.2f}")


if __name__ == "__main__":
    main()
