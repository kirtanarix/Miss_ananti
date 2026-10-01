"""Compute written marks from supplied levels; no model or network calls."""

import argparse
import json
from decimal import Decimal
from pathlib import Path


RUBRIC_PATH = Path(__file__).parent / "config" / "rubric_written_v1.0.yaml"


def load_rubric():
    # The config uses JSON syntax with full-line YAML comments, requiring no packages.
    text = RUBRIC_PATH.read_text(encoding="utf-8")
    return json.loads("\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")))


def score_question(question, levels, weights):
    if not isinstance(levels, dict):
        raise ValueError(f"{question}: levels must be a JSON object.")
    for parameter in levels:
        if parameter not in weights:
            raise ValueError(f"{question}: unknown parameter {parameter}.")
        if weights[parameter] is None:
            raise ValueError(f"{question}: {parameter} is inactive (not scored); omit it.")
    fraction = Decimal(0)
    for parameter, weight in weights.items():
        if weight is None:
            continue
        if parameter not in levels:
            raise ValueError(f"{question}: missing level for active cell {parameter}.")
        value = levels[parameter]
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise ValueError(f"{question} {parameter}: level must be a number from 0 to 8.")
        level = Decimal(str(value))
        if not level.is_finite() or not Decimal(0) <= level <= Decimal(8):
            raise ValueError(f"{question} {parameter}: level must be from 0 to 8; got {value}.")
        fraction += level / 8 * Decimal(weight.removesuffix("%")) / 100
    return {"percent": fraction * 100, "marks": fraction * 8}


def score_written(levels, rubric=None):
    matrix = (rubric if rubric is not None else load_rubric())["weight_matrix"]
    if not isinstance(levels, dict):
        raise ValueError("Input must be a JSON object keyed by Q1, Q2, Q3, Q4, Q5.")
    unknown = set(levels) - set(matrix)
    if unknown:
        raise ValueError(f"Unknown question(s): {', '.join(sorted(unknown))}.")
    results = {}
    for question, weights in matrix.items():
        if question not in levels:
            raise ValueError(f"Missing question {question}.")
        results[question] = score_question(question, levels[question], weights)
    return results, sum((r["marks"] for r in results.values()), Decimal(0))


def display(number):
    return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else str(number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON object: Q1..Q5, each mapping active parameters to levels")
    args = parser.parse_args()
    try:
        levels = json.loads(args.input.read_text(encoding="utf-8-sig"), parse_float=Decimal)
        results, total = score_written(levels)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    for question, result in results.items():
        print(f"{question}: {display(result['percent'])}% | {display(result['marks'])} / 8 marks")
    print(f"Written total: {display(total)} / 40 marks")


if __name__ == "__main__":
    main()
