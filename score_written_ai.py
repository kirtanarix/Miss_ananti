"""Score written answers with Gemini; retain replies and append judge review rows."""

import argparse
import csv
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import types

from score_written import display, load_rubric, score_question, score_written

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "prompts" / "score_written_v1.txt"
FIELDS = ["candidate_id", "question_id", "parameter", "answer_text", "ai_level",
          "evidence", "ai_reasoning", "evidence_verified", "run", "judge_verdict",
          "judge_level", "judge_comment"]


def collapse(text):
    return " ".join(text.split())


def fill_prompt(template, question, answer, active, rubric):
    blocks = []
    for code in active:
        descriptor = rubric["parameters"][code]
        blocks.append("\n".join([code, descriptor["name"], descriptor["what_earns_marks"]]
                                + [f"{level}: {text}" for level, text in descriptor["levels"].items()]))
    values = {"question": question, "answer": answer,
              "parameter_block": "\n\n".join(blocks),
              "not_scored_block": "\n".join(rubric["not_scored"])}
    # Replace only placeholders in the original template, never text in an answer.
    parts = re.split(r"(\{(?:question|answer|parameter_block|not_scored_block)\})", template)
    for index, part in enumerate(parts):
        for name, value in values.items():
            if part == "{" + name + "}":
                parts[index] = part.replace("{" + name + "}", value)
                break
    return "".join(parts)


def validate(raw, active, answer):
    result = {"levels": {}, "evidence": {}, "reasoning": {}, "flags": [],
              "evidence_verified": {code: False for code in active},
              "marks": None, "status": "unable_to_evaluate", "raw_reply": raw}
    try:
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("Duplicate JSON key")
                obj[key] = value
            return obj
        parsed = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(parsed, dict):
            raise ValueError("Reply must be a JSON object")
        for field in ("levels", "evidence", "reasoning"):
            if not isinstance(parsed.get(field), dict) or set(parsed[field]) != set(active):
                raise ValueError(f"{field} must contain exactly the active parameters")
            result[field] = parsed[field]
        flags = parsed.get("flags")
        if not isinstance(flags, list) or any(not isinstance(flag, str) or flag not in
                {"BLANK_OR_OFF_TOPIC", "BRAND_VALUES_REVIEW", "HARMFUL_CONTENT"} for flag in flags):
            raise ValueError("Invalid flags")
        result["flags"] = flags
        errors = []
        for code in active:
            level = parsed["levels"][code]
            phrase = parsed["evidence"][code]
            if type(level) is not int or not 0 <= level <= 8:
                errors.append(f"{code}: level must be an integer from 0 to 8")
            if not isinstance(parsed["reasoning"][code], str):
                errors.append(f"{code}: reasoning must be text")
            found = isinstance(phrase, str) and bool(collapse(phrase)) and collapse(phrase) in collapse(answer)
            result["evidence_verified"][code] = found
            if not isinstance(phrase, str) or (level != 0 and not found):
                errors.append(f"{code}: evidence not found in answer")
            elif level == 0 and phrase != "":
                errors.append(f"{code}: zero level requires empty evidence")
        if errors:
            raise ValueError("; ".join(errors))
        result["status"] = "valid"
    except (ValueError, TypeError) as error:
        result["reason"] = str(error)
    return result


def reserve_run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    run = 1
    while True:
        path = folder / f"written_ai_result_run{run}.json"
        try:
            return run, path.open("x", encoding="utf-8")
        except FileExistsError:
            run += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--only", help="Candidate folder ID")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    logging.disable(logging.CRITICAL)
    key = dotenv_values(ROOT / ".env", interpolate=False).get("GEMINI_API_KEY")
    if not key:
        parser.exit(1, "GEMINI_API_KEY is missing from .env\n")
    rubric = load_rubric()
    template = PROMPT.read_text(encoding="utf-8")
    candidates = sorted(path for path in (ROOT / "data").iterdir()
                        if path.is_dir() and (path / "written_answers.json").is_file()
                        and (args.only is None or path.name == args.only))
    if not candidates:
        parser.exit(1, "No matching candidates with written_answers.json\n")
    sheet = ROOT / "outputs" / "written_review_sheet.csv"
    if sheet.exists() and sheet.stat().st_size:
        with sheet.open(encoding="utf-8-sig", newline="") as existing:
            if next(csv.reader(existing)) != FIELDS:
                parser.exit(1, "Existing review sheet has incompatible columns; left unchanged\n")
    with genai.Client(api_key=key) as client:
        for candidate in candidates:
            try:
                answers = json.loads((candidate / "written_answers.json").read_text(encoding="utf-8-sig"))
                if not isinstance(answers, dict):
                    raise ValueError()
            except (ValueError, OSError):
                answers = {}
            history = []
            for _ in range(args.repeats):
                run, handle = reserve_run(ROOT / "outputs" / candidate.name)
                result = {"candidate_id": candidate.name, "run": run, "model": args.model,
                          "run_date": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
                          "prompt_file": "prompts/score_written_v1.txt", "questions": {}}
                with handle:
                    for qid, question in rubric["questions"].items():
                        active = [code for code, weight in rubric["weight_matrix"][qid].items() if weight is not None]
                        answer = answers.get(qid)
                        raw = ""
                        failure = None
                        if not isinstance(answer, str):
                            failure = "Missing or invalid answer text"
                        else:
                            try:
                                response = client.models.generate_content(
                                    model=args.model,
                                    contents=fill_prompt(template, question, answer, active, rubric),
                                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                                raw = response.text or ""
                            except Exception as error:
                                # Do not expose SDK exception messages, request URLs or credentials.
                                failure = f"API failure ({type(error).__name__})"
                        evaluated = validate(raw, active, answer if isinstance(answer, str) else "")
                        if failure:
                            evaluated["reason"] = failure
                        if evaluated["status"] == "valid":
                            evaluated["marks"] = float(score_question(qid, evaluated["levels"], rubric["weight_matrix"][qid])["marks"])
                        result["questions"][qid] = evaluated
                    valid = {qid: item["levels"] for qid, item in result["questions"].items() if item["status"] == "valid"}
                    result["unable_to_evaluate_count"] = len(result["questions"]) - len(valid)
                    result["total_is_partial"] = bool(result["unable_to_evaluate_count"])
                    total = (sum((score_question(qid, levels, rubric["weight_matrix"][qid])["marks"]
                                  for qid, levels in valid.items()), Decimal(0))
                             if result["total_is_partial"] else score_written(valid, rubric)[1])
                    result["written_total"] = float(total)
                    result["written_total_out_of"] = 40
                    json.dump(result, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                needs_header = not sheet.exists() or sheet.stat().st_size == 0
                with sheet.open("a", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    if needs_header:
                        writer.writeheader()
                    for qid, item in result["questions"].items():
                        for code, weight in rubric["weight_matrix"][qid].items():
                            if weight is not None:
                                writer.writerow({"candidate_id": candidate.name, "question_id": qid,
                                    "parameter": code, "answer_text": answers.get(qid, ""),
                                    "ai_level": item["levels"].get(code, ""), "evidence": item["evidence"].get(code, ""),
                                    "ai_reasoning": item["reasoning"].get(code, ""),
                                    "evidence_verified": item["evidence_verified"][code], "run": run})
                history.append(result)
                print(f"\n{candidate.name} | run {run}")
                print("Question | " + " | ".join(rubric["parameters"]) + " | Marks")
                for qid, item in result["questions"].items():
                    cells = [str(item["levels"].get(code, "-")) for code in rubric["parameters"]]
                    marks = str(item["marks"]) if item["status"] == "valid" else "unable_to_evaluate"
                    print(qid + " | " + " | ".join(cells) + " | " + marks)
                print(f"Written total{' (PARTIAL: valid questions only)' if result['total_is_partial'] else ''}: {display(total)} / 40")
                print(f"Unable to evaluate: {result['unable_to_evaluate_count']} questions")
                for qid, item in result["questions"].items():
                    if item["status"] != "valid":
                        print(f"{qid}: {item['reason']}")
            if args.repeats > 1:
                for qid, weights in rubric["weight_matrix"].items():
                    for code, weight in weights.items():
                        if weight is not None:
                            levels = [r["questions"][qid]["levels"].get(code)
                                      if r["questions"][qid]["status"] == "valid" else None for r in history]
                            state = "changed" if len(set(v for v in levels if v is not None)) > 1 else "unchanged"
                            if None in levels:
                                state += "; comparison incomplete (unable_to_evaluate)"
                            print(f"{qid} {code}: {state}; levels={levels}")


if __name__ == "__main__":
    main()
