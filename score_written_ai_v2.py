"""Separate written v2 evaluation. No evaluation occurs on import.

Run only after approval: python score_written_ai_v2.py --only ID --model MODEL
Results go exclusively to outputs_v2/ID/written_ai_result_runN.json.
"""

import argparse
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import types

from score_written import RUBRIC_PATH, load_rubric, score_question
from score_written_ai import collapse, data_folder, fill_prompt, read_answers

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "prompts" / "score_written_v2.txt"
OUTPUT_ROOT = ROOT / "outputs_v2"
GENERATION_CONFIG = {"temperature": 0, "response_mime_type": "application/json"}
RESPONSE_FIELDS = ("evidence", "element_checks", "reasoning", "next_level_explanation", "levels", "flags")
ALLOWED_FLAGS = {"BLANK_OR_OFF_TOPIC", "BRAND_VALUES_REVIEW", "HARMFUL_CONTENT"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate(raw, active, answer, rubric=None):
    """Check structure, exact descriptor source, and quotes; not semantic correctness."""
    rubric = rubric if rubric is not None else load_rubric()
    result = {name: {} for name in RESPONSE_FIELDS[:-1]}
    result.update(flags=[], evidence_verified={code: [] for code in active}, marks=None,
                  status="unable_to_evaluate", raw_reply=raw)
    try:
        parsed = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(parsed, dict) or set(parsed) != set(RESPONSE_FIELDS):
            raise ValueError("Response must contain exactly evidence, element_checks, reasoning, next_level_explanation, levels, flags")
        for name in RESPONSE_FIELDS[:-1]:
            if not isinstance(parsed[name], dict) or set(parsed[name]) != set(active):
                raise ValueError(f"{name} must contain exactly the active parameters")
            result[name] = parsed[name]
        flags = parsed["flags"]
        if not isinstance(flags, list) or any(not isinstance(flag, str) or flag not in ALLOWED_FLAGS for flag in flags):
            raise ValueError("Invalid flags: use only BLANK_OR_OFF_TOPIC, BRAND_VALUES_REVIEW, HARMFUL_CONTENT")
        result["flags"] = flags
        errors = []
        for code in active:
            level = parsed["levels"][code]
            if type(level) is not int or not 0 <= level <= 8:
                errors.append(f"{code}: level must be an integer from 0 to 8")
                continue
            quotes = parsed["evidence"][code]
            if not isinstance(quotes, list) or (level == 0 and quotes != []) or (
                    level != 0 and not 1 <= len(quotes) <= 2):
                errors.append(f"{code}: evidence must be [] at level 0, otherwise one or two exact quotes")
            else:
                for index, quote in enumerate(quotes, 1):
                    found = nonempty(quote) and collapse(quote) in collapse(answer)
                    result["evidence_verified"][code].append(bool(found))
                    if not found:
                        errors.append(f"{code}: quote {index} not found in answer after whitespace normalization")
                    if isinstance(quote, str) and len(quote.split()) > 15:
                        errors.append(f"{code}: quote {index} exceeds 15 words")
            if not nonempty(parsed["reasoning"][code]):
                errors.append(f"{code}: reasoning must be non-empty text")
            checks = parsed["element_checks"][code]
            if not isinstance(checks, list) or not checks:
                errors.append(f"{code}: element_checks must be a non-empty list")
            else:
                for check in checks:
                    if (not isinstance(check, dict) or set(check) != {"descriptor_element", "status", "explanation"}
                            or not nonempty(check.get("descriptor_element"))
                            or check.get("status") not in ("supported", "unsupported", "not_assessable")
                            or not nonempty(check.get("explanation"))):
                        errors.append(f"{code}: each element check requires descriptor_element, allowed status and non-empty explanation")
            nxt = parsed["next_level_explanation"][code]
            expected_keys = {"next_anchor_level", "descriptor", "missing_elements", "explanation"}
            if not isinstance(nxt, dict) or set(nxt) != expected_keys:
                errors.append(f"{code}: next_level_explanation requires next_anchor_level, descriptor, missing_elements, explanation")
                continue
            missing = nxt["missing_elements"]
            if not isinstance(missing, list) or any(not nonempty(item) for item in missing):
                errors.append(f"{code}: missing_elements must be a list of non-empty text")
            if not nonempty(nxt["explanation"]):
                errors.append(f"{code}: next-level explanation must be non-empty text")
            if level == 8:
                if nxt["next_anchor_level"] is not None or nxt["descriptor"] is not None or missing != []:
                    errors.append(f"{code}: maximum level has no next anchor; use null, null, []")
            else:
                anchor = next(value for value in (0, 2, 4, 6, 8) if value > level)
                if type(nxt["next_anchor_level"]) is not int or nxt["next_anchor_level"] != anchor:
                    errors.append(f"{code}: next_anchor_level must be {anchor}")
                if nxt["descriptor"] != rubric["parameters"][code]["levels"][str(anchor)]:
                    errors.append(f"{code}: next descriptor must exactly match the rubric anchor {anchor}")
                if isinstance(missing, list) and not missing:
                    errors.append(f"{code}: identify at least one missing next-descriptor element below level 8")
        if errors:
            raise ValueError("; ".join(errors))
        result["status"] = "valid"
    except (ValueError, TypeError) as error:
        # Diagnostic messages contain field names/rules, never quoted candidate text.
        result["reason"] = "Invalid JSON or response types"
        # Own validation errors are safe; JSON parser messages are not included.
        if type(error) is ValueError:
            result["reason"] = str(error)
    return result


def evaluate_question(client, model, prompt, active, answer, rubric, sleep=time.sleep):
    """Same three attempts as v1; invalid-response diagnostics inform the next request."""
    feedback = None
    history = []
    for attempt, delay in enumerate((0, 2, 5), 1):
        if delay:
            sleep(delay)
        sent_prompt = prompt
        if feedback:
            sent_prompt += ("\n\nVALIDATOR FEEDBACK (structural correction only):\n" + feedback
                            + "\nReturn the complete corrected JSON. Keep judging only the original answer and rubric.\n")
        try:
            response = client.models.generate_content(model=model, contents=sent_prompt,
                                                      config=types.GenerateContentConfig(**GENERATION_CONFIG))
            evaluated = validate(response.text or "", active, answer, rubric)
        except Exception:
            evaluated = validate("", active, answer, rubric)
            evaluated["reason"] = "API failure; no valid response received"
            feedback = None
        else:
            feedback = evaluated.get("reason") if evaluated["status"] != "valid" else None
        history.append({"attempt": attempt, "prompt_sha256": hashlib.sha256(sent_prompt.encode()).hexdigest(),
                        "validator_error": evaluated.get("reason"), "generation_config": dict(GENERATION_CONFIG)})
        if evaluated["status"] == "valid":
            break
    evaluated.update(attempts=attempt, attempt_history=history,
                     filled_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    return evaluated


def reserve_run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    run = 1
    while True:
        path = folder / f"written_ai_result_run{run}.json"
        try:
            return run, path.open("x", encoding="utf-8")
        except FileExistsError:
            run += 1


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {name: redact(item, key) for name, item in value.items()}
    return value


def score_contestant(candidate_id, model):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id) or not model.strip():
        raise ValueError("Valid candidate ID and model are required")
    rubric = load_rubric()
    template = PROMPT.read_text(encoding="utf-8")
    answers = read_answers(data_folder() / candidate_id)
    key = dotenv_values(ROOT / ".env", interpolate=False).get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is missing from .env")
    run, handle = reserve_run(OUTPUT_ROOT / candidate_id)
    path = Path(handle.name)
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        result = {"evaluation_version": "written_v2", "candidate_id": candidate_id, "run": run,
                  "model": model, "temperature": GENERATION_CONFIG["temperature"],
                  "generation_config": dict(GENERATION_CONFIG),
                  "prompt_file": "prompts/score_written_v2.txt",
                  "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
                  "rubric_file": "config/rubric_written_v1.0.yaml",
                  "rubric_version": "v1.0", "rubric_version_source": "frozen source filename; YAML has no version field",
                  "rubric_sha256": hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest(),
                  "run_date": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(), "questions": {}}
        with handle, genai.Client(api_key=key) as client:
            for qid, question in rubric["questions"].items():
                active = [c for c, weight in rubric["weight_matrix"][qid].items() if weight is not None]
                answer = answers.get(qid)
                if not isinstance(answer, str):
                    evaluated = validate("", active, "", rubric)
                    evaluated.update(reason="Missing or invalid answer text", attempts=0, attempt_history=[])
                else:
                    prompt = fill_prompt(template, question, answer, active, rubric)
                    evaluated = evaluate_question(client, model, prompt, active, answer, rubric)
                evaluated["answer_sha256"] = hashlib.sha256(collapse(answer).encode()).hexdigest() if isinstance(answer, str) else None
                if evaluated["status"] == "valid":
                    evaluated["marks"] = float(score_question(qid, evaluated["levels"], rubric["weight_matrix"][qid])["marks"])
                result["questions"][qid] = evaluated
            valid = {q: item["levels"] for q, item in result["questions"].items() if item["status"] == "valid"}
            result["unable_to_evaluate_count"] = len(result["questions"]) - len(valid)
            result["total_is_partial"] = bool(result["unable_to_evaluate_count"])
            total = sum((score_question(q, levels, rubric["weight_matrix"][q])["marks"] for q, levels in valid.items()), Decimal(0))
            result.update(written_total=float(total), written_total_out_of=40)
            result = redact(result, key)
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return result, path
    finally:
        handle.close()
        logging.disable(previous_logging)
        if path.exists() and path.stat().st_size == 0:
            path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--only", required=True, help="One candidate ID; no bulk run in this step")
    args = parser.parse_args(argv)
    try:
        result, path = score_contestant(args.only, args.model)
    except Exception:
        print("V2 evaluation failed; check input, key, model and output access. No exception details are printed.")
        return 1
    print(f"{args.only}: saved {path.name}; failed questions {result['unable_to_evaluate_count']}")
    if not result["total_is_partial"]:
        print(f"Written v2 total: {result['written_total']:.2f} / 40")
    return 1 if result["total_is_partial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
