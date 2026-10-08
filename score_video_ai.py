"""Exploratory direct-video Gemini scoring; frozen V1 calculates all marks.

Run: .venv\\Scripts\\python.exe score_video_ai.py ID VIDEO --model MODEL
One technical retry: add --retry-of RUN, optionally with a replacement video.
Results append to outputs/ID/video_ai_runN.json; no official report is touched.
"""

import argparse
import hashlib
import json
import logging
import math
import mimetypes
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from score_video import RUBRIC_PATH, load_rubric, score_video

ROOT = Path(__file__).resolve().parent
PROMPT = ROOT / "prompts" / "score_video_v1.txt"
Signal = Literal["speech", "task_completion", "repeated_stopping", "losing_thread",
                 "recovery", "eye_contact", "hand_movement", "voice_confidence",
                 "observable_nervousness", "significant_hesitation", "heavy_reading",
                 "normal_hesitation"]
Reduction = Literal["task_completion", "repeated_stopping", "losing_thread", "recovery",
                    "eye_contact", "hand_movement", "voice_confidence",
                    "observable_nervousness", "significant_hesitation", "heavy_reading"]
Flag = Literal["VIDEO_TECHNICAL_FAILURE", "PROMPT_NOT_ANSWERED", "TRANSCRIPT_LOW_QUALITY",
               "POSSIBLE_HEAVY_READING", "POSSIBLE_TEMPLATE_RESPONSE",
               "CONTENT_SAFETY_REVIEW", "UNSUPPORTED_LANGUAGE"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(StrictModel):
    kind: Literal["speech", "observation"]
    text: str = Field(min_length=1)
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(ge=0, allow_inf_nan=False)
    supports: list[Signal] = Field(min_length=1)


class Parameter(StrictModel):
    level: int = Field(ge=0, le=8)
    evidence: list[Evidence] = Field(min_length=1)
    reason: str = Field(min_length=1)
    reduction_basis: list[Reduction]


class Parameters(StrictModel):
    SA: Parameter
    Goals: Parameter
    Purpose: Parameter
    Clarity: Parameter
    Voice: Parameter
    Composure: Parameter


class VideoResponse(StrictModel):
    status: Literal["evaluated", "technical_failure", "unsupported_language", "recognition_failure"]
    primary_language: str = Field(min_length=1)
    overall_flags: list[Flag]
    parameters: Parameters | None
    review_reason: str
    review_evidence: list[Evidence]


def gemini_schema():
    """The installed API rejects additionalProperties; backend stays strict."""
    def compatible(value):
        if isinstance(value, dict):
            return {key: compatible(item) for key, item in value.items()
                    if key != "additionalProperties"}
        if isinstance(value, list):
            return [compatible(item) for item in value]
        return value
    return compatible(VideoResponse.model_json_schema())


# Conservative text screening is a review gate, not proof of absence of bias.
PROHIBITED = re.compile(
    r"\b(?:beauty|beautiful|pretty|attractiv\w*|facial symmetry|skin (?:colou?r|tone)|"
    r"complexion|eye colou?r|body (?:shape|appearance)|height|weight|fitness|makeup|"
    r"hairstyle|clothing|jewell?ery|wealth|camera quality|phone quality|room appearance|"
    r"caste|religion|social status|prestigi\w*|follower count|elite (?:college|employer))\b"
    r"|खूबसूरत|सुंदर|त्वचा का रंग|गोरी|सांवली", re.IGNORECASE)
READING_PATTERN = re.compile(r"repeated|sustained|throughout|pattern|several|twice|multiple", re.I)
VAGUE_OBSERVATION = re.compile(
    r"(?:the candidate |candidate |she )?(?:was |is |appeared |seemed )?"
    r"(?:nervous|hesitant|confident|reading|reading from a script)[.! ]*", re.I)


def build_prompt():
    rubric = load_rubric()
    # Do not send historical transcription pipeline, arithmetic or Round-1 weights.
    selected = {key: rubric[key] for key in (
        "version", "video_prompts", "context_guardrail", "parameters",
        "prompt_to_parameter_evidence_map", "prohibited_judging_factors",
        "non_scoring_flags", "scoring_related_flags", "confirmed_project_decisions",
        "ambiguities_for_bd")}
    for value in selected["parameters"].values():
        value.pop("maximum_marks")  # AI selects levels; V1 alone reads maxima.
    return PROMPT.read_text(encoding="utf-8-sig").replace(
        "{rubric_json}", json.dumps(selected, ensure_ascii=False, indent=2))


def failure(category, reason):
    return {"status": "unable_to_evaluate", "overall_flags": [], "parameters": None,
            "marks": None, "total": None, "review_required": True,
            "technical_retry_available": False, "error_category": category, "reason": reason}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def check_evidence(items, *, scoring=True):
    for item in items:
        if not item.text.strip() or item.end_seconds < item.start_seconds:
            raise ValueError("Empty evidence or invalid timestamp interval")
        if scoring and PROHIBITED.search(item.text):
            raise ValueError("Prohibited scoring evidence; human review required")
        if item.kind == "speech" and item.supports != ["speech"]:
            raise ValueError("Speech evidence requires speech support only")
        if item.kind == "observation" and "speech" in item.supports:
            raise ValueError("Observable evidence requires observable support labels")
        if item.kind == "observation" and VAGUE_OBSERVATION.fullmatch(item.text.strip()):
            raise ValueError("Observable evidence must describe concrete behaviour")


def validate_response(raw, *, technical_attempt=1):
    """Validate structure and safety gates; cannot independently verify video truth."""
    if type(technical_attempt) is not int or technical_attempt not in (1, 2):
        raise ValueError("Only an original attempt and one technical retry are allowed")
    try:
        parsed = VideoResponse.model_validate(json.loads(raw, object_pairs_hook=unique_object))
        if not parsed.primary_language.strip() or (
                not parsed.review_reason.strip() and parsed.status != "evaluated"):
            raise ValueError("Review requires a concrete reason and primary language")
        # Technical review can mention recording quality; it is never merit evidence.
        check_evidence(parsed.review_evidence, scoring=False)
        result = parsed.model_dump()
        result.update(marks=None, total=None, review_required=False, technical_retry_available=False)
        flags = set(parsed.overall_flags)
        if parsed.status != "evaluated":
            if parsed.parameters is not None or not parsed.review_evidence:
                raise ValueError("Non-evaluated response requires null parameters and review evidence")
            expected = {"technical_failure": "VIDEO_TECHNICAL_FAILURE",
                        "unsupported_language": "UNSUPPORTED_LANGUAGE",
                        "recognition_failure": "TRANSCRIPT_LOW_QUALITY"}[parsed.status]
            if expected not in flags:
                raise ValueError("Missing status-specific review flag")
            if parsed.status == "unsupported_language" and parsed.primary_language in (
                    load_rubric()["confirmed_project_decisions"]["languages"]["supported"]):
                raise ValueError("Supported language cannot be flagged as unsupported")
            result["review_required"] = True
            if parsed.status == "technical_failure":
                result["status"] = "retry_required" if technical_attempt == 1 else "unable_to_evaluate"
                result["technical_retry_available"] = technical_attempt == 1
            else:
                result["status"] = "needs_review"
            return result
        if parsed.parameters is None:
            raise ValueError("All six parameters are required")
        if parsed.review_reason or parsed.review_evidence:
            raise ValueError("Evaluated response must not include failure review evidence")
        levels, reading = {}, []
        for code, parameter in parsed.parameters:
            if not parameter.reason.strip() or PROHIBITED.search(parameter.reason):
                raise ValueError("Missing or prohibited reasoning")
            check_evidence(parameter.evidence)
            if code != "Composure" and parameter.level > 0 and not any(
                    item.kind == "speech" for item in parameter.evidence):
                raise ValueError("Nonzero content level requires evidence from actual speech")
            observations = [e for e in parameter.evidence if e.kind == "observation"]
            for basis in parameter.reduction_basis:
                if not any(basis in e.supports for e in observations):
                    raise ValueError("Delivery reduction lacks matching observable evidence")
            reading.extend(e for e in observations if "heavy_reading" in e.supports)
            if "heavy_reading" in parameter.reduction_basis and not any(
                    "heavy_reading" in e.supports and READING_PATTERN.search(e.text)
                    for e in observations):
                raise ValueError("Reading reduction requires repeated/sustained evidence")
            levels[code] = parameter.level
        if "POSSIBLE_HEAVY_READING" in flags and not any(READING_PATTERN.search(e.text) for e in reading):
            raise ValueError("Reading flag requires repeated/sustained evidence")
        supported = load_rubric()["confirmed_project_decisions"]["languages"]["supported"]
        if parsed.primary_language not in supported:
            result.update(status="needs_review", review_required=True,
                          overall_flags=sorted(flags | {"UNSUPPORTED_LANGUAGE"}), parameters=None)
            return result
        if flags & {"VIDEO_TECHNICAL_FAILURE", "TRANSCRIPT_LOW_QUALITY", "UNSUPPORTED_LANGUAGE"}:
            raise ValueError("Unassessable flags require the corresponding non-evaluated status")
        marks, total = score_video(levels)
        result.update(marks={code: float(value) for code, value in marks.items()},
                      total=float(total), review_required="CONTENT_SAFETY_REVIEW" in flags)
        return result
    except (ValueError, TypeError):
        # No raw invalid output/exception details: they could contain secrets.
        return failure("invalid_response", "Invalid, unsupported or prohibited AI output; human review required")


def evaluate_video(client, video, model, *, technical_attempt=1, processing_timeout=120, sleep=time.sleep):
    """Upload unchanged video, wait for ACTIVE, evaluate once, always attempt deletion."""
    if processing_timeout <= 0 or not math.isfinite(processing_timeout):
        raise ValueError("Processing timeout must be finite and positive")
    uploaded_name = None
    result = failure("api_error", "Gemini request failed")
    stage = "upload"
    try:
        uploaded = client.files.upload(file=video, config=types.UploadFileConfig(
            mime_type=mimetypes.guess_type(str(video))[0] or "video/mp4"))
        uploaded_name = uploaded.name
        if not uploaded_name:
            raise ValueError("Upload returned no file name")
        deadline = time.monotonic() + processing_timeout
        stage = "processing"
        while uploaded.state == types.FileState.PROCESSING:
            if time.monotonic() >= deadline:
                raise TimeoutError()
            sleep(min(2, max(0, deadline - time.monotonic())))
            uploaded = client.files.get(name=uploaded_name)
        if uploaded.state != types.FileState.ACTIVE:
            raise RuntimeError("Uploaded file did not become active")
        stage = "generation"
        response = client.models.generate_content(
            model=model, contents=[uploaded], config=types.GenerateContentConfig(
                system_instruction=build_prompt(), temperature=0,
                response_mime_type="application/json", response_schema=gemini_schema()))
        result = validate_response(response.text or "", technical_attempt=technical_attempt)
    except Exception as error:
        result = failure("api_error", "Gemini upload, processing or evaluation failed; no score assigned")
        result["api_failure_stage"] = stage
        code = getattr(error, "code", None)
        if type(code) is int and 100 <= code <= 599:
            result["http_status"] = code  # Never expose SDK message/URL/request details.
    finally:
        result["upload_cleanup"] = "not_uploaded"
        if uploaded_name:
            try:
                client.files.delete(name=uploaded_name)
                result["upload_cleanup"] = "deleted"
            except Exception:
                result["upload_cleanup"] = "failed"
                result["review_required"] = True
                result["remote_file_to_delete"] = uploaded_name
    return result


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {redact(name, key): redact(item, key) for name, item in value.items()}
    return value


def score_contestant(candidate_id, video, model, *, retry_of=None, processing_timeout=120):
    """Append a run; a referenced technical failure can be retried once."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate_id):
        raise ValueError("Candidate ID must contain letters, numbers, underscores or hyphens")
    video = Path(video).resolve()
    if not video.is_file() or video.stat().st_size == 0 or video.suffix.lower() not in {
            ".mp4", ".mov", ".webm", ".avi", ".mpeg", ".mpg", ".m4v", ".3gp"}:
        raise ValueError("Supply an existing nonempty original video file")
    if not model.strip() or processing_timeout <= 0 or not math.isfinite(processing_timeout):
        raise ValueError("Model required; processing timeout must be finite and positive")
    folder = ROOT / "outputs" / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".video_ai.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Candidate has an active run or stale .video_ai.lock; inspect before retrying") from None
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    path = None
    try:
        attempt = 1
        if retry_of is not None:
            if type(retry_of) is not int or retry_of < 1:
                raise ValueError("--retry-of must reference a positive run number")
            try:
                prior = json.loads((folder / f"video_ai_run{retry_of}.json").read_text(encoding="utf-8"))
                history = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("video_ai_run*.json")]
            except (OSError, ValueError):
                raise ValueError("Cannot read referenced retry history") from None
            if (prior.get("candidate_id") != candidate_id or prior.get("status") != "retry_required"
                    or prior.get("technical_attempt") != 1 or not prior.get("technical_retry_available")
                    or any(item.get("retry_of") == retry_of for item in history)):
                raise ValueError("Only one retry of an original technical failure is allowed")
            attempt = 2
        try:
            key = dotenv_values(ROOT / ".env", interpolate=False).get("GEMINI_API_KEY")
        except Exception:
            raise ValueError("Could not load GEMINI_API_KEY from .env") from None
        if not key:
            raise ValueError("GEMINI_API_KEY missing from .env")
        prompt_hash = hashlib.sha256(build_prompt().encode("utf-8")).hexdigest()
        run = 1
        while True:
            path = folder / f"video_ai_run{run}.json"
            try:
                handle = path.open("x", encoding="utf-8")
                break
            except FileExistsError:
                run += 1
        with handle:
            result = None
            try:
                with genai.Client(api_key=key, http_options=types.HttpOptions(
                        timeout=60000, retry_options=types.HttpRetryOptions(attempts=1))) as client:
                    result = evaluate_video(client, video, model, technical_attempt=attempt,
                                            processing_timeout=processing_timeout)
            except Exception:
                cleanup = {k: result[k] for k in ("upload_cleanup", "remote_file_to_delete")
                           if isinstance(result, dict) and k in result}
                result = failure("api_error", "Gemini client setup or teardown failed; no score assigned")
                result.update(cleanup)
            result.update(candidate_id=candidate_id, run=run, model=model, retry_of=retry_of,
                          technical_attempt=attempt, video_rubric_version=load_rubric()["version"],
                          rubric_sha256=hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest(),
                          prompt_sha256=prompt_hash, prompt_file="prompts/score_video_v1.txt",
                          run_date=datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
                          experimental=True, official_score=False)
            result = redact(result, key)
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return result, path
    finally:
        if path is not None and path.exists() and path.stat().st_size == 0:
            path.unlink()
        logging.disable(previous_logging)
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--retry-of", type=int, help="Original run awaiting its single technical retry")
    parser.add_argument("--processing-timeout", type=float, default=120,
                        help="Upload wait in seconds; never a candidate duration limit")
    args = parser.parse_args(argv)
    try:
        result, path = score_contestant(args.candidate_id, args.video, args.model,
                                       retry_of=args.retry_of, processing_timeout=args.processing_timeout)
    except Exception:
        print("Error: V3 could not run. Check input, .env key, model, retry history and output access.")
        return 1
    print(f"Saved {path.name}; status={result['status']}")
    if result.get("api_failure_stage"):
        print(f"API failure stage: {result['api_failure_stage']}; HTTP status: {result.get('http_status', 'unavailable')}")
    if result["total"] is not None:
        maximum = sum(p["maximum_marks"] for p in load_rubric()["parameters"].values())
        print(f"Exploratory video total: {result['total']:.2f} / {maximum:.2f}")
    if result["technical_retry_available"]:
        print(f"One technical retry available: --retry-of {result['run']}")
    if result.get("upload_cleanup") == "failed":
        print("Remote cleanup failed; saved result identifies the file needing deletion.")
    return 0 if result["status"] == "evaluated" and not result["review_required"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
