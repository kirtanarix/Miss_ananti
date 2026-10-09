"""PDF Section 7 pilot: audio transcription, transcript content scoring, video Composure."""

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
RUBRIC_PATH = ROOT / "config" / "rubric_video_fullPDF_v1.0.yaml"
TRANSCRIBE_PROMPT = ROOT / "prompts" / "transcribe_v2.txt"
CONTENT_PROMPT = ROOT / "prompts" / "score_video_content_fullPDF_v1.txt"
COMPOSURE_PROMPT = ROOT / "prompts" / "score_video_composure_fullPDF_v1.txt"
CODES = ("SA", "Goals", "Purpose", "Clarity", "Voice", "Composure")
MODEL = "gemini-3.8-flash"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TranscriptEvidence(Strict):
    line: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ContentParameter(Strict):
    level: int = Field(ge=0, le=8)
    evidence: list[str]
    transcript_evidence: list[TranscriptEvidence]
    absence_observation: str
    reason: str = Field(min_length=1)
    next_anchor_gap: str


class ContentParameters(Strict):
    SA: ContentParameter
    Goals: ContentParameter
    Purpose: ContentParameter
    Clarity: ContentParameter
    Voice: ContentParameter


class ContentResponse(Strict):
    status: Literal["evaluated", "unsupported_language", "recognition_failure"]
    primary_language: str = Field(min_length=1)
    parameters: ContentParameters | None
    flags: list[Literal["PROMPT_NOT_ANSWERED", "POSSIBLE_TEMPLATE_RESPONSE"]]
    flag_reasons: dict[str, str]
    review_reason: str


class Observation(Strict):
    text: str = Field(min_length=1)
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(ge=0, allow_inf_nan=False)
    level_support: str = Field(min_length=1)


class ComposureResponse(Strict):
    status: Literal["evaluated", "technical_failure", "recognition_failure"]
    level: int | None = Field(ge=0, le=8)
    evidence: list[Observation]
    reason: str
    next_anchor_gap: str
    flags: list[Literal["POSSIBLE_HEAVY_READING"]]
    flag_reasons: dict[str, str]
    review_reason: str


def rubric():
    text = RUBRIC_PATH.read_text(encoding="utf-8")
    return json.loads("\n".join(line for line in text.splitlines()
                            if not line.lstrip().startswith("#")))


def schema(model):
    def compatible(value):
        if isinstance(value, dict):
            return {key: compatible(item) for key, item in value.items()
                    if key != "additionalProperties"}
        if isinstance(value, list):
            return [compatible(item) for item in value]
        return value
    return compatible(model.model_json_schema())


def prompt(path, selected):
    template = path.read_text(encoding="utf-8-sig")
    return template.replace("{rubric_json}", json.dumps(selected, ensure_ascii=False, indent=2))


def content_prompt(r):
    return prompt(CONTENT_PROMPT, {
        "video_prompts": r["video_prompts"],
        "context_guardrail": r["context_guardrail"],
        "parameters": {code: {key: value for key, value in r["parameters"][code].items()
                              if key != "maximum_marks"} for code in CODES[:5]},
        "prompt_to_parameter_evidence_map": {
            "columns": r["prompt_to_parameter_evidence_map"]["columns"][:6],
            "rows": [row[:6] for row in r["prompt_to_parameter_evidence_map"]["rows"]],
            "guidance": r["prompt_to_parameter_evidence_map"]["guidance"],
        },
        "prohibited_judging_factors": r["prohibited_judging_factors"],
        "non_scoring_flags": {key: r["non_scoring_flags"][key] for key in
                              ("PROMPT_NOT_ANSWERED", "POSSIBLE_TEMPLATE_RESPONSE")},
    })


def composure_prompt(r):
    return prompt(COMPOSURE_PROMPT, {
        "video_prompts": r["video_prompts"],
        "parameter": {key: value for key, value in r["parameters"]["Composure"].items()
                      if key != "maximum_marks"},
        "prohibited_judging_factors": r["prohibited_judging_factors"],
        "non_scoring_flags": {
            "POSSIBLE_HEAVY_READING": r["non_scoring_flags"]["POSSIBLE_HEAVY_READING"]},
    })


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def duration_seconds(path):
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(completed.stdout.strip())


def extract_audio(video, destination):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required")
    completed = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination)], capture_output=True, text=True)
    if completed.returncode or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed; no evaluation was scored")


def generate(client, model, contents, instruction, response_model=None):
    settings = {"system_instruction": instruction, "temperature": 0}
    if response_model:
        settings.update(response_mime_type="application/json",
                        response_schema=schema(response_model))
    response = client.models.generate_content(
        model=model, contents=contents, config=types.GenerateContentConfig(**settings))
    if not response.text or not response.text.strip():
        raise ValueError("Gemini returned no usable text")
    return response.text


def check_flags(flags, reasons):
    # Optional verification flags without an explanation are omitted, not scored.
    unique = list(dict.fromkeys(flags))
    kept = [flag for flag in unique if reasons.get(flag, "").strip()]
    return kept, {flag: reasons[flag].strip() for flag in kept}


def check_content(result, transcript):
    if result.status != "evaluated":
        if result.parameters is not None or not result.review_reason.strip():
            raise ValueError("Unassessable content requires null parameters and review reason")
        return
    if result.primary_language not in {"English", "Hindi", "Hinglish"}:
        raise ValueError("Unsupported primary language cannot be scored")
    if result.parameters is None or result.review_reason:
        raise ValueError("Evaluated content needs all five parameters and no review reason")
    result.flags, result.flag_reasons = check_flags(result.flags, result.flag_reasons)
    for code, parameter in result.parameters:
        if parameter.level > 0 and not parameter.evidence:
            raise ValueError(f"{code}: nonzero level needs transcript evidence")
        if parameter.level > 0 and not parameter.transcript_evidence:
            raise ValueError(f"{code}: nonzero level needs exact transcript lines and explanations")
        if parameter.level == 0 and not parameter.evidence and not parameter.absence_observation.strip():
            raise ValueError(f"{code}: level zero needs an observable absence")
        if parameter.level < 8 and not parameter.next_anchor_gap.strip():
            raise ValueError(f"{code}: next anchor gap is required")
        if parameter.level > 0 and parameter.absence_observation:
            raise ValueError(f"{code}: absence observation is for level zero only")
        if any(not quote.strip() or quote not in transcript for quote in parameter.evidence):
            raise ValueError(f"{code}: evidence phrase is not verbatim in the saved transcript")
        for item in parameter.transcript_evidence:
            if item.line not in transcript:
                raise ValueError(f"{code}: transcript line is not verbatim in the saved transcript")
            if not any(quote in item.line for quote in parameter.evidence):
                raise ValueError(f"{code}: transcript line does not contain its evidence phrase")


def check_composure(result, duration):
    if result.status != "evaluated":
        if result.level is not None or not result.review_reason.strip():
            raise ValueError("Unassessable Composure requires null level and review reason")
        return
    if result.level is None or not result.evidence or not result.reason.strip() or result.review_reason:
        raise ValueError("Composure requires level, observable evidence and reasoning")
    if result.level < 8 and not result.next_anchor_gap.strip():
        raise ValueError("Composure next anchor gap is required")
    result.flags, result.flag_reasons = check_flags(result.flags, result.flag_reasons)
    for item in result.evidence:
        if not item.text.strip() or item.end_seconds < item.start_seconds:
            raise ValueError("Composure evidence has an invalid time interval")
        if not item.level_support.strip():
            raise ValueError("Each Composure observation needs an explanation of its scoring relevance")
        if item.end_seconds > duration + 1:
            raise ValueError("Composure evidence time exceeds recording duration")


def calculate(levels, r):
    if set(levels) != set(CODES):
        raise ValueError("Exactly six parameters are required")
    marks = {}
    for code in CODES:
        level = levels[code]
        if type(level) is not int or not 0 <= level <= 8:
            raise ValueError(f"{code}: integer level 0..8 required")
        marks[code] = Decimal(level) / 8 * Decimal(str(r["parameters"][code]["maximum_marks"]))
    if sum(Decimal(str(value["maximum_marks"])) for value in r["parameters"].values()) != 20:
        raise ValueError("PDF parameter maxima do not sum to 20")
    return marks, sum(marks.values(), Decimal(0))


def run(candidate, video, model=MODEL, reuse_transcript=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise ValueError("Invalid candidate ID")
    video = Path(video).resolve()
    if not video.is_file() or video.suffix.lower() not in {".mp4", ".mov", ".webm"}:
        raise ValueError("Original video file is required")
    key = dotenv_values(ROOT / ".env", interpolate=False).get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY missing from .env")
    folder = ROOT / "outputs" / candidate
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".video_fullPDF.lock"
    lock.mkdir()
    try:
        run_number = 1
        while any((folder / name).exists() for name in (
                f"video_fullPDF_run{run_number}.json",
                f"video_fullPDF_audio_run{run_number}.wav",
                f"video_fullPDF_transcript_run{run_number}.txt")):
            run_number += 1
        audio = folder / f"video_fullPDF_audio_run{run_number}.wav"
        transcript_file = folder / f"video_fullPDF_transcript_run{run_number}.txt"
        output = folder / f"video_fullPDF_run{run_number}.json"
        r = rubric()
        instruction_content = content_prompt(r)
        instruction_composure = composure_prompt(r)
        instruction_stt = TRANSCRIBE_PROMPT.read_text(encoding="utf-8-sig")
        result = {
            "candidate_id": candidate, "run": run_number, "status": "unable_to_evaluate",
            "model": model, "video_rubric_version": r["version"],
            "rubric_sha256": sha_bytes(RUBRIC_PATH.read_bytes()),
            "prompt_sha256": {"transcription": sha_bytes(instruction_stt.encode()),
                              "content": sha_bytes(instruction_content.encode()),
                              "composure": sha_bytes(instruction_composure.encode())},
            "source_video_sha256": sha_bytes(video.read_bytes()),
            "transcript_path": None, "transcript_sha256": None,
            "audio_path": None, "audio_sha256": None, "parameters": None,
            "marks": None, "total": None, "flags": [], "flag_reasons": {},
            "review_required": True, "decision_history": [], "experimental": True,
            "official_score": False,
            "run_date": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
        }
        stage = "audio_extraction"
        raw_content = None
        raw_composure = None
        try:
            video_duration = duration_seconds(video)
            logging.disable(logging.CRITICAL)
            with genai.Client(api_key=key, http_options=types.HttpOptions(
                    timeout=300000, retry_options=types.HttpRetryOptions(attempts=1))) as client:
                if reuse_transcript is None:
                    stage = "audio_extraction"
                    extract_audio(video, audio)
                    result["audio_path"] = str(audio.relative_to(ROOT))
                    result["audio_sha256"] = sha_bytes(audio.read_bytes())
                    result["decision_history"].append({"stage": "audio_extraction", "input": "original_video", "output": result["audio_path"]})
                    stage = "audio_transcription"
                    raw_transcript = generate(client, model, [types.Part.from_bytes(
                        data=audio.read_bytes(), mime_type="audio/wav")], instruction_stt)
                    transcript = raw_transcript.strip()
                    if not transcript:
                        raise ValueError("Empty transcription")
                    with transcript_file.open("x", encoding="utf-8") as stream:
                        stream.write(transcript + "\n")
                    result["transcript_path"] = str(transcript_file.relative_to(ROOT))
                    result["decision_history"].append({"stage": "transcription", "input": "audio_only", "output": result["transcript_path"], "model": model})
                else:
                    prior_transcript = Path(reuse_transcript).resolve()
                    if not prior_transcript.is_file() or ROOT not in prior_transcript.parents:
                        raise ValueError("Reusable transcript must be an existing project output")
                    transcript = prior_transcript.read_text(encoding="utf-8").strip()
                    if not transcript:
                        raise ValueError("Reusable transcript is empty")
                    result["transcript_path"] = str(prior_transcript.relative_to(ROOT))
                    result["decision_history"].append({"stage": "transcription_reuse", "input": "previous_audio_only_transcription", "output": result["transcript_path"]})
                transcript_bytes = (prior_transcript.read_bytes() if reuse_transcript is not None
                                    else transcript_file.read_bytes())
                result["transcript_sha256"] = sha_bytes(transcript_bytes)

                stage = "content_scoring"
                raw_content = generate(client, model, transcript, instruction_content, ContentResponse)
                content = ContentResponse.model_validate_json(raw_content)
                check_content(content, transcript)
                result["decision_history"].append({"stage": "content_scoring", "input": "transcript_text_only", "parameters": list(CODES[:5]), "model": model, "status": content.status})
                if content.status != "evaluated":
                    result.update(status="needs_review", review_reason=content.review_reason,
                                  flags=["UNSUPPORTED_LANGUAGE" if content.status == "unsupported_language" else "TRANSCRIPT_LOW_QUALITY"])
                else:
                    stage = "composure_scoring"
                    media = types.Part.from_bytes(data=video.read_bytes(), mime_type="video/mp4")
                    raw_composure = generate(client, model, [media], instruction_composure, ComposureResponse)
                    composure = ComposureResponse.model_validate_json(raw_composure)
                    check_composure(composure, video_duration)
                    result["decision_history"].append({"stage": "composure_scoring", "input": "original_video_only", "parameters": ["Composure"], "model": model, "status": composure.status})
                    if composure.status != "evaluated":
                        result.update(status="needs_review", review_reason=composure.review_reason,
                                      flags=["VIDEO_TECHNICAL_FAILURE" if composure.status == "technical_failure" else "TRANSCRIPT_LOW_QUALITY"])
                    else:
                        levels = {code: getattr(content.parameters, code).level for code in CODES[:5]}
                        levels["Composure"] = composure.level
                        marks, total = calculate(levels, r)
                        result["parameters"] = {code: getattr(content.parameters, code).model_dump()
                                                for code in CODES[:5]}
                        result["parameters"]["Composure"] = {
                            "level": composure.level,
                            "evidence": [item.model_dump() for item in composure.evidence],
                            "reason": composure.reason,
                            "next_anchor_gap": composure.next_anchor_gap,
                        }
                        result.update(status="evaluated", marks={k: float(v) for k, v in marks.items()},
                                      total=float(total), review_required=False,
                                      flags=content.flags + composure.flags,
                                      flag_reasons={**content.flag_reasons, **composure.flag_reasons})
                        result["decision_history"].append({"stage": "calculation", "input": "six_validated_levels", "formula": "level/8*maximum_marks_from_PDF_config", "total": float(total)})
        except Exception as error:
            result.update(error_stage=stage, error_type=type(error).__name__,
                          error_detail=str(error).replace(key, "[REDACTED]")[:500])
            if stage == "content_scoring" and raw_content is not None:
                result["unvalidated_content_response"] = raw_content.replace(key, "[REDACTED]")
            if stage == "composure_scoring" and raw_composure is not None:
                result["unvalidated_composure_response"] = raw_composure.replace(key, "[REDACTED]")
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return result, output
    finally:
        lock.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reuse-transcript", type=Path,
                        help="Reuse a prior saved audio-only transcript; content still receives transcript text only")
    args = parser.parse_args(argv)
    try:
        result, output = run(args.candidate_id, args.video, args.model, args.reuse_transcript)
    except Exception as error:
        parser.exit(1, f"Cannot start evaluation: {type(error).__name__}\n")
    print(f"Saved {output.name}; status={result['status']}")
    if result["status"] == "evaluated":
        print(f"Video total: {result['total']:.2f} / 20")
    elif result.get("error_stage"):
        print(f"Failure stage: {result['error_stage']}; type: {result['error_type']}")
    return 0 if result["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
