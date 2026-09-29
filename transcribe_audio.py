import argparse
from datetime import datetime
import logging
from pathlib import Path
import re
import sys

from dotenv import dotenv_values
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.8-flash"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--run-tag")
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.only is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.only):
        parser.error("--only must be a candidate folder name")
    if args.only is not None and not (ROOT / "outputs" / args.only).is_dir():
        parser.error(f"Candidate output folder does not exist: {args.only}")
    if args.run_tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("--run-tag must contain only letters, numbers, underscores or hyphens")
    suffix = f"_{args.run_tag}" if args.run_tag is not None else ""
    prompt = args.prompt_file.read_text(encoding="utf-8")
    # Suppress library logging so credentials and .env parse details stay private.
    logging.disable(logging.CRITICAL)
    try:
        api_key = dotenv_values(ROOT / ".env", interpolate=False).get("GEMINI_API_KEY")
    except Exception:
        print("ERROR: Could not load GEMINI_API_KEY from .env.")
        return 1
    if not api_key:
        print("ERROR: GEMINI_API_KEY is missing or empty in .env.")
        return 1

    failed = False
    with genai.Client(api_key=api_key) as client:
        folders = [ROOT / "outputs" / args.only] if args.only is not None else sorted((ROOT / "outputs").iterdir())
        for folder in folders:
            if not folder.is_dir():
                continue
            try:
                destination = folder / f"transcript_gemini_v2{suffix}.txt"
                if destination.exists():
                    raise FileExistsError(f"Will not overwrite {destination}")
                audio = folder / "audio.wav"
                uploaded = client.files.upload(
                    file=audio,
                    config=types.UploadFileConfig(mime_type="audio/wav"),
                )
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[prompt, uploaded],
                    config=types.GenerateContentConfig(temperature=0),
                )
                transcript = response.text
                if not transcript:
                    raise RuntimeError("Gemini returned no transcript text.")
                run_date = datetime.now().astimezone().isoformat(timespec="seconds")
                with destination.open("x", encoding="utf-8") as output:
                    output.write(
                        f"Model: {MODEL}\nRun date: {run_date}\n"
                        f"Prompt file: {args.prompt_file.name}\n\n{transcript}"
                    )
                print(f"{folder.name}: saved {destination}")
            except Exception as exc:
                failed = True
                error = str(exc).replace(api_key, "[REDACTED]")
                print(f"FAILED {folder.name}: {type(exc).__name__}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
