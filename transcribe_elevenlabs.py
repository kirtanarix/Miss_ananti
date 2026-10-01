import argparse
from datetime import datetime
import errno
import logging
from pathlib import Path
import sys

from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs


ROOT = Path(__file__).resolve().parent
MODEL = "scribe_v2"
KEYTERM = "Miss Ananti India"


def main():
    parser = argparse.ArgumentParser(description="Transcribe candidate audio with ElevenLabs.")
    parser.add_argument("--only", help="Process one candidate folder")
    parser.add_argument("--keyterm", action="store_true", help="Use Miss Ananti India as a key term")
    args = parser.parse_args()
    outputs = ROOT / "outputs"
    if args.only is not None and (
        args.only in ("", ".", "..") or any(c in args.only for c in '/\\:')
    ):
        parser.error("--only must be a candidate folder name")
    if not outputs.is_dir():
        parser.error("outputs/ does not exist")
    folders = [outputs / args.only] if args.only is not None else sorted(outputs.iterdir())
    folders = [folder for folder in folders if folder.is_dir() and (folder / "audio.wav").is_file()]
    if not folders:
        parser.error("No candidate folders with audio.wav found")

    # Prevent SDK and dotenv diagnostics from exposing credentials or .env contents.
    logging.disable(logging.CRITICAL)
    try:
        api_key = dotenv_values(ROOT / ".env", interpolate=False).get("ELEVENLABS_API_KEY")
    except Exception:
        print("ERROR: Could not load ELEVENLABS_API_KEY from .env.")
        return 1
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY is missing or empty in .env.")
        return 1

    client = ElevenLabs(api_key=api_key)
    options = {"keyterms": [KEYTERM]} if args.keyterm else {}
    version = "v2" if args.keyterm else "v1"
    failed = False
    for folder in folders:
        try:
            destination = folder / f"transcript_elevenlabs_{version}.txt"
            if destination.exists():
                raise FileExistsError(f"Will not overwrite {destination}")
            print(f"{folder.name}: transcribing ({version})...", flush=True)
            with (folder / "audio.wav").open("rb") as audio:
                response = client.speech_to_text.convert(
                    file=audio,
                    model_id=MODEL,
                    diarize=False,
                    tag_audio_events=False,
                    no_verbatim=False,
                    **options,
                )
            run_date = datetime.now().astimezone().isoformat(timespec="seconds")
            header = (
                f"Model: {MODEL}\nRun date: {run_date}\n"
                f"Key term used: {KEYTERM if args.keyterm else 'No'}\n"
                f"Detected language: {response.language_code}\n"
                f"Language probability: {response.language_probability}\n\n"
            )
            with destination.open("x", encoding="utf-8", newline="") as output:
                output.write(header)
                output.write(response.text)
            print(f"{folder.name}: saved {destination}", flush=True)
        except Exception as exc:
            failed = True
            error = str(exc).replace(api_key, "[REDACTED]")
            print(f"FAILED {folder.name}: {type(exc).__name__}: {error}", flush=True)
            if isinstance(exc, OSError) and (
                exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in (39, 112)
            ):
                print("Disk space exhausted; stopping without retrying.", flush=True)
                return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
