"""
Identity check EXPERIMENT: does the person in the submitted video match the submitted image?

Rules for this script (from project AGENTS.md):
- Result NEVER affects any score. It only produces a raw decision + a human-review column.
- Only answers "same person or not". No description/judgement of appearance.
- Key comes from .env (GEMINI_API_KEY). It is never printed or logged.
- Results go to outputs/ (gitignored). Never commit videos, images or results.

Expected layout:
    data/C001/Video_001.mp4   data/C001/Image_001.jfif   (any name starting with "video" / "image";
    data/C002/...                                        image may be .jpg .jpeg .png .jfif)

Examples (PowerShell, from project root):
    # genuine pairs only (video X vs image X)
    .\\.venv\\Scripts\\python.exe .\\identity_check_gemini.py --model MODEL_NAME

    # full test: every video vs every image (genuine + impostor pairs)
    .\\.venv\\Scripts\\python.exe .\\identity_check_gemini.py --model MODEL_NAME --all-pairs

    # one candidate only, repeated 3 times to check stability
    .\\.venv\\Scripts\\python.exe .\\identity_check_gemini.py --model MODEL_NAME --only C003 --repeats 3
"""
import argparse
import csv
import mimetypes
import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROMPT = """You are given two things: one image and one video. Decide only whether they show the same person.
Answer with exactly three lines and nothing else:
Line 1: exactly one of SAME_PERSON, DIFFERENT_PERSON, CANNOT_DETERMINE
Line 2: exactly one of HIGH, MEDIUM, LOW (your confidence)
Line 3: only image or video quality problems that limit the decision (for example: face too small, face turned away, blurry, low light, face covered, no face visible), or NONE
Do not describe or comment on attractiveness, skin tone, age, body, makeup, clothing or background.
If you cannot decide reliably, answer CANNOT_DETERMINE."""

DECISIONS = ("SAME_PERSON", "DIFFERENT_PERSON", "CANNOT_DETERMINE")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
VIDEO_EXTS = {".mp4", ".mov", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".jfif": "image/jpeg", ".png": "image/png"}
CSV_COLUMNS = [
    "video_id", "image_id", "is_same_person_truth", "repeat",
    "decision", "confidence", "quality_problems",
    "needs_human_review", "model", "run_date", "error",
]


def find_candidates(data_dir: Path, only: str | None):
    """Return {candidate_id: (video_path, image_path)}.

    In each candidate folder: the video is a file whose name starts with "video" (e.g. Video_001.mp4),
    the image is a file whose name starts with "image" (e.g. Image_001.jfif / .jpeg / .jpg / .png).
    """
    found = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if only and folder.name != only:
            continue
        files = sorted(f for f in folder.iterdir() if f.is_file())
        video = next((f for f in files if f.suffix.lower() in VIDEO_EXTS and f.stem.lower().startswith("video")), None)
        image = next((f for f in files if f.suffix.lower() in IMAGE_EXTS and f.stem.lower().startswith("image")), None)
        if video is not None and image is not None:
            found[folder.name] = (video, image)
        else:
            print(f"[skip] {folder.name}: need a file starting with 'video' (.mp4) and one starting with 'image' (.jpg/.jpeg/.png/.jfif)")
    return found


def clean(line: str) -> str:
    return line.strip().strip("*`_ .:").upper()


def parse_response(text: str):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    decision = clean(lines[0]) if len(lines) > 0 else ""
    confidence = clean(lines[1]) if len(lines) > 1 else ""
    quality = lines[2].strip() if len(lines) > 2 else ""
    if decision not in DECISIONS:
        decision = "PARSE_ERROR"
    if confidence not in CONFIDENCES:
        confidence = ""
    return decision, confidence, quality


def upload_video(client, path: Path):
    f = client.files.upload(file=str(path))
    while f.state is not None and f.state.name == "PROCESSING":
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state is not None and f.state.name != "ACTIVE":
        raise RuntimeError(f"video upload ended in state {f.state.name}")
    return f


def image_part(path: Path):
    mime = IMAGE_MIME.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Gemini model name (copy it from the header of a transcript_gemini_v2.txt file)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs/identity_gemini_results.csv")
    ap.add_argument("--only", help="run only this candidate's video (e.g. C003)")
    ap.add_argument("--all-pairs", action="store_true",
                    help="compare each video with every candidate's image (genuine + impostor)")
    ap.add_argument("--repeats", type=int, default=1, help="repeat each pair N times (stability check)")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not found in .env")
    client = genai.Client(api_key=api_key)

    all_candidates = find_candidates(Path(args.data_dir), None)
    video_ids = [c for c in all_candidates if not args.only or c == args.only]
    if not video_ids:
        sys.exit("No candidate folders with video + image found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    counts = Counter()

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for vid in video_ids:
            video_path = all_candidates[vid][0]
            image_ids = list(all_candidates) if args.all_pairs else [vid]
            uploaded = None
            try:
                print(f"[upload] video {vid} ...")
                uploaded = upload_video(client, video_path)
            except Exception as e:  # never print the key; error text does not contain it
                print(f"[error] video {vid}: {type(e).__name__}: {e}")
                continue

            try:
                for img_id in image_ids:
                    img_path = all_candidates[img_id][1]
                    truth = vid == img_id
                    for rep in range(1, args.repeats + 1):
                        row = {
                            "video_id": vid, "image_id": img_id, "is_same_person_truth": truth,
                            "repeat": rep, "model": args.model, "run_date": date.today().isoformat(),
                            "error": "",
                        }
                        try:
                            resp = client.models.generate_content(
                                model=args.model,
                                contents=[uploaded, image_part(img_path), PROMPT],
                                config=types.GenerateContentConfig(temperature=0),
                            )
                            decision, confidence, quality = parse_response(resp.text)
                        except Exception as e:
                            decision, confidence, quality = "API_ERROR", "", ""
                            row["error"] = f"{type(e).__name__}: {e}"
                        row.update(decision=decision, confidence=confidence, quality_problems=quality,
                                   needs_human_review=(decision != "SAME_PERSON"))
                        writer.writerow(row)
                        fh.flush()
                        counts[("genuine" if truth else "impostor", decision)] += 1
                        print(f"video {vid} vs image {img_id} (run {rep}): {decision} {confidence}")
            finally:
                if uploaded is not None:
                    try:
                        client.files.delete(name=uploaded.name)  # do not leave candidate video on Google's side
                    except Exception:
                        print(f"[warn] could not delete uploaded video for {vid}; it expires automatically")

    print("\nSummary (counts only, no judgement):")
    for (kind, decision), n in sorted(counts.items()):
        print(f"  {kind:9s} {decision:18s} {n}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
