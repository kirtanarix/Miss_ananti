# Project: Miss Ananti India Round-1 AI screening

## Current stage
ONLY: video -> audio -> Gemini transcript -> compare with my manual check.
No scoring, no rubric, no segmentation.

## Hard rules
- Send AUDIO ONLY to Gemini, never video. No analysis of face, body, clothing,
  background, expression, ever.
- API key only from .env (GEMINI_API_KEY). Never print, log or commit it.
- Never commit data/, outputs/, .env, videos, audio, transcripts.
- Transcription must be verbatim: no translation, no grammar fixing, no summary.
- One simple script per step, output saved to outputs/<candidate_id>/.
- Do not add features I did not ask for.

## Environment setup rules
- Before running anything, check that required tools exist (e.g. ffmpeg, python).
  If something is missing, install it using the method for this OS, then tell me
  exactly what you installed.
- Python packages: use a virtual environment in .venv and list every package in
  requirements.txt. Install only what the current task needs.
- If an install needs admin/sudo and fails, stop and give me the exact command
  instead of working around it.
- Never open, print, or log the contents of .env.
- Never add .venv/ to git.

## Identity match experiment (Task 7)
- This is an experiment only, not an approved BD feature. Result must NEVER affect any
  score; at most it produces a review flag.
- Face images/frames are used ONLY to answer "same person or not". No description or
  judgement of appearance, attractiveness, skin tone, age, body, makeup, clothing, background.
- Send only the extracted still frames and the candidate image, never the full video.
- Extracted frames and results stay in outputs/ (gitignored). Never commit any image/frame.
- Key from .env, never print it.