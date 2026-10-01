# Project: Miss Ananti India Round-1 AI screening

## Current stage
Written scoring (Task 6). Earlier work (transcription, identity) is finished or on hold; its rules below stay valid.

## Hard rules
- For transcription, send AUDIO ONLY to Gemini, never video. No analysis of face, body, clothing,
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
(On hold. Not for this coding agent; the identity script is run by the user only.)
- This is an experiment only, not an approved BD feature. Result must NEVER affect any
  score; at most it produces a review flag.
- Face images/frames are used ONLY to answer "same person or not". No description or
  judgement of appearance, attractiveness, skin tone, age, body, makeup, clothing, background.
- Send only the extracted still frames and the candidate image, never the full video.
- Extracted frames and results stay in outputs/ (gitignored). Never commit any image/frame.
- Key from .env, never print it.

## ElevenLabs STT comparison (Task 8)
- Approved by my supervisor. Audio only (outputs/<id>/audio.wav) may go to ElevenLabs. Never video or images.
- Key only from .env (ELEVENLABS_API_KEY). Never print, log or commit it.
- Same layout as Gemini: outputs/<id>/transcript_elevenlabs_v1.txt (no key terms) and
  transcript_elevenlabs_v2.txt (with key term "Miss Ananti India").
- Names/cities are still context only and never scoring evidence.

## Written scoring (Task 6, active)
Source of truth: docs/Miss_Ananti_Written_Screening_Framework_v1_0.docx (v1.0 Frozen) and
config/rubric_written_v1.0.yaml. Calculator: score_written.py (tests in tests/).
Copy rubric wording exactly. Never invent criteria, weights or thresholds; TODO_BD otherwise.
Python computes all marks; the LLM outputs only levels, evidence phrases, short reasoning and flags.
Written answers in data/<id>/written_answers.json are real sample-candidate text from BD. Never copy them into code, docs or chat.
Do not score grammar, spelling, language, fluency, viewpoint, scale/money/prestige.
Video scoring: on hold until BD confirms the final video questions and duration.

- Task 6: only the question text, the rubric text and the candidate answer go to Gemini (BD approved).
  Judge levels are never shown to the AI. Prompt text lives in prompts/score_written_v1.txt.
- No score without an evidence phrase that appears verbatim in the candidate's answer.
- Never treat missing or invalid AI output as a low score; mark it "unable_to_evaluate".
- Do not decide any pass/fail threshold. No AI-generated-text detection.
- Open points from BD (do not decide them yourself): odd levels (Section 5 allows them, Section 7
  lists only 0/2/4/6/8); "consistent across answers" in level 8 of SA and CV vs "judging only that
  answer"; Section 13 items (languages, brand-values list, cutoff band, audit sample).