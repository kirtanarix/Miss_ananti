"""One tiny Gemini request; no candidate data or evaluator imports."""

import logging
from pathlib import Path

from dotenv import dotenv_values
from google import genai
from google.genai import types


def main():
    key = dotenv_values(Path(__file__).resolve().parent / ".env", interpolate=False).get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is missing from .env")
        return 1

    def redact(value):
        return str(value).replace(key, "[REDACTED]")

    logging.disable(logging.CRITICAL)
    try:
        with genai.Client(api_key=key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        )) as client:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents="Reply with the single word OK",
                config=types.GenerateContentConfig(temperature=0),
            )
        print("Response text:", redact(response.text))
        for candidate in response.candidates or []:
            print("Finish reason:", redact(candidate.finish_reason))
        return 0
    except Exception as exc:
        print("Exception type:", type(exc).__module__ + "." + type(exc).__qualname__)
        print("Exception message:", redact(exc))
        for name in ("code", "status", "status_code", "message"):
            value = getattr(exc, name, None)
            if value is not None:
                print(name + ":", redact(value))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
