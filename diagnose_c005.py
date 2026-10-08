"""One authorized C005 evaluation with redacted exception diagnostics."""

import traceback
from pathlib import Path

from dotenv import dotenv_values
from score_written_ai_v2 import score_contestant


def main():
    key = dotenv_values(Path(__file__).resolve().parent / ".env", interpolate=False).get("GEMINI_API_KEY")

    def redact(value):
        text = str(value)
        return text.replace(key, "[REDACTED]") if key else text

    try:
        result, path = score_contestant("C005", "gemini-3.8-flash")
        print("Saved:", path.name)
        print("Failed questions:", result["unable_to_evaluate_count"])
        for question, item in result["questions"].items():
            print(question, "status:", item["status"], "attempts:", item.get("attempts"),
                  "reason:", redact(item.get("reason")))
        print("Total:", result["written_total"], "/ 40; partial:", result["total_is_partial"])
        return int(bool(result["unable_to_evaluate_count"]))
    except Exception as exc:
        print("Exception type:", type(exc).__module__ + "." + type(exc).__qualname__)
        print("Exception message:", redact(exc))
        print("Traceback:\n" + redact(traceback.format_exc()))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
