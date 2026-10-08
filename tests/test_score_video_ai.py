"""Deterministic V3 contract, orchestration and safety checks; no live AI calls."""

import io
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import types

import score_video_ai as scorer


def evidence(text="I practise teaching with my neighbours each week.", kind="speech", supports=None):
    return {"kind": kind, "text": text, "start_seconds": 2, "end_seconds": 5,
            "supports": supports or (["speech"] if kind == "speech" else ["task_completion"])}


def reply():
    parameters = {}
    for code, level in zip(("SA", "Goals", "Purpose", "Clarity", "Voice", "Composure"), (6, 8, 6, 6, 6, 4)):
        parameters[code] = {"level": level, "evidence": [evidence()],
                            "reason": "A specific activity supports the selected descriptor.",
                            "reduction_basis": []}
    parameters["Composure"]["evidence"] = [evidence(
        "She finishes the three responses and continues after a pause.", "observation")]
    return {"status": "evaluated", "primary_language": "English", "overall_flags": [],
            "parameters": parameters, "review_reason": "", "review_evidence": []}


def technical_reply():
    return {"status": "technical_failure", "primary_language": "unknown",
            "overall_flags": ["VIDEO_TECHNICAL_FAILURE"], "parameters": None,
            "review_reason": "Recording interference prevents fair evaluation.",
            "review_evidence": [evidence("Camera quality makes the response inaudible during the recording.",
                                         "observation")]}


class ResponseTests(unittest.TestCase):
    def validate(self, value, **kwargs):
        return scorer.validate_response(json.dumps(value), **kwargs)

    def test_valid_response_integrates_with_frozen_calculator(self):
        value = self.validate(reply())
        self.assertEqual(value["status"], "evaluated")
        self.assertEqual(value["total"], 15.5)
        self.assertEqual(value["marks"], dict(SA=2.25, Goals=4, Purpose=3, Clarity=2.25, Voice=3, Composure=1))

    def test_odd_levels_accepted(self):
        for level in (1, 3, 5, 7):
            data = reply()
            for item in data["parameters"].values():
                item["level"] = level
            self.assertEqual(self.validate(data)["total"], level / 8 * 20)

    def test_invalid_levels_rejected(self):
        for level in (-1, 9, 5.5, 5.0, True, "6", None):
            with self.subTest(level=level):
                data = reply()
                data["parameters"]["SA"]["level"] = level
                self.assertIsNone(self.validate(data)["total"])

    def test_each_missing_or_unknown_parameter_rejected(self):
        for code in reply()["parameters"]:
            data = reply()
            del data["parameters"][code]
            self.assertEqual(self.validate(data)["error_category"], "invalid_response")
        data = reply()
        data["parameters"]["Beauty"] = deepcopy(data["parameters"]["SA"])
        self.assertIsNone(self.validate(data)["total"])

    def test_missing_evidence_or_reason_rejected_including_zero(self):
        for field, value in (("evidence", []), ("evidence", None), ("reason", ""), ("reason", "   ")):
            data = reply()
            data["parameters"]["SA"]["level"] = 0
            data["parameters"]["SA"][field] = value
            self.assertIsNone(self.validate(data)["total"])

    def test_level_zero_with_absence_observation_accepted(self):
        data = reply()
        data["parameters"]["SA"].update(level=0, evidence=[evidence(
            "Only name and city are supplied; no further personal content is given.", "observation")])
        self.assertEqual(self.validate(data)["status"], "evaluated")

    def test_nonzero_content_requires_actual_speech_evidence(self):
        data = reply()
        data["parameters"]["Goals"]["evidence"] = [evidence(
            "She finishes the future goals response.", "observation")]
        self.assertIsNone(self.validate(data)["total"])

    def test_prohibited_appearance_in_evidence_or_reason_blocks_score(self):
        for text in ("Her skin tone merits full marks.", "She is beautiful.", "सुंदर चेहरा", "Her body shape is ideal."):
            for field in ("reason", "evidence"):
                with self.subTest(text=text, field=field):
                    data = reply()
                    data["parameters"]["SA"][field] = text if field == "reason" else [evidence(text)]
                    value = self.validate(data)
                    self.assertIsNone(value["total"])
                    self.assertTrue(value["review_required"])

    def test_unsupported_language_review_without_retry_or_penalty(self):
        data = reply()
        data["primary_language"] = "Tamil"
        value = self.validate(data)
        self.assertEqual(value["status"], "needs_review")
        self.assertIn("UNSUPPORTED_LANGUAGE", value["overall_flags"])
        self.assertIsNone(value["total"])
        self.assertFalse(value["technical_retry_available"])
        data.update(status="unsupported_language", parameters=None,
                    overall_flags=["UNSUPPORTED_LANGUAGE"], review_reason="Primary language is Tamil.",
                    review_evidence=[evidence("The three responses are spoken in Tamil.", "observation")])
        self.assertEqual(self.validate(data)["status"], "needs_review")

    def test_supported_languages_evaluated(self):
        for language in ("English", "Hindi", "Hinglish"):
            data = reply()
            data["primary_language"] = language
            self.assertEqual(self.validate(data)["status"], "evaluated")

    def test_heavy_reading_needs_matching_repeated_observation(self):
        data = reply()
        item = data["parameters"]["Composure"]
        item["reduction_basis"] = ["heavy_reading"]
        data["overall_flags"] = ["POSSIBLE_HEAVY_READING"]
        self.assertIsNone(self.validate(data)["total"])
        item["evidence"] = [evidence("Repeated line-tracking eye movements accompany stopping and resuming speech.",
                                     "observation", ["heavy_reading"])]
        self.assertEqual(self.validate(data)["total"], 15.5)
        item["evidence"][0]["text"] = "She glanced away once."
        self.assertIsNone(self.validate(data)["total"])

    def test_normal_hesitation_never_automatic_reduction(self):
        data = reply()
        item = data["parameters"]["Composure"]
        item["evidence"].append(evidence("She briefly pauses, then finishes the response.",
                                         "observation", ["normal_hesitation"]))
        self.assertEqual(self.validate(data)["total"], 15.5)
        item["reduction_basis"] = ["normal_hesitation"]
        self.assertIsNone(self.validate(data)["total"])
        item["reduction_basis"] = ["significant_hesitation"]
        self.assertIsNone(self.validate(data)["total"])

    def test_vague_delivery_claim_rejected(self):
        data = reply()
        data["parameters"]["Composure"]["evidence"] = [evidence("She was nervous.", "observation")]
        self.assertIsNone(self.validate(data)["total"])

    def test_technical_failure_has_one_retry_then_no_score(self):
        first = self.validate(technical_reply(), technical_attempt=1)
        second = self.validate(technical_reply(), technical_attempt=2)
        self.assertEqual(first["status"], "retry_required")
        self.assertTrue(first["technical_retry_available"])
        self.assertEqual(second["status"], "unable_to_evaluate")
        self.assertFalse(second["technical_retry_available"])
        self.assertIsNone(first["total"])
        self.assertIsNone(second["total"])
        with self.assertRaises(ValueError):
            self.validate(technical_reply(), technical_attempt=3)

    def test_recognition_problem_reviews_without_technical_retry(self):
        data = technical_reply()
        data.update(status="recognition_failure", overall_flags=["TRANSCRIPT_LOW_QUALITY"])
        value = self.validate(data)
        self.assertEqual(value["status"], "needs_review")
        self.assertFalse(value["technical_retry_available"])
        self.assertIsNone(value["total"])

    def test_flags_do_not_apply_arithmetic_or_invent_thresholds(self):
        data = reply()
        data["overall_flags"] = ["CONTENT_SAFETY_REVIEW", "POSSIBLE_TEMPLATE_RESPONSE", "PROMPT_NOT_ANSWERED"]
        value = self.validate(data)
        self.assertEqual(value["total"], 15.5)
        self.assertTrue(value["review_required"])
        for flag in ("SCORING_DISAGREEMENT", "NEW_PENALTY"):
            data["overall_flags"] = [flag]
            self.assertIsNone(self.validate(data)["total"])

    def test_malformed_duplicate_and_extra_fields_rejected(self):
        for raw in ("", "broken", "[]", '{"status":"evaluated","status":"technical_failure"}'):
            self.assertIsNone(scorer.validate_response(raw)["total"])
        data = reply()
        data["total"] = 20
        self.assertIsNone(self.validate(data)["total"])

    def test_remote_schema_compatible_while_backend_remains_strict(self):
        self.assertNotIn("additionalProperties", json.dumps(scorer.gemini_schema()))
        data = reply()
        data["parameters"]["SA"]["marks"] = 3
        self.assertIsNone(self.validate(data)["total"])

    def test_evidence_timestamp_and_kind_checked(self):
        for field, value in (("start_seconds", -1), ("end_seconds", 1),
                             ("supports", ["heavy_reading"]), ("text", "   ")):
            data = reply()
            data["parameters"]["SA"]["evidence"][0][field] = value
            self.assertIsNone(self.validate(data)["total"])

    def test_prompt_loads_frozen_rubric_without_arithmetic_or_external_transcription(self):
        prompt = scorer.build_prompt()
        self.assertNotIn("{rubric_json}", prompt)
        self.assertNotIn('"maximum_marks"', prompt)
        self.assertIn("A separate transcript is not required", prompt)
        self.assertIn("No video or prompt duration", prompt.replace("There is no\nvideo", "No video"))
        for parameter in scorer.load_rubric()["parameters"].values():
            for descriptor in parameter["levels"].values():
                self.assertIn(descriptor, prompt)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.video = self.root / "original.mp4"
        self.video.write_bytes(b"invented mock-only bytes, never sent to a real API")
        self.original = self.video.read_bytes()
        root_patch = patch.object(scorer, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        key_patch = patch.object(scorer, "dotenv_values", return_value={"GEMINI_API_KEY": "fake-secret"})
        key_patch.start()
        self.addCleanup(key_patch.stop)
        self.client = MagicMock()
        self.uploaded = SimpleNamespace(name="files/mock-upload", state=types.FileState.ACTIVE)
        self.client.files.upload.return_value = self.uploaded
        self.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(reply()))
        client_patch = patch.object(scorer.genai, "Client")
        self.client_class = client_patch.start()
        self.client_class.return_value.__enter__.return_value = self.client
        self.addCleanup(client_patch.stop)

    def run_score(self, **kwargs):
        return scorer.score_contestant("FAKE001", self.video, "fake-model", **kwargs)

    def test_direct_original_upload_no_transcript_and_cleanup(self):
        result, path = self.run_score()
        self.assertEqual(result["total"], 15.5)
        self.assertEqual(self.video.read_bytes(), self.original)
        self.assertEqual(self.client.files.upload.call_args.kwargs["file"], self.video)
        call = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(call["contents"], [self.uploaded])
        self.assertEqual(call["config"].response_schema, scorer.gemini_schema())
        self.client.files.delete.assert_called_once_with(name=self.uploaded.name)
        self.assertEqual(result["upload_cleanup"], "deleted")
        self.assertEqual(path.name, "video_ai_run1.json")
        self.assertEqual(json.loads(path.read_text()), result)
        self.assertFalse(result["official_score"])
        self.assertFalse((self.root / "outputs" / "written_final_report.xlsx").exists())

    def test_append_only_and_lock_cleanup(self):
        _, first = self.run_score()
        saved = first.read_bytes()
        _, second = self.run_score()
        self.assertEqual(second.name, "video_ai_run2.json")
        self.assertEqual(first.read_bytes(), saved)
        self.assertFalse((first.parent / ".video_ai.lock").exists())

    def test_processing_wait_and_failure_cleanup(self):
        self.uploaded.state = types.FileState.PROCESSING
        active = SimpleNamespace(name=self.uploaded.name, state=types.FileState.ACTIVE)
        self.client.files.get.return_value = active
        with patch.object(scorer.time, "monotonic", side_effect=[0, 0, 0]):
            value = scorer.evaluate_video(self.client, self.video, "fake-model", sleep=lambda _: None)
        self.assertEqual(value["status"], "evaluated")
        self.client.files.get.assert_called_once()
        self.client.files.delete.assert_called_once()

    def test_processing_timeout_no_generation_cleanup(self):
        self.uploaded.state = types.FileState.PROCESSING
        with patch.object(scorer.time, "monotonic", side_effect=[0, 121]):
            result = scorer.evaluate_video(self.client, self.video, "fake-model")
        self.assertEqual(result["error_category"], "api_error")
        self.client.models.generate_content.assert_not_called()
        self.client.files.delete.assert_called_once()

    def test_failed_processing_and_get_error_cleanup(self):
        for state in (types.FileState.FAILED, types.FileState.PROCESSING):
            self.client.reset_mock()
            self.uploaded.state = state
            self.client.files.get.side_effect = RuntimeError("fake-secret")
            with patch.object(scorer.time, "monotonic", side_effect=[0, 0, 0]):
                result = scorer.evaluate_video(self.client, self.video, "fake-model", sleep=lambda _: None)
            self.assertIsNone(result["total"])
            self.client.files.delete.assert_called_once()

    def test_upload_and_generation_api_errors_do_not_leak_key(self):
        for operation in (self.client.files.upload, self.client.models.generate_content):
            operation.side_effect = RuntimeError("request URL includes fake-secret")
            with redirect_stdout(io.StringIO()) as printed:
                result, path = self.run_score()
            self.assertIsNone(result["total"])
            self.assertNotIn("fake-secret", path.read_text())
            self.assertNotIn("fake-secret", printed.getvalue())
            operation.side_effect = None
        self.assertEqual(self.client.files.delete.call_count, 1)

    def test_malformed_output_still_deletes_upload(self):
        self.client.models.generate_content.return_value.text = "malformed fake-secret"
        result, path = self.run_score()
        self.assertEqual(result["error_category"], "invalid_response")
        self.assertNotIn("fake-secret", path.read_text())
        self.client.files.delete.assert_called_once()

    def test_cleanup_error_exposed_without_secret(self):
        self.client.files.delete.side_effect = RuntimeError("fake-secret")
        result, path = self.run_score()
        self.assertEqual(result["upload_cleanup"], "failed")
        self.assertEqual(result["remote_file_to_delete"], self.uploaded.name)
        self.assertTrue(result["review_required"])
        self.assertNotIn("fake-secret", path.read_text())

    def test_client_setup_failure_saved_and_logs_restored(self):
        previous = logging.root.manager.disable
        self.client_class.side_effect = RuntimeError("fake-secret")
        result, path = self.run_score()
        self.assertIsNone(result["total"])
        self.assertNotIn("fake-secret", path.read_text())
        self.assertEqual(logging.root.manager.disable, previous)

    def test_api_diagnostics_include_only_stage_and_http_status(self):
        error = RuntimeError("request URL includes fake-secret")
        error.code = 404
        self.client.models.generate_content.side_effect = error
        result, path = self.run_score()
        self.assertEqual(result["api_failure_stage"], "generation")
        self.assertEqual(result["http_status"], 404)
        self.assertNotIn("fake-secret", path.read_text())

    def test_client_teardown_error_preserves_remote_cleanup_status(self):
        self.client_class.return_value.__exit__.side_effect = RuntimeError("fake-secret")
        result, path = self.run_score()
        self.assertEqual(result["upload_cleanup"], "deleted")
        self.assertIsNone(result["total"])
        self.assertNotIn("fake-secret", path.read_text())

    def test_one_persisted_technical_retry_and_no_third(self):
        self.client.models.generate_content.return_value.text = json.dumps(technical_reply())
        first, _ = self.run_score()
        second, _ = self.run_score(retry_of=first["run"])
        self.assertEqual(first["status"], "retry_required")
        self.assertEqual(second["status"], "unable_to_evaluate")
        for run in (first["run"], second["run"]):
            with self.assertRaisesRegex(ValueError, "Only one retry"):
                self.run_score(retry_of=run)
        self.assertEqual(self.client.models.generate_content.call_count, 2)

    def test_retry_can_use_replacement_video_and_succeed(self):
        self.client.models.generate_content.return_value.text = json.dumps(technical_reply())
        first, _ = self.run_score()
        self.client.models.generate_content.return_value.text = json.dumps(reply())
        replacement = self.root / "replacement.mp4"
        replacement.write_bytes(b"mock replacement")
        second, _ = scorer.score_contestant("FAKE001", replacement, "fake-model", retry_of=first["run"])
        self.assertEqual(second["status"], "evaluated")
        self.assertEqual(self.client.files.upload.call_args.kwargs["file"], replacement)

    def test_key_echo_redacted_and_cli_errors_safe(self):
        data = reply()
        data["parameters"]["SA"]["reason"] = "The explanation echoed fake-secret."
        self.client.models.generate_content.return_value.text = json.dumps(data)
        _, path = self.run_score()
        self.assertNotIn("fake-secret", path.read_text())
        with patch.object(scorer, "score_contestant", side_effect=RuntimeError("fake-secret")):
            with redirect_stdout(io.StringIO()) as output:
                code = scorer.main(["FAKE001", str(self.video), "--model", "fake-model"])
        self.assertEqual(code, 1)
        self.assertNotIn("fake-secret", output.getvalue())

    def test_invalid_candidate_video_retry_or_timeout_prevents_api(self):
        for kwargs in (dict(candidate_id="../escape"), dict(video=self.root / "missing.mp4"),
                       dict(retry_of=0), dict(processing_timeout=float("nan"))):
            args = dict(candidate_id="FAKE001", video=self.video, model="fake-model")
            args.update(kwargs)
            with self.assertRaises(ValueError):
                scorer.score_contestant(**args)
        self.client.files.upload.assert_not_called()

    def test_missing_key_no_api_and_no_secret_printing(self):
        with patch.object(scorer, "dotenv_values", return_value={}):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY missing"):
                self.run_score()
        self.client_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
