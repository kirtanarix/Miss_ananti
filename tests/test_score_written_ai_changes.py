"""Task 10A/10C checks using invented answers and a mocked Gemini client only."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import score_written_ai as scorer


class WrittenAIChangesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folder = self.root / "Data" / "FAKE001"
        self.folder.mkdir(parents=True)
        self.answer = "Invented example action."
        self.write_answers(self.answer)
        root_patch = patch.object(scorer, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        env_patch = patch.object(scorer, "dotenv_values", return_value={"GEMINI_API_KEY": "fake-test-key"})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        client_patch = patch.object(scorer.genai, "Client")
        self.client_class = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.call = self.client_class.return_value.__enter__.return_value.models.generate_content
        rubric = scorer.load_rubric()
        self.replies = []
        for weights in rubric["weight_matrix"].values():
            active = [code for code, weight in weights.items() if weight is not None]
            reply = {"levels": dict.fromkeys(active, 4),
                     "evidence": dict.fromkeys(active, "example action."),
                     "reasoning": dict.fromkeys(active, "Invented test explanation."), "flags": []}
            self.replies.append(SimpleNamespace(text=json.dumps(reply)))
        self.call.side_effect = self.replies
        sleep_patch = patch.object(scorer.time, "sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def write_answers(self, answer):
        (self.folder / "written_answers.json").write_text(
            json.dumps(dict.fromkeys(("Q1", "Q2", "Q3", "Q4", "Q5"), answer)), encoding="utf-8")

    def command(self, replies, extra=()):
        self.call.side_effect = replies
        output = io.StringIO()
        with patch("sys.argv", ["score_written_ai.py", "--model", "fake-model", "--only", "FAKE001", *extra]), redirect_stdout(output):
            code = scorer.main()
        return code, output.getvalue()

    def test_fingerprint_saved_spacing_stable_and_no_csv_or_table(self):
        with redirect_stdout(io.StringIO()) as printed:
            first, path = scorer.score_contestant("FAKE001", "fake-model")
        original = path.read_bytes()
        self.assertEqual(json.loads(original), first)
        expected = hashlib.sha256(self.answer.encode("utf-8")).hexdigest()
        self.write_answers(" \tInvented\nexample   action.  ")
        self.call.side_effect = self.replies
        second, second_path = scorer.score_contestant("FAKE001", "fake-model")
        self.assertNotEqual(path, second_path)
        self.assertEqual(path.read_bytes(), original)
        for q in first["questions"]:
            self.assertEqual(first["questions"][q]["answer_sha256"], expected)
            self.assertEqual(second["questions"][q]["answer_sha256"], expected)
        self.assertEqual(printed.getvalue(), "")
        self.assertFalse((self.root / "outputs" / "written_review_sheet.csv").exists())
        self.assertEqual(self.call.call_count, 10)
        for call in self.call.call_args_list:
            self.assertEqual(call.kwargs["model"], "fake-model")
            self.assertEqual(call.kwargs["config"].temperature, 0)

    def test_cli_success_saves_csv_and_table_before_exit_zero(self):
        code, output = self.command(self.replies)
        self.assertEqual(code, 0)
        self.assertIn("Question | SA | CR | EV | IJ | CV | Marks", output)
        self.assertIn("Written total: 20 / 40", output)
        with (self.root / "outputs" / "written_review_sheet.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, scorer.FIELDS)
            rows = list(reader)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["answer_text"] == self.answer for row in rows))
        self.assertTrue(all(row["judge_level"] == row["judge_verdict"] == row["judge_comment"] == "" for row in rows))

    def test_cli_failure_saved_before_exit_one(self):
        for bad in (ConnectionError("fake connection error"), SimpleNamespace(text="broken JSON")):
            with self.subTest(failure=type(bad).__name__):
                code, _ = self.command([bad, bad, bad, *self.replies[1:]])
                self.assertEqual(code, 1)
        paths = sorted((self.root / "outputs" / "FAKE001").glob("*.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(result["questions"]["Q1"]["status"], "unable_to_evaluate")
            self.assertIsNone(result["questions"]["Q1"]["marks"])
            self.assertTrue(result["total_is_partial"])
        with (self.root / "outputs" / "written_review_sheet.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 40)

    def test_missing_evidence_failure(self):
        bad = json.loads(self.replies[0].text)
        bad["evidence"]["SA"] = "not in invented answer"
        code, _ = self.command([SimpleNamespace(text=json.dumps(bad))] * 3 + self.replies[1:])
        self.assertEqual(code, 1)

    def test_failure_in_earlier_repeat_still_exits_one(self):
        code, output = self.command([SimpleNamespace(text="bad")] * 3 + self.replies[1:] + self.replies, ["--repeats", "2"])
        self.assertEqual(code, 1)
        self.assertIn("comparison incomplete (unable_to_evaluate)", output)
        self.assertEqual(self.call.call_count, 12)

    def test_only_limits_candidate(self):
        other = self.root / "Data" / "FAKE002"
        other.mkdir()
        (other / "written_answers.json").write_text(json.dumps(dict.fromkeys(("Q1", "Q2", "Q3", "Q4", "Q5"), self.answer)))
        code, _ = self.command(self.replies)
        self.assertEqual(code, 0)
        self.assertEqual(self.call.call_count, 5)
        self.assertFalse((self.root / "outputs" / "FAKE002").exists())

    def test_empty_reservation_removed_on_exception_or_interrupt(self):
        for error in (RuntimeError("fake exception"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.call.side_effect = self.replies
                with patch.object(scorer, "validate", side_effect=error):
                    with self.assertRaises(type(error)):
                        scorer.score_contestant("FAKE001", "fake-model")
                self.assertEqual(list((self.root / "outputs" / "FAKE001").glob("*.json")), [])

    def test_nonempty_file_kept_on_write_exception(self):
        def partial_write(result, handle, **kwargs):
            handle.write("partial test content")
            raise RuntimeError("fake write exception")
        with patch.object(scorer.json, "dump", side_effect=partial_write):
            with self.assertRaises(RuntimeError):
                scorer.score_contestant("FAKE001", "fake-model")
        path = self.root / "outputs" / "FAKE001" / "written_ai_result_run1.json"
        self.assertEqual(path.read_text(encoding="utf-8"), "partial test content")

    def test_data_folder_both_cases(self):
        self.assertEqual(scorer.data_folder(), self.root / "Data")
        # A separate root makes this meaningful on case-insensitive Windows too.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / "data").mkdir()
            with patch.object(scorer, "ROOT", root):
                self.assertEqual(scorer.data_folder().resolve(), (root / "data").resolve())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with patch.object(scorer, "ROOT", Path(directory)):
                with self.assertRaises(FileNotFoundError):
                    scorer.data_folder()

    def test_empty_reservation_removed_when_client_creation_fails(self):
        self.client_class.side_effect = RuntimeError("fake client setup error")
        with self.assertRaises(RuntimeError):
            scorer.score_contestant("FAKE001", "fake-model")
        self.assertEqual(list((self.root / "outputs" / "FAKE001").glob("*.json")), [])


    def run_first_reply(self, reply, failures=False):
        responses = [SimpleNamespace(text=json.dumps(reply))] * (3 if failures else 1)
        self.call.side_effect = responses + self.replies[1:]
        result, path = scorer.score_contestant("FAKE001", "fake-model")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
        return result["questions"]["Q1"]

    def test_extra_inactive_reasoning_is_dropped(self):
        # Q2 has inactive IJ; use its genuine active-parameter set.
        reply = json.loads(self.replies[1].text)
        reply["reasoning"]["IJ"] = "Ignored invented explanation."
        self.call.side_effect = [self.replies[0], SimpleNamespace(text=json.dumps(reply)), *self.replies[2:]]
        result, path = scorer.score_contestant("FAKE001", "fake-model")
        item = json.loads(path.read_text(encoding="utf-8"))["questions"]["Q2"]
        self.assertEqual(item["status"], "valid")
        self.assertNotIn("IJ", item["reasoning"])
        self.assertIn("Extra", item["reasoning_note"])
        self.assertEqual(item["attempts"], 1)
        self.assertEqual(result["written_total"], 20)
        self.sleep.assert_not_called()

    def test_missing_reasoning_is_empty(self):
        reply = json.loads(self.replies[0].text)
        del reply["reasoning"]["SA"]
        item = self.run_first_reply(reply)
        self.assertEqual(item["status"], "valid")
        self.assertEqual(item["reasoning"]["SA"], "")
        self.assertIn("Missing", item["reasoning_note"])
        self.assertEqual(item["attempts"], 1)

    def test_invalid_reasoning_never_fails_question(self):
        for reasoning in (None, [], "explanation", {"SA": 123}):
            with self.subTest(reasoning=reasoning):
                reply = json.loads(self.replies[0].text)
                reply["reasoning"] = reasoning
                item = self.run_first_reply(reply)
                self.assertEqual(item["status"], "valid")
                self.assertTrue(all(value == "" for value in item["reasoning"].values()))
                self.assertIn("reasoning_note", item)

    def test_missing_level_still_fails_after_retries(self):
        reply = json.loads(self.replies[0].text)
        del reply["levels"]["SA"]
        item = self.run_first_reply(reply, failures=True)
        self.assertEqual(item["status"], "unable_to_evaluate")
        self.assertEqual(item["attempts"], 3)
        self.assertIsNone(item["marks"])
        self.assertNotIn("SA", item["levels"])
        self.assertIn("levels must contain exactly", item["reason"])

    def test_second_attempt_valid_same_question_prompt_and_temperature(self):
        for failure in (ConnectionError("fake"), SimpleNamespace(text="broken JSON")):
            with self.subTest(failure=type(failure).__name__):
                self.call.reset_mock()
                self.sleep.reset_mock()
                self.call.side_effect = [failure, *self.replies]
                result, path = scorer.score_contestant("FAKE001", "fake-model")
                item = json.loads(path.read_text(encoding="utf-8"))["questions"]["Q1"]
                self.assertEqual(item["status"], "valid")
                self.assertEqual(item["attempts"], 2)
                self.assertIsNotNone(item["marks"])
                self.assertEqual(self.call.call_count, 6)
                first, second = self.call.call_args_list[:2]
                self.assertEqual(first.kwargs["contents"], second.kwargs["contents"])
                self.assertEqual(first.kwargs["config"], second.kwargs["config"])
                self.assertEqual(second.kwargs["config"].temperature, 0)
                self.sleep.assert_called_once_with(2)
                self.assertTrue(all(q["attempts"] == 1 for qid, q in result["questions"].items() if qid != "Q1"))

    def test_three_failures_last_reason_and_no_marks(self):
        self.call.side_effect = [SimpleNamespace(text="bad"), ConnectionError("fake"), TimeoutError("fake secret")] + self.replies[1:]
        result, path = scorer.score_contestant("FAKE001", "fake-model")
        item = json.loads(path.read_text(encoding="utf-8"))["questions"]["Q1"]
        self.assertEqual(item["status"], "unable_to_evaluate")
        self.assertEqual(item["attempts"], 3)
        self.assertIsNone(item["marks"])
        self.assertEqual(item["reason"], "API failure (TimeoutError)")
        self.assertEqual(item["levels"], {})
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2, 5])
        self.assertEqual(self.call.call_count, 7)

    def test_unverifiable_evidence_rejected_after_retries(self):
        reply = json.loads(self.replies[0].text)
        reply["evidence"]["SA"] = "Never present in the invented answer."
        item = self.run_first_reply(reply, failures=True)
        self.assertEqual(item["status"], "unable_to_evaluate")
        self.assertEqual(item["attempts"], 3)
        self.assertIsNone(item["marks"])
        self.assertFalse(item["evidence_verified"]["SA"])
        self.assertEqual(item["evidence"]["SA"], reply["evidence"]["SA"])
        self.assertIn("evidence not found", item["reason"])

    def test_missing_answer_records_zero_model_attempts(self):
        answers = dict.fromkeys(("Q2", "Q3", "Q4", "Q5"), self.answer)
        (self.folder / "written_answers.json").write_text(json.dumps(answers), encoding="utf-8")
        self.call.side_effect = self.replies[1:]
        result, _ = scorer.score_contestant("FAKE001", "fake-model")
        item = result["questions"]["Q1"]
        self.assertEqual(item["attempts"], 0)
        self.assertEqual(item["status"], "unable_to_evaluate")
        self.assertIsNone(item["marks"])
        self.assertEqual(self.call.call_count, 4)


if __name__ == "__main__":
    unittest.main()
