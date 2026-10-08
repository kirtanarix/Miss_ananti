"""Video calculator checks using synthetic levels only."""

import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from score_video import load_rubric, score_video


PARAMETERS = ("SA", "Goals", "Purpose", "Clarity", "Voice", "Composure")
EXAMPLE = dict(zip(PARAMETERS, (6, 8, 6, 6, 6, 4)))


class VideoScoreTests(unittest.TestCase):
    def test_user_calculated_example(self):
        marks, total = score_video(EXAMPLE)
        self.assertEqual(marks, dict(zip(PARAMETERS, map(Decimal, ("2.25", "4", "3", "2.25", "3", "1")))))
        self.assertEqual(total, Decimal("15.5"))

    def test_all_eight(self):
        self.assertEqual(score_video(dict.fromkeys(PARAMETERS, 8))[1], Decimal(20))

    def test_all_zero(self):
        self.assertEqual(score_video(dict.fromkeys(PARAMETERS, 0))[1], Decimal(0))

    def test_yaml_maxima(self):
        self.assertEqual(sum(p["maximum_marks"] for p in load_rubric()["parameters"].values()), 20)

    def test_odd_levels_and_full_precision(self):
        for level in (1, 3, 5, 7):
            with self.subTest(level=level):
                marks, total = score_video(dict.fromkeys(PARAMETERS, level))
                self.assertEqual(marks["SA"], Decimal(level) / 8 * 3)
                self.assertEqual(total, Decimal(level) / 8 * 20)

    def test_invalid_levels(self):
        for level in (9, -1, 5.5, 5.0, True, None, "5", float("nan"), float("inf")):
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "SA: level must"):
                score_video({**EXAMPLE, "SA": level})

    def test_missing_parameter(self):
        with self.assertRaisesRegex(ValueError, "Missing parameter.*SA"):
            score_video({p: v for p, v in EXAMPLE.items() if p != "SA"})

    def test_unknown_parameter(self):
        with self.assertRaisesRegex(ValueError, "Unknown parameter.*Extra"):
            score_video({**EXAMPLE, "Extra": 4})

    def test_non_object(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            score_video([])

    def test_maximum_marks_read_from_yaml(self):
        rubric = load_rubric()
        rubric["parameters"]["SA"]["maximum_marks"] = 7
        with patch("score_video.load_rubric", return_value=rubric):
            marks, total = score_video(dict.fromkeys(PARAMETERS, 8))
        self.assertEqual(marks["SA"], Decimal(7))
        self.assertEqual(total, Decimal(24))

    def test_cli_format_and_rejection(self):
        script = Path(__file__).resolve().parents[1] / "score_video.py"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "levels.json"
            path.write_text(json.dumps(EXAMPLE), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(script), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SA: 2.25\n", result.stdout)
            self.assertIn("Video total: 15.50 / 20.00", result.stdout)
            path.write_text(json.dumps({**EXAMPLE, "SA": 5.5}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(script), str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SA: level must be an integer", result.stderr)
            self.assertEqual(result.stdout, "")


class VideoPolicyTests(unittest.TestCase):
    """Check V1's encoded policies, not video inference or an AI evaluator."""

    def setUp(self):
        self.rubric = load_rubric()
        self.policy = self.rubric["confirmed_project_decisions"]

    def test_heavy_reading_can_affect_existing_levels(self):
        reading = self.policy["heavy_reading"]
        self.assertTrue(reading["can_reduce_marks"])
        self.assertTrue(reading["observable_evidence_required"])
        self.assertTrue(reading["relevant_existing_parameters_only"])
        self.assertFalse(reading["informational_only"])
        self.assertFalse(reading["automatic_deduction"])
        self.assertFalse(reading["separate_numerical_penalty"])
        self.assertNotIn(reading["flag"], self.rubric["non_scoring_flags"])
        self.assertIn(reading["flag"], self.rubric["scoring_related_flags"])
        # A supported lower supplied level changes marks through the same formula.
        self.assertEqual(score_video(EXAMPLE)[1] -
                         score_video({**EXAMPLE, "Composure": 2})[1], Decimal("0.5"))

    def test_normal_hesitation_has_no_automatic_deduction(self):
        self.assertFalse(self.policy["normal_hesitation"]["automatic_deduction"])
        self.assertIn("by themselves do not reduce marks", self.policy["normal_hesitation"]["rule"])
        self.assertTrue(self.policy["nervousness"]["can_reduce_marks"])
        self.assertFalse(self.policy["nervousness"]["separate_parameter"])

    def test_supported_and_unsupported_language_policy(self):
        languages = self.policy["languages"]
        self.assertEqual(languages["supported"], ["English", "Hindi", "Hinglish"])
        self.assertEqual(languages["unsupported_action"], "flag_for_review")
        self.assertFalse(languages["unsupported_automatic_retry"])
        self.assertFalse(languages["unsupported_language_penalty"])

    def test_no_video_or_prompt_duration_penalty(self):
        duration = self.policy["duration"]
        self.assertIsNone(duration["video_limit"])
        self.assertIsNone(duration["per_prompt_limit"])
        self.assertFalse(duration["duration_penalty"])
        self.assertFalse(duration["duration_score"])

    def test_concrete_evidence_required_for_every_parameter(self):
        evidence = self.policy["evidence_requirements"]
        for field in ("required_for_every_parameter", "candidate_speech_where_applicable",
                      "concrete_observable_evidence_where_applicable",
                      "chosen_level_and_reduction_must_be_explained"):
            with self.subTest(field=field):
                self.assertTrue(evidence[field])
        self.assertIn("are insufficient", evidence["rule"])

    def test_technical_failure_one_retry_then_review(self):
        technical = self.policy["technical_failure"]
        self.assertEqual(technical["retry_count"], 1)
        self.assertEqual(technical["after_retry_action"], "flag_for_review")
        self.assertEqual(technical["after_retry_status"], "unable_to_be_fairly_evaluated")
        self.assertEqual(technical["flag"], "VIDEO_TECHNICAL_FAILURE")
        self.assertFalse(technical["numerical_penalty"])

    def test_appearance_and_location_never_affect_scoring(self):
        fairness = self.policy["appearance_and_fairness"]
        self.assertEqual(set(fairness["never_affect_score"]), {
            "beauty", "facial attractiveness", "skin colour", "skin tone", "eye colour",
            "physical attractiveness", "body/physical appearance",
            "any similar appearance-based judgement",
        })
        self.assertTrue(fairness["pdf_appearance_and_fairness_prohibitions_preserved"])
        self.assertTrue(fairness["location_background_prestige_never_affect_score"])
        self.assertEqual(len(self.rubric["prohibited_judging_factors"]["factors"]), 8)

    def test_delivery_observations_and_pdf_precedence(self):
        delivery = self.policy["observable_delivery"]
        self.assertTrue(delivery["relevant_existing_parameters_only"])
        self.assertEqual(set(delivery["observations"]), {
            "whether the candidate completes the prompt/task", "repeated stopping",
            "losing the thread", "restarting/recovering", "eye contact", "hand movement",
            "voice confidence", "other observable delivery behaviour relevant to the existing rubric",
        })
        self.assertTrue(self.policy["precedence"]["operational_rules_take_precedence"])
        self.assertTrue(self.policy["precedence"]["pdf_text_preserved"])

    def test_combined_evidence_and_odd_levels(self):
        combined = self.policy["combined_evidence"]
        self.assertEqual(combined["levels_per_parameter"], 1)
        self.assertFalse(combined["average_prompt_scores"])
        self.assertEqual(self.policy["odd_levels"]["allowed"], [1, 3, 5, 7])
        for level in self.policy["odd_levels"]["allowed"]:
            self.assertEqual(score_video(dict.fromkeys(PARAMETERS, level))[1],
                             Decimal(level) / 8 * 20)

    def test_introduction_and_unanswered_prompt_are_evidence_based(self):
        self.assertFalse(self.policy["introduction"]["name_city_alone_earn_marks"])
        unanswered = self.policy["unanswered_prompt"]
        self.assertFalse(unanswered["automatic_reduction_all_parameters"])
        self.assertIn("Inspect the full video", unanswered["rule"])
        self.assertIn("Composure/Clarity", unanswered["rule"])

    def test_level_zero_and_generic_answers_use_existing_rubric(self):
        self.assertTrue(self.policy["level_zero"]["use_existing_pdf_definitions"])
        generic = self.policy["generic_answers"]
        self.assertEqual(generic["parameters"], ["Voice", "Purpose"])
        self.assertFalse(generic["separate_parameter"])
        self.assertFalse(generic["separate_numerical_penalty"])

    def test_parameter_names_maxima_and_anchors_preserved(self):
        expected = {
            "SA": ("Self-awareness & identity", 3),
            "Goals": ("Goal clarity & direction", 4),
            "Purpose": ("Purpose for joining Miss Ananti India", 4),
            "Clarity": ("Clarity & structure of communication", 3),
            "Voice": ("Personal specificity & voice", 4),
            "Composure": ("Composure & task engagement", 2),
        }
        self.assertEqual(set(self.rubric["parameters"]), set(expected))
        for code, (name, maximum) in expected.items():
            with self.subTest(parameter=code):
                parameter = self.rubric["parameters"][code]
                self.assertEqual(parameter["name"], name)
                self.assertEqual(parameter["maximum_marks"], maximum)
                self.assertEqual(set(parameter["levels"]), {"0", "2", "4", "6", "8"})
        self.assertEqual(sum(p["maximum_marks"] for p in self.rubric["parameters"].values()), 20)


if __name__ == "__main__":
    unittest.main()
