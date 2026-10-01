import unittest
from decimal import Decimal

from score_written import load_rubric, score_question


class WrittenScoreTests(unittest.TestCase):
    def test_q1_worked_example(self):
        result = score_question(
            "Q1", {"SA": 4, "CR": 6, "EV": 2, "IJ": 6, "CV": 4},
            load_rubric()["weight_matrix"]["Q1"],
        )
        self.assertEqual(result["percent"], Decimal("61.25"))
        self.assertEqual(result["marks"], Decimal("4.9"))

    def test_q2_ij_rejected(self):
        with self.assertRaisesRegex(ValueError, "IJ is inactive"):
            score_question(
                "Q2", {"SA": 4, "CR": 6, "EV": 2, "IJ": 4, "CV": 4},
                load_rubric()["weight_matrix"]["Q2"],
            )

    def test_q2_ev_accepted(self):
        result = score_question(
            "Q2", {"SA": 0, "CR": 0, "EV": 8, "CV": 0},
            load_rubric()["weight_matrix"]["Q2"],
        )
        self.assertEqual(result["percent"], Decimal("10"))
        self.assertEqual(result["marks"], Decimal("0.8"))

    def test_q5_cr_rejected(self):
        with self.assertRaisesRegex(ValueError, "CR is inactive"):
            score_question(
                "Q5", {"SA": 4, "CR": 4, "IJ": 4, "CV": 4},
                load_rubric()["weight_matrix"]["Q5"],
            )

    def test_odd_levels_accepted(self):
        for level in (1, 3, 5, 7):
            with self.subTest(level=level):
                result = score_question(
                    "Q1", dict.fromkeys(("SA", "CR", "EV", "IJ", "CV"), level),
                    load_rubric()["weight_matrix"]["Q1"],
                )
                self.assertEqual(result["percent"], Decimal(level) / 8 * 100)
                self.assertEqual(result["marks"], Decimal(level))

    def test_invalid_active_levels(self):
        for level in (-1, 9, True, None, "4", float("nan"), float("inf")):
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "level must"):
                score_question("Q1", {"SA": level}, {"SA": "100%"})

    def test_missing_active_level(self):
        with self.assertRaisesRegex(ValueError, "missing level for active cell SA"):
            score_question("Q1", {}, {"SA": "100%"})

    def test_inactive_level(self):
        with self.assertRaisesRegex(ValueError, "EV is inactive"):
            score_question("synthetic", {"EV": 4}, {"EV": None})


if __name__ == "__main__":
    unittest.main()
