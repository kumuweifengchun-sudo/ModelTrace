import copy
import unittest
from unittest.mock import patch

import numpy as np

from bank_builder import reference_digest, validate_environments
from fingerprint import assess_identification


class IdentificationTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"model": "model-a", "score": 1.7},
            {"model": "model-b", "score": 1.6},
            {"model": "model-c", "score": -1.0},
        ]
        self.bank = {"validation": {
            "method": "leave-one-environment-out",
            "queries": {"3": {
                "models": {"model-a": {"samples": 12, "correct": 9, "confusions": {"model-b": 3}}},
                "confused_pairs": [{"models": ["model-a", "model-b"], "errors": 5, "max_error_margin": 0.3}],
            }},
        }}

    def test_close_confused_models_abstain_without_changing_ranking(self):
        original = copy.deepcopy(self.results)
        decision = assess_identification(self.results, self.bank, "3")
        self.assertEqual(decision["status"], "ambiguous")
        self.assertEqual(decision["candidates"], ["model-a", "model-b"])
        self.assertEqual(self.results, original)

    def test_clear_margin_only_reports_closest_match(self):
        self.results[1]["score"] = 1.0
        self.assertEqual(assess_identification(self.results, self.bank, "3")["status"], "closest_match")

    def test_missing_validation_and_single_candidate_cannot_confirm(self):
        self.assertEqual(assess_identification(self.results, {}, "3")["status"], "unvalidated")
        self.assertEqual(assess_identification(self.results[:1], self.bank, "3")["status"], "unvalidated")

    def test_equal_scores_abstain_even_without_validation(self):
        self.results[1]["score"] = self.results[0]["score"]
        self.assertEqual(assess_identification(self.results, {}, "3")["status"], "ambiguous")

    def test_query_count_uses_its_own_validation(self):
        self.assertEqual(assess_identification(self.results, self.bank, "1")["status"], "unvalidated")

    def test_reference_digest_tracks_data_and_labels(self):
        rows = [{"source": "model-a", "numbers": [1, 2], "condition_id": "env"}]
        original = reference_digest(rows)
        rows[0]["numbers"] = [2, 1]
        self.assertNotEqual(original, reference_digest(rows))
        rows[0]["numbers"] = [1, 2]
        rows[0]["source"] = "model-b"
        self.assertNotEqual(original, reference_digest(rows))


class EnvironmentValidationTests(unittest.TestCase):
    def test_all_models_and_repeated_challenges_leave_training_together(self):
        rows = [
            {"source": model, "condition_id": condition, "challenge_id": str(challenge),
             "numbers": [condition, model]}
            for condition in ("env1", "env2", "env3")
            for model in ("model-a", "model-b")
            for challenge in range(3)
            for repeat in range(2)
        ]

        def fit(train, models):
            return {"conditions": {row["condition_id"] for row in train}}

        def score(numbers, bank):
            self.assertNotIn(numbers[0], bank["robust"]["conditions"])
            # Both labels intentionally produce indistinguishable features.
            return {"fused": np.array([1.0, 0.9])}

        with patch("bank_builder.fit_robust_artifacts", side_effect=fit), patch("bank_builder.robust_score_numbers", side_effect=score):
            result = validate_environments(rows, ["model-a", "model-b"])
        three = result["queries"]["3"]
        self.assertEqual(three["samples"], 6)
        self.assertEqual(three["accuracy"], 0.5)
        self.assertEqual(three["models"]["model-b"]["confusions"], {"model-a": 3})
        self.assertAlmostEqual(three["confused_pairs"][0]["max_error_margin"], 0.1)

    def test_single_environment_is_not_independent_validation(self):
        rows = [{"source": "model-a", "condition_id": "only"}]
        result = validate_environments(rows, ["model-a"])
        self.assertIsNone(result["queries"]["3"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
