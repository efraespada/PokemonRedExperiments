import unittest

from prism_experiment_pipeline import classify_result, newest_checkpoint


class PrismExperimentPipelineTest(unittest.TestCase):
    def test_accepts_completed_candidate_above_thresholds(self):
        result = classify_result(
            {"status": "completed"},
            {"success_rates": {"victory": 0.4, "story_event": 1.0, "party_growth": 1.0, "battle_defeat": 0.0}},
            min_victory_rate=0.25,
            min_story_rate=1.0,
            min_party_rate=1.0,
        )
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertEqual(result["reasons"], [])

    def test_discards_failed_or_below_threshold_candidate(self):
        result = classify_result(
            {"status": "failed"},
            {"success_rates": {"victory": 0.1, "battle_defeat": 0.2}},
            min_victory_rate=0.25,
            max_defeat_rate=0.0,
        )
        self.assertEqual(result["decision"], "DISCARD")
        self.assertGreaterEqual(len(result["reasons"]), 3)

    def test_newest_checkpoint_uses_numeric_step_order(self):
        self.assertEqual(newest_checkpoint("/tmp/does-not-exist"), None)


if __name__ == "__main__":
    unittest.main()
