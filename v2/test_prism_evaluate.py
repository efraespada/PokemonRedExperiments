import unittest

from prism_evaluate import success_rates


class PrismEvaluationTest(unittest.TestCase):
    def test_success_rates(self):
        episodes = [
            {"battle_steps": 10, "experience_gained": 17, "maps": 2, "deaths": 0},
            {"battle_steps": 0, "experience_gained": 0, "maps": 1, "deaths": 0},
            {"battle_steps": 4, "experience_gained": 0, "maps": 1, "deaths": 1},
        ]
        self.assertEqual(
            success_rates(episodes),
            {
                "battle": 2 / 3,
                "victory": 1 / 3,
                "map_transition": 1 / 3,
                "death": 1 / 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
