import unittest

from prism_evaluate import success_rates


class PrismEvaluationTest(unittest.TestCase):
    def test_success_rates(self):
        episodes = [
            {
                "encounters": 1,
                "victories": 1,
                "battle_defeats": 0,
                "maps": 2,
                "deaths": 0,
            },
            {
                "encounters": 0,
                "victories": 0,
                "battle_defeats": 0,
                "maps": 1,
                "deaths": 0,
            },
            {
                "encounters": 1,
                "victories": 0,
                "battle_defeats": 1,
                "maps": 1,
                "deaths": 1,
            },
        ]
        self.assertEqual(
            success_rates(episodes),
            {
                "battle": 2 / 3,
                "victory": 1 / 3,
                "battle_defeat": 1 / 3,
                "map_transition": 1 / 3,
                "death": 1 / 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
