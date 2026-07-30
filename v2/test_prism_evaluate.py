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
                "pokedex_seen_progress": 1,
                "pokedex_caught_progress": 1,
                "event_progress": 1,
                "item_progress": 1,
                "key_item_progress": 1,
                "ball_progress": 1,
                "badge_progress": 1,
                "party_species_progress": 1,
                "experience_gained": 10,
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
                "pokedex_seen": 1 / 3,
                "capture": 1 / 3,
                "story_event": 1 / 3,
                "item_acquisition": 1 / 3,
                "key_item_acquisition": 1 / 3,
                "ball_acquisition": 1 / 3,
                "badge_acquisition": 1 / 3,
                "party_growth": 1 / 3,
                "experience_gain": 1 / 3,
                "death": 1 / 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
