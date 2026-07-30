import unittest

from prism_bootstrap import (
    LARVITAR_EXTENDED_ROUTE_MOVES,
    LARVITAR_LONG_ROUTE_MOVES,
    LARVITAR_NEARBY_MOVES,
    LARVITAR_ROUTE_MOVES,
    PRESETS,
)


class PrismBootstrapTest(unittest.TestCase):
    def test_larvitar_microcurriculum_presets_share_reproducible_prefix(self):
        offer = PRESETS["larvitar_offer_adam"]
        accept = PRESETS["larvitar_accept_adam"]
        ready = PRESETS["larvitar_ready_adam"]

        self.assertEqual(accept[: len(offer)], offer)
        self.assertEqual(accept[len(offer) :], [("a", 180)] * 12)
        self.assertEqual(ready[-35:], [("a", 180)] * 35)

    def test_larvitar_nearby_preset_stops_five_moves_before_offer(self):
        nearby = PRESETS["larvitar_nearby_adam"]
        offer = PRESETS["larvitar_offer_adam"]

        self.assertEqual(LARVITAR_NEARBY_MOVES, 5)
        self.assertEqual(nearby[-1], ("wait", 240))
        self.assertEqual(
            nearby[:-1],
            offer[: -(LARVITAR_NEARBY_MOVES + 1)],
        )

    def test_larvitar_route_preset_stops_ten_moves_before_offer(self):
        route = PRESETS["larvitar_route_adam"]
        offer = PRESETS["larvitar_offer_adam"]

        self.assertEqual(LARVITAR_ROUTE_MOVES, 10)
        self.assertEqual(route[-1], ("wait", 240))
        self.assertEqual(
            route[:-1],
            offer[: -(LARVITAR_ROUTE_MOVES + 1)],
        )

    def test_larvitar_long_route_preset_stops_twenty_moves_before_offer(self):
        route = PRESETS["larvitar_long_route_adam"]
        offer = PRESETS["larvitar_offer_adam"]

        self.assertEqual(LARVITAR_LONG_ROUTE_MOVES, 20)
        self.assertEqual(route[-1], ("wait", 240))
        self.assertEqual(
            route[:-1],
            offer[: -(LARVITAR_LONG_ROUTE_MOVES + 1)],
        )

    def test_larvitar_extended_route_preset_stops_forty_moves_before_offer(self):
        route = PRESETS["larvitar_extended_route_adam"]
        offer = PRESETS["larvitar_offer_adam"]

        self.assertEqual(LARVITAR_EXTENDED_ROUTE_MOVES, 40)
        self.assertEqual(route[-1], ("wait", 240))
        self.assertEqual(
            route[:-1],
            offer[: -(LARVITAR_EXTENDED_ROUTE_MOVES + 1)],
        )


if __name__ == "__main__":
    unittest.main()
