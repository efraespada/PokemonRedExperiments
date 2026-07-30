import unittest

from prism_bootstrap import PRESETS


class PrismBootstrapTest(unittest.TestCase):
    def test_larvitar_microcurriculum_presets_share_reproducible_prefix(self):
        offer = PRESETS["larvitar_offer_adam"]
        accept = PRESETS["larvitar_accept_adam"]
        ready = PRESETS["larvitar_ready_adam"]

        self.assertEqual(accept[: len(offer)], offer)
        self.assertEqual(accept[len(offer) :], [("a", 180)] * 12)
        self.assertEqual(ready[-35:], [("a", 180)] * 35)


if __name__ == "__main__":
    unittest.main()
