import unittest

from run_prism_interactive import select_policy


class FakeEnv:
    def __init__(self, in_battle):
        self.in_battle = in_battle

    def is_in_battle(self):
        return self.in_battle


class PrismPolicySelectionTest(unittest.TestCase):
    def test_selects_navigation_outside_battle(self):
        navigation = object()
        battle = object()
        self.assertIs(select_policy(FakeEnv(False), navigation, battle), navigation)

    def test_selects_specialist_in_battle(self):
        navigation = object()
        battle = object()
        self.assertIs(select_policy(FakeEnv(True), navigation, battle), battle)

    def test_falls_back_to_navigation_without_specialist(self):
        navigation = object()
        self.assertIs(select_policy(FakeEnv(True), navigation), navigation)


if __name__ == "__main__":
    unittest.main()
