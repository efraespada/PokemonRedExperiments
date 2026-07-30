import unittest

from prism_memory import (
    BADGES,
    BALL_COUNT,
    BALLS,
    BATTLE_MODE,
    ENEMY_LEVEL,
    ENEMY_HP,
    ENEMY_MAX_HP,
    ENEMY_SPECIES,
    EVENT_FLAGS,
    EVENT_FLAG_BYTES,
    PARTY_COUNT,
    PARTY_EXP,
    PARTY_HP,
    PARTY_LEVELS,
    PARTY_MAX_HP,
    PARTY_SPECIES,
    ITEM_COUNT,
    ITEMS,
    KEY_ITEM_COUNT,
    KEY_ITEMS,
    MAX_BALLS,
    MAX_ITEMS,
    MAX_KEY_ITEMS,
    POKEDEX_CAUGHT,
    POKEDEX_SEEN,
    PRISM_WRAM_BANK,
    active_party_values,
    classify_battle_outcome,
    count_bits,
    is_productive_interaction,
    monotonic_progress,
    read_u16_be,
    read_u24_be,
    read_item_pocket,
    read_bit_counts,
    read_set_bit_indices,
    update_discovered_indices,
)


class PrismMemoryTest(unittest.TestCase):
    def test_confirmed_addresses(self):
        self.assertEqual(PARTY_COUNT, 0xDCD7)
        self.assertEqual(PARTY_SPECIES, (0xDCDF, 0xDD0F, 0xDD3F, 0xDD6F, 0xDD9F, 0xDDCF))
        self.assertEqual(PARTY_EXP, (0xDCE7, 0xDD17, 0xDD47, 0xDD77, 0xDDA7, 0xDDD7))
        self.assertEqual(PARTY_LEVELS, (0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE))
        self.assertEqual(PARTY_HP, (0xDD01, 0xDD31, 0xDD61, 0xDD91, 0xDDC1, 0xDDF1))
        self.assertEqual(PARTY_MAX_HP, (0xDD03, 0xDD33, 0xDD63, 0xDD93, 0xDDC3, 0xDDF3))
        self.assertEqual(POKEDEX_CAUGHT, 0xDE99)
        self.assertEqual(POKEDEX_SEEN, 0xDEB9)
        self.assertEqual((EVENT_FLAGS, EVENT_FLAG_BYTES), (0xDA72, 250))
        self.assertEqual(BADGES, (0xDED9, 0xDEDA, 0xDEDB))
        self.assertEqual((ITEM_COUNT, ITEMS, MAX_ITEMS), (0xD866, 0xD867, 40))
        self.assertEqual(
            (KEY_ITEM_COUNT, KEY_ITEMS, MAX_KEY_ITEMS), (0xD8A4, 0xD8A5, 50)
        )
        self.assertEqual((BALL_COUNT, BALLS, MAX_BALLS), (0xD8BC, 0xD8BD, 25))
        self.assertEqual(BATTLE_MODE, 0xD22D)
        self.assertEqual(ENEMY_SPECIES, 0xD206)
        self.assertEqual(ENEMY_LEVEL, 0xD213)
        self.assertEqual(ENEMY_HP, 0xD216)
        self.assertEqual(ENEMY_MAX_HP, 0xD218)
        self.assertEqual(PRISM_WRAM_BANK, 1)

    def test_count_bits(self):
        memory = {0x1000: 0b10100001, 0x1001: 0b11110000}
        self.assertEqual(count_bits(lambda address: memory[address], 0x1000, 2), 7)

    def test_read_u16_be(self):
        memory = {0x1000: 0x12, 0x1001: 0x34}
        self.assertEqual(read_u16_be(memory.__getitem__, 0x1000), 0x1234)

    def test_read_u24_be(self):
        memory = {0x1000: 0x12, 0x1001: 0x34, 0x1002: 0x56}
        self.assertEqual(read_u24_be(memory.__getitem__, 0x1000), 0x123456)

    def test_active_party_values_ignores_inactive_slots(self):
        memory = {0x1000: 5, 0x1010: 8, 0x1020: 99}
        values = active_party_values(
            memory.__getitem__, (0x1000, 0x1010, 0x1020), party_count=2
        )
        self.assertEqual(values, (5, 8))

    def test_active_party_values_clamps_invalid_count(self):
        memory = {0x1000: 5, 0x1010: 8}
        self.assertEqual(
            active_party_values(memory.__getitem__, (0x1000, 0x1010), 9),
            (5, 8),
        )
        self.assertEqual(
            active_party_values(memory.__getitem__, (0x1000, 0x1010), -1),
            (),
        )

    def test_classify_battle_outcome_uses_experience_for_victory(self):
        self.assertEqual(classify_battle_outcome(100, 117, 0.5), "victory")

    def test_classify_battle_outcome_uses_empty_party_for_defeat(self):
        self.assertEqual(classify_battle_outcome(100, 100, 0.0), "defeat")

    def test_classify_battle_outcome_keeps_non_decisive_exit_separate(self):
        self.assertEqual(classify_battle_outcome(100, 100, 0.5), "other")

    def test_read_item_pocket_returns_item_quantity_pairs(self):
        memory = {0x1000: 2, 0x1001: 7, 0x1002: 3, 0x1003: 9, 0x1004: 12}
        self.assertEqual(
            read_item_pocket(memory.__getitem__, 0x1000, 0x1001, 40),
            ((7, 3), (9, 12)),
        )

    def test_read_item_pocket_supports_key_items_and_clamps_count(self):
        memory = {0x1000: 9, 0x1001: 4, 0x1002: 8}
        self.assertEqual(
            read_item_pocket(
                memory.__getitem__, 0x1000, 0x1001, 2, quantities=False
            ),
            ((4, 1), (8, 1)),
        )

    def test_read_set_bit_indices_preserves_flag_identity(self):
        memory = {0x1000: 0b10000001, 0x1001: 0b00000110}
        self.assertEqual(
            read_set_bit_indices(memory.__getitem__, 0x1000, 2),
            frozenset({0, 7, 9, 10}),
        )

    def test_read_bit_counts_keeps_badge_regions_separate(self):
        memory = {0x1000: 0b10100001, 0x1001: 0b11110000, 0x1002: 0}
        self.assertEqual(
            read_bit_counts(memory.__getitem__, (0x1000, 0x1001, 0x1002)),
            (3, 4, 0),
        )

    def test_update_discovered_indices_is_monotonic_and_ignores_initial(self):
        discovered = set()
        initial = frozenset({1, 4})
        self.assertEqual(
            update_discovered_indices(discovered, frozenset({1, 4, 7}), initial),
            1,
        )
        self.assertEqual(
            update_discovered_indices(discovered, frozenset({1, 4, 9}), initial),
            2,
        )
        self.assertEqual(discovered, {7, 9})

    def test_monotonic_progress_ignores_initial_and_does_not_regress(self):
        maximum, progress = monotonic_progress(4, initial=3, previous_max=3)
        self.assertEqual((maximum, progress), (4, 1))
        self.assertEqual(
            monotonic_progress(2, initial=3, previous_max=maximum),
            (4, 1),
        )

    def test_productive_interaction_requires_new_stationary_a_screen(self):
        self.assertTrue(
            is_productive_interaction(4, (1, 2), (1, 2), True, 0.25, False)
        )
        self.assertFalse(
            is_productive_interaction(4, (1, 2), (2, 2), True, 0.25, False)
        )
        self.assertFalse(
            is_productive_interaction(5, (1, 2), (1, 2), True, 0.25, False)
        )
        self.assertFalse(
            is_productive_interaction(4, (1, 2), (1, 2), True, 0.25, True)
        )
        self.assertFalse(
            is_productive_interaction(4, (1, 2), (1, 2), True, 0.05, False)
        )


if __name__ == "__main__":
    unittest.main()
