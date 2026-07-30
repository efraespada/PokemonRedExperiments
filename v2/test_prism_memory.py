import unittest

from prism_memory import (
    BADGES,
    PARTY_COUNT,
    PARTY_HP,
    PARTY_LEVELS,
    PARTY_MAX_HP,
    POKEDEX_CAUGHT,
    POKEDEX_SEEN,
    count_bits,
)


class PrismMemoryTest(unittest.TestCase):
    def test_confirmed_addresses(self):
        self.assertEqual(PARTY_COUNT, 0xDCD7)
        self.assertEqual(PARTY_LEVELS, (0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE))
        self.assertEqual(PARTY_HP, (0xDD01, 0xDD31, 0xDD61, 0xDD91, 0xDDC1, 0xDDF1))
        self.assertEqual(PARTY_MAX_HP, (0xDD03, 0xDD33, 0xDD63, 0xDD93, 0xDDC3, 0xDDF3))
        self.assertEqual(POKEDEX_CAUGHT, 0xDE99)
        self.assertEqual(POKEDEX_SEEN, 0xDEB9)
        self.assertEqual(BADGES, (0xDED9, 0xDEDA, 0xDEDB))

    def test_count_bits(self):
        memory = {0x1000: 0b10100001, 0x1001: 0b11110000}
        self.assertEqual(count_bits(lambda address: memory[address], 0x1000, 2), 7)


if __name__ == "__main__":
    unittest.main()
