PARTY_COUNT = 0xDCD7
PARTY_MON_1 = 0xDCDF
PARTY_MON_SIZE = 0x30
PARTY_SIZE = 6

PARTY_LEVEL_OFFSET = 0x1F
PARTY_HP_OFFSET = 0x22
PARTY_MAX_HP_OFFSET = 0x24

PARTY_LEVELS = tuple(
    PARTY_MON_1 + index * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET
    for index in range(PARTY_SIZE)
)
PARTY_HP = tuple(
    PARTY_MON_1 + index * PARTY_MON_SIZE + PARTY_HP_OFFSET
    for index in range(PARTY_SIZE)
)
PARTY_MAX_HP = tuple(
    PARTY_MON_1 + index * PARTY_MON_SIZE + PARTY_MAX_HP_OFFSET
    for index in range(PARTY_SIZE)
)

POKEDEX_BYTES = 32
POKEDEX_CAUGHT = 0xDE99
POKEDEX_SEEN = 0xDEB9

BADGES = (0xDED9, 0xDEDA, 0xDEDB)
BATTLE_MODE = 0xD22D


def count_bits(read_byte, start, length):
    return sum(int(read_byte(start + offset)).bit_count() for offset in range(length))
