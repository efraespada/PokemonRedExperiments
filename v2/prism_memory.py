PARTY_COUNT = 0xDCD7
PARTY_MON_1 = 0xDCDF
PARTY_MON_SIZE = 0x30
PARTY_SIZE = 6

PARTY_LEVEL_OFFSET = 0x1F
PARTY_EXP_OFFSET = 0x08
PARTY_HP_OFFSET = 0x22
PARTY_MAX_HP_OFFSET = 0x24

PARTY_LEVELS = tuple(
    PARTY_MON_1 + index * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET
    for index in range(PARTY_SIZE)
)
PARTY_SPECIES = tuple(
    PARTY_MON_1 + index * PARTY_MON_SIZE
    for index in range(PARTY_SIZE)
)
PARTY_EXP = tuple(
    PARTY_MON_1 + index * PARTY_MON_SIZE + PARTY_EXP_OFFSET
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

EVENT_FLAGS = 0xDA72
EVENT_FLAG_BYTES = 250

ITEM_COUNT = 0xD866
ITEMS = 0xD867
MAX_ITEMS = 40
KEY_ITEM_COUNT = 0xD8A4
KEY_ITEMS = 0xD8A5
MAX_KEY_ITEMS = 50
BALL_COUNT = 0xD8BC
BALLS = 0xD8BD
MAX_BALLS = 25

BADGES = (0xDED9, 0xDEDA, 0xDEDB)
BATTLE_MODE = 0xD22D
ENEMY_SPECIES = 0xD206
ENEMY_LEVEL = 0xD213
ENEMY_HP = 0xD216
ENEMY_MAX_HP = 0xD218
PRISM_WRAM_BANK = 1


def classify_battle_outcome(initial_experience, final_experience, party_hp_fraction):
    if int(final_experience) > int(initial_experience):
        return "victory"
    if float(party_hp_fraction) <= 0:
        return "defeat"
    return "other"


def read_item_pocket(read_byte, count_address, items_address, capacity, quantities=True):
    count = max(0, min(int(read_byte(count_address)), int(capacity)))
    stride = 2 if quantities else 1
    entries = []
    for index in range(count):
        address = items_address + index * stride
        item_id = int(read_byte(address))
        quantity = int(read_byte(address + 1)) if quantities else 1
        entries.append((item_id, quantity))
    return tuple(entries)


def read_set_bit_indices(read_byte, start, length):
    indices = set()
    for offset in range(length):
        value = int(read_byte(start + offset))
        for bit in range(8):
            if value & (1 << bit):
                indices.add(offset * 8 + bit)
    return frozenset(indices)


def read_bit_counts(read_byte, addresses):
    return tuple(int(read_byte(address)).bit_count() for address in addresses)


def update_discovered_indices(discovered, current, initial):
    discovered.update(set(current) - set(initial))
    return len(discovered)


def monotonic_progress(current, initial, previous_max):
    new_max = max(int(current), int(previous_max))
    return new_max, max(0, new_max - int(initial))


def count_bits(read_byte, start, length):
    return sum(int(read_byte(start + offset)).bit_count() for offset in range(length))


def read_u16_be(read_byte, start):
    return 256 * int(read_byte(start)) + int(read_byte(start + 1))


def read_u24_be(read_byte, start):
    return (
        65536 * int(read_byte(start))
        + 256 * int(read_byte(start + 1))
        + int(read_byte(start + 2))
    )


def read_u8(read_byte, address):
    return int(read_byte(address))


def active_party_values(read_byte, addresses, party_count, read_value=read_u8):
    active_count = max(0, min(int(party_count), len(addresses)))
    return tuple(read_value(read_byte, address) for address in addresses[:active_count])
