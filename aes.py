import numpy as np
from numpy.typing import NDArray
from collections.abc import Iterator


from key_schedule import key_schedule
from byte_substitution import s
from shift_rows import shift_rows
from mix_column import mix_column


def aes_128_encrypt(x: NDArray[np.uint8], k: NDArray[np.uint8]):
    key_generator = key_schedule(k)
    return apply_rounds(x, key_generator)


def aes_128_decrypt(x: NDArray[np.uint8], k: NDArray[np.uint8]):
    key_generator = reversed(tuple(key_schedule(k)))
    return apply_rounds(x, key_generator)


def apply_rounds(x: NDArray[np.uint8], key_schedule: Iterator):
    state = x ^ next(key_schedule)

    for round, key in enumerate(key_schedule, 1):
        state = s(state)
        state = shift_rows(state)

        if round < 10:
            state = mix_column(state)

        state ^= key

    return state
