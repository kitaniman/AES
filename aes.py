import numpy as np
from numpy.typing import NDArray


from key_schedule import key_schedule
from byte_substitution import s, inv_s
from shift_rows import shift_rows, inv_shift_rows
from mix_column import mix_column, inv_mix_column


def aes_128_encrypt(x: NDArray[np.uint8], k: NDArray[np.uint8]):
    key_generator = key_schedule(k)
    state = x ^ next(key_generator)

    for round in range(1, 11):
        state = s(state)
        state = shift_rows(state)

        if round < 10:
            state = mix_column(state)

        state ^= next(key_generator)

    return state


def aes_128_decrypt(x: NDArray[np.uint8], k: NDArray[np.uint8]):
    key_generator = reversed(tuple(key_schedule(k)))
    state = x ^ next(key_generator)

    for round in range(10, 1, -1):
        state = inv_shift_rows(state)
        state = inv_s(state)
        state ^= next(key_generator)
        state = inv_mix_column(state)

    # Final round
    state = inv_shift_rows(state)
    state = inv_s(state)
    state ^= next(key_generator)

    return state
