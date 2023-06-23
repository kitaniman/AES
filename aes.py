import numpy as np
from numpy.typing import NDArray


from key_schedule import key_schedule
from byte_substitution import s
from shift_rows import shift_rows
from mix_column import mix_column


def aes_128(x: NDArray[np.unit8], k: NDArray[np.unit8]):
    key_generator = key_schedule(k)
    state = x
    state ^= next(key_generator)

    # TODO: remove mix_column form the last step
    for key in key_generator:
        state = s(state)
        state = shift_rows(state)
        state = mix_column(state)
        state ^= key

    return state
