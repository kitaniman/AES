import numpy as np
from numpy.typing import NDArray


constant_matrix = np.array([
    [0x02, 0x03, 0x01, 0x01],
    [0x01, 0x02, 0x03, 0x01],
    [0x01, 0x01, 0x02, 0x03],
    [0x03, 0x01, 0x01, 0x02],
], dtype=np.uint8)


inv_constant_matrix = np.array([
    [0x0E, 0x0B, 0x0D, 0x09],
    [0x09, 0x0E, 0x0B, 0x0D],
    [0x0D, 0x09, 0x0E, 0x0B],
    [0x0B, 0x0D, 0x09, 0x0E],
], dtype=np.uint8)


mult_table_0x02 = [
    (x << 1 if x < 128 else (x << 1) ^ 0x11B) & 0xFF
    for x in range(256)
]


def gf_mult(x, y) -> int:
    p = 0b100011011
    m = 0

    for _ in range(8):
        m <<= 1

        if m & 0b100000000:
            m ^= p

        if y & 0b010000000:
            m ^= x

        y <<= 1

    return m


def finite_field_256_dot(v: NDArray[np.uint8], u: NDArray[np.uint8]) -> int:
    result = 0

    for i, (vi, ui) in enumerate(zip(v, u)):
        if ui == 0x01:
            result ^= vi
        elif ui == 0x02:
            result ^= mult_table_0x02[vi]
        elif ui == 0x03:
            result ^= (mult_table_0x02[vi] ^ vi) & 0xFF
        else:
            result ^= gf_mult(vi, ui)

    return result


def mix_column(b: NDArray[np.uint8]):
    reshaped = b.reshape((4, 4))

    c = np.array([
        [
            finite_field_256_dot(reshaped[:, j], constant_matrix[i])
            for j in range(4)
        ]
        for i in range(4)
    ])

    return c.reshape((16,))


def inv_mix_column(c: NDArray[np.uint8]):
    reshaped = c.reshape((4, 4))

    b = np.array([
        [
            finite_field_256_dot(reshaped[:, j], inv_constant_matrix[i])
            for j in range(4)
        ]
        for i in range(4)
    ])

    return b.reshape((16,))
