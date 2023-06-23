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
    [0x0B, 0x0D, 0x01, 0x0E],
], dtype=np.uint8)


mult_table_0x02 = [
    (x << 1 if x < 128 else (x << 1) ^ 0x11B) & 0xFF
    for x in range(256)
]


def finite_field_256_dot(v: NDArray[np.uint8], u: NDArray[np.uint8]):
    result = np.zeros(4, dtype=np.uint8)

    for i, (vi, ui) in enumerate(zip(v, u)):
        if ui == 0x01:
            result[i] = vi
        elif ui == 0x02:
            result[i] = mult_table_0x02[vi]
        elif ui == 0x03:
            result[i] = (mult_table_0x02[vi] ^ vi) & 0xFF
        else:
            raise NotImplementedError('GF(2^8) * not implemented for u > 3.')

    return result


def mix_column(b: NDArray[np.uint8]):
    reshaped = b.reshape((4, 4))

    c = np.array([
        finite_field_256_dot(reshaped[:, i], constant_matrix[i])
        for i in range(4)
    ])

    return c.reshape((16,))


def inv_mix_column(c: NDArray[np.uint8]):
    reshaped = c.reshape((4, 4))

    b = np.array([
        finite_field_256_dot(reshaped[:, i], inv_constant_matrix[i])
        for i in range(4)
    ])

    return b.reshape((16,))
