import numpy as np
from numpy.typing import NDArray


constant_matrix = np.array([
    [0x02, 0x03, 0x01, 0x01],
    [0x01, 0x02, 0x03, 0x01],
    [0x01, 0x01, 0x02, 0x03],
    [0x03, 0x01, 0x01, 0x02],
])


def gf_multiply(x, y):
    r = 0

    for _ in range(8):
        if y & 1:
            r ^= x

        hi_bit_set = x & 0x80
        x <<= 1

        if hi_bit_set:
            x ^= 0x11b

        y >>= 1

    return r & 0xFF


def finite_field_256_dot(v: NDArray[np.uint8], u: NDArray[np.uint8]):
    result = np.zeros(4, dtype=np.uint8)

    for i, (vi, ui) in enumerate(zip(v, u)):
        if ui == 0x01:
            result[i] ^= vi
        else:
            result[i] ^= gf_multiply(vi, ui)

    return result


def mix_column(b: NDArray[np.uint8]):
    reshaped = b.reshape((4, 4))

    result = np.array([
        finite_field_256_dot(reshaped[:, i], constant_matrix[i])
        for i in range(4)
    ])

    return result.reshape((16,))
