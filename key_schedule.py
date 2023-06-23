import numpy as np
from numpy.typing import NDArray

from s_box import s_box


RC = (
    0b0000_0001,
    0b0000_0010,
    0b0000_0100,
    0b0000_1000,
    0b0001_0000,
    0b0010_0000,
    0b0100_0000,
    0b1000_0000,
    0b0001_1011,
    0b0011_0110,
)


def g(v: NDArray[np.uint8], i: int):
    rotated = np.roll(v, -1)

    substituted = np.array([
        s_box[xi]
        for xi in rotated
    ])

    substituted[0] ^= RC[i]

    return substituted


def key_schedule(k: NDArray[np.uint8]):
    w = np.split(k, 4)
    yield np.concatenate(w)

    for i in range(10):
        w[0] ^= g(w[-1], i)
        w[1] ^= w[0]
        w[2] ^= w[1]
        w[3] ^= w[2]
        yield np.concatenate(w)
