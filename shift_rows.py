import numpy as np
from numpy.typing import NDArray


def shift_rows(b: NDArray[np.uint8]):
    shifted = b.reshape((4, 4))

    for i in range(4):
        shifted[i] = np.roll(shifted, -i)

    return shifted.reshape((16,))
