import numpy as np
from numpy.typing import NDArray


from s_box import s_box


def s(a: NDArray[np.uint8]):
    return np.array([s_box[ai] for ai in a])
