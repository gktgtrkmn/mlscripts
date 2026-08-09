import numpy as np
import numpy.typing as npt
from collections.abc import Callable


type FloatArray = npt.NDArray[np.float64]


type Transform = Callable[[FloatArray], FloatArray]

type InPlaceTransform = Callable[[FloatArray], None]
