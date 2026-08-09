from collections.abc import Callable

import numpy as np
import numpy.typing as npt


type FloatArray = npt.NDArray[np.float64]
type Transform = Callable[[FloatArray], FloatArray]
type InPlaceTransform = Callable[[FloatArray], None]
type Signal = Callable[[FloatArray, np.random.Generator], FloatArray]
