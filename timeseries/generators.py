from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

Signal = Callable[
    [npt.NDArray[np.float64], np.random.Generator], npt.NDArray[np.float64]
]


@dataclass(slots=True)
class TimeSeriesGenerator:
    n: int
    duration: float
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def time(self) -> npt.NDArray[np.float64]:
        return np.linspace(0, self.duration, self.n, endpoint=False)

    def generate(self, signal: Signal) -> npt.NDArray[np.float64]:
        values = signal(self.time, self.rng)
        return values
