import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from generators import TimeSeriesGenerator, Signal


@dataclass(frozen=True)
class StaticTimeSeries:
    time: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Generation:
    generator: TimeSeriesGenerator
    signal: Signal

    def materialize(self) -> StaticTimeSeries:
        return StaticTimeSeries(
            time=self.generator.time,
            values=self.generator.generate(self.signal),
        )


type TimeSeries = StaticTimeSeries | Generation
