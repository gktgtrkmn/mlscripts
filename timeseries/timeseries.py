import numpy as np
from dataclasses import dataclass
from types import FloatArray, Transform, InPlaceTransform


def prepare_arrays(
    time: FloatArray,
    values: FloatArray,
    *,
    writable: bool,
) -> tuple[FloatArray, FloatArray]:
    time = np.array(time, dtype=np.float64, copy=True)
    values = np.array(values, dtype=np.float64, copy=True)

    if time.ndim != 1 or values.ndim != 1:
        raise ValueError("time and values must be one dimensional")

    if time.shape != values.shape:
        raise ValueError("time and values must have matching shapes")

    if not writable:
        time.setflags(write=False)
        values.setflags(write=False)

    return time, values


@dataclass(frozen=True, slots=True)
class ImmutableTimeSeries:
    time: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        time, values = prepare_arrays(self.time, self.values, writable=False)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)

    def transformed(self, transform: Transform) -> ImmutableTimeSeries:
        return ImmutableTimeSeries(time=self.time, values=transform(self.values))

    def thaw(self) -> MutableTimeSeries:
        return MutableTimeSeries(self.time, self.values)


@dataclass(slots=True)
class MutableTimeSeries:
    time: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        self.time, self.values = prepare_arrays(
            self.time,
            self.values,
            writable=True,
        )

    def transform_inplace(self, transform: InPlaceTransform) -> None:
        transform(self.values)

    def freeze(self) -> ImmutableTimeSeries:
        return ImmutableTimeSeries(self.time, self.values)


type MaterializedTimeSeries = MutableTimeSeries | ImmutableTimeSeries
