from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .generators import TimeSeriesGenerator
from .ts_types import FloatArray, InPlaceTransform, Signal, Transform


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


@dataclass(frozen=True, slots=True)
class LazyTimeSeries:
    generator: TimeSeriesGenerator
    signal: Signal
    chunk_size: int = 1024

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def __iter__(self) -> Iterator[ImmutableTimeSeries]:
        return self.iter_chunks()

    def iter_chunks(
        self,
        chunk_size: int | None = None,
    ) -> Iterator[ImmutableTimeSeries]:
        size = self.chunk_size if chunk_size is None else chunk_size

        for chunk in self.generator.iter_chunks(self.signal, size):
            yield ImmutableTimeSeries(time=chunk.time, values=chunk.values)

    def materialize(self) -> ImmutableTimeSeries:
        generated = self.generator.evaluate(self.signal)
        return ImmutableTimeSeries(time=generated.time, values=generated.values)

    def materialize_range(
        self,
        start: float,
        stop: float,
    ) -> ImmutableTimeSeries:
        generated = self.generator.generate_range(self.signal, start, stop)
        return ImmutableTimeSeries(time=generated.time, values=generated.values)


type TimeSeries = MaterializedTimeSeries | LazyTimeSeries
