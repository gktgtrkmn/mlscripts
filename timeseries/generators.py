from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .ts_types import FloatArray, Signal


@dataclass(frozen=True, slots=True)
class GeneratedChunk:
    time: FloatArray
    values: FloatArray


@dataclass(frozen=True, slots=True)
class TimeSeriesGenerator:
    n: int
    duration: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    @property
    def time(self) -> FloatArray:
        return np.linspace(0, self.duration, self.n, endpoint=False)

    def _evaluate(
        self,
        signal: Signal,
        time: FloatArray,
        rng: np.random.Generator,
    ) -> GeneratedChunk:
        values = np.asarray(signal(time, rng), dtype=np.float64)

        if values.shape != time.shape:
            raise ValueError("signal output must have the same shape as time")

        return GeneratedChunk(time=time, values=values)

    def evaluate(self, signal: Signal) -> GeneratedChunk:
        rng = np.random.default_rng(self.seed)
        return self._evaluate(signal, self.time, rng)

    def generate(self, signal: Signal) -> FloatArray:
        return self.evaluate(signal).values

    def generate_range(
        self,
        signal: Signal,
        start: float,
        stop: float,
    ) -> GeneratedChunk:
        if not 0 <= start < stop <= self.duration:
            raise ValueError("range must satisfy 0 <= start < stop <= duration")

        generated = self.evaluate(signal)
        left = int(np.searchsorted(generated.time, start, side="left"))
        right = int(np.searchsorted(generated.time, stop, side="left"))

        return GeneratedChunk(
            time=generated.time[left:right],
            values=generated.values[left:right],
        )

    def iter_chunks(
        self,
        signal: Signal,
        chunk_size: int,
    ) -> Iterator[GeneratedChunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        time = self.time
        rng = np.random.default_rng(self.seed)

        for start in range(0, self.n, chunk_size):
            chunk_time = time[start : start + chunk_size]
            yield self._evaluate(signal, chunk_time, rng)
