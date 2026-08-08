from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from generators import Signal, TimeSeriesGenerator
from timeseries import Generation


def recurrence_plot(
    data: npt.NDArray[np.float64], threshold: float
) -> npt.NDArray[np.bool_]:
    n = len(data)
    recurrence = np.eye(n, dtype=np.bool_)
    for i, j in combinations(range(n), 2):
        is_recurrent = np.abs(data[i] - data[j]) <= threshold
        recurrence[i, j] = is_recurrent
        recurrence[j, i] = is_recurrent
    return recurrence


def plot(data: npt.NDArray[np.bool_]) -> None:
    sns.heatmap(
        data,
        cmap="binary",
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    plt.xlabel("Time")
    plt.ylabel("Time")
    plt.title("Recurrence Plot")
    plt.show()


def sine(
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    def signal(
        time: npt.NDArray[np.float64],
        _rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        return amplitude * np.sin(2 * np.pi * frequency * time + phase)

    return signal


def main() -> None:
    generator = TimeSeriesGenerator(n=500, duration=5.0, seed=42)
    generation = Generation(generator=generator, signal=sine(frequency=1.0))
    series = generation.materialize()
    recurrence = recurrence_plot(series.values, threshold=0.5)

    plot(recurrence)


if __name__ == "__main__":
    main()
