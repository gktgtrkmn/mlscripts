import numpy as np
import numpy.typing as npt
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Callable
from typing import ParamSpec

np.random.seed(42)
P = ParamSpec("P")


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


def plot(data: npt.NDArray[np.bool_]):
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


def generate(
    func: Callable[P, npt.NDArray[np.float64]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> npt.NDArray[np.float64]:
    return func(*args, **kwargs)


def sine(n: int, periods: float) -> npt.NDArray[np.float64]:
    t = np.linspace(0, periods * 2 * np.pi, n)
    return np.sin(t)


recurrence = recurrence_plot(generate(sine, n=500, periods=5), threshold=0.5)

plot(recurrence)
