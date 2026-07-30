import numpy as np
import numpy.typing as npt
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


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


def generate(wave: str) -> npt.NDArray[np.float64]:
    if wave == "sin":
        return np.linspace(0, 10 * np.pi, 500)
    return np.random.normal(size=500)


recurrence = recurrence_plot(generate("sin"), threshold=0.5)

plot(recurrence)
