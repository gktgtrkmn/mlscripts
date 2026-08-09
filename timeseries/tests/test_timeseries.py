import numpy as np
import pytest

from timeseries import (
    FloatArray,
    LazyTimeSeries,
    MutableTimeSeries,
    TimeSeriesGenerator,
)


def noisy_signal(
    time: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    return np.sin(time) + rng.normal(0, 0.1, size=time.shape)


def double_inplace(values: FloatArray) -> None:
    np.multiply(values, 2, out=values)


def test_materialized_series_mutability() -> None:
    mutable = MutableTimeSeries(
        time=np.arange(3, dtype=np.float64),
        values=np.array([1.0, 2.0, 3.0]),
    )
    original_values = mutable.values

    mutable.transform_inplace(double_inplace)

    assert mutable.values is original_values
    np.testing.assert_array_equal(mutable.values, [2.0, 4.0, 6.0])

    immutable = mutable.freeze()
    assert not immutable.time.flags.writeable
    assert not immutable.values.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        immutable.values[0] = 0

    thawed = immutable.thaw()
    thawed.values[0] = 10
    assert immutable.values[0] == 2


def test_lazy_chunks_are_repeatable_and_match_materialization() -> None:
    lazy = LazyTimeSeries(
        generator=TimeSeriesGenerator(n=10, duration=1.0, seed=42),
        signal=noisy_signal,
        chunk_size=4,
    )

    first_pass = list(lazy)
    second_pass = list(lazy)
    materialized = lazy.materialize()

    assert [len(chunk.time) for chunk in first_pass] == [4, 4, 2]
    np.testing.assert_array_equal(
        np.concatenate([chunk.values for chunk in first_pass]),
        materialized.values,
    )

    for first, second in zip(first_pass, second_pass, strict=True):
        np.testing.assert_array_equal(first.values, second.values)


def test_random_range_matches_materialized_slice() -> None:
    lazy = LazyTimeSeries(
        generator=TimeSeriesGenerator(n=100, duration=10.0, seed=42),
        signal=noisy_signal,
    )
    materialized = lazy.materialize()
    ranged = lazy.materialize_range(2.0, 4.0)
    left = int(np.searchsorted(materialized.time, 2.0, side="left"))
    right = int(np.searchsorted(materialized.time, 4.0, side="left"))

    np.testing.assert_array_equal(ranged.time, materialized.time[left:right])
    np.testing.assert_array_equal(ranged.values, materialized.values[left:right])
    assert np.all(ranged.time < 4.0)
