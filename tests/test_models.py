"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,0],[0,0],[0,0]],[0,0]),
        ([[1,2],[3,4],[5,6]],[3,4])
    ]
)

def test_daily_mean(test, expected):
    """Test the mean function works for array of zeroes and positive integers"""
    from inflammation.models import daily_mean
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 1, 3], [2, 3, 8], [5, 1, 0]], [5, 3, 8]),
        ([[-1, -1, -3], [-2, -3, -8], [-5, -1, 0]], [-1, -1, 0])
    ]
)



def test_daily_max(test, expected):
    """Test that max function works"""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 1, 3], [2, 3, 8], [5, 1, 0]], [1, 1, 0]),
        ([[-1, -1, -3], [-2, -3, -8], [-5, -1, 0]], [-5, -3, -8]),
        ([[10,1,10],[2,20,10],[30,30,3]],[2,1,3])
    ]
)


def test_daily_min(test, expected):
    """Test that min function works"""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(test), expected)



def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected",
    [
        (
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        (
                [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])

def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)


