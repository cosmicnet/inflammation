"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch

@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))

@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, 4], [5, 6]], [5, 6]),
     ([[-1, -2], [-3, -4], [-5, -6]], [-1, -2]),
     ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [4, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test max function works for array of zeroes, positive, and negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))

@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, 4], [5, 6]], [1, 2]),
     ([[-1, -2], [-3, -4], [-5, -6]], [-5, -6]),
    ])
def test_daily_min(test, expected):
    """Test min function works for array of zeroes, positive, and negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))


@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'

# TODO(lesson-automatic) Implement tests for the other statistical functions
# TODO(lesson-mocking) Implement a unit test for the load_csv function
