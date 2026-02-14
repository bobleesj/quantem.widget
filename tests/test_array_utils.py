import numpy as np
import torch

from quantem.widget.array_utils import get_array_backend, to_numpy


def test_backend_numpy():
    arr = np.array([1, 2, 3])
    assert get_array_backend(arr) == "numpy"


def test_backend_torch():
    tensor = torch.tensor([1, 2, 3])
    assert get_array_backend(tensor) == "torch"


def test_backend_list():
    assert get_array_backend([1, 2, 3]) == "unknown"


def test_to_numpy_from_numpy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, arr)


def test_to_numpy_from_torch():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_to_numpy_from_torch_2d():
    tensor = torch.rand(10, 10)
    result = to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)


def test_to_numpy_dtype_conversion():
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = to_numpy(arr, dtype=np.float32)
    assert result.dtype == np.float32


def test_to_numpy_from_list():
    result = to_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)
