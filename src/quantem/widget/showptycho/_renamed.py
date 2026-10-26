"""ShowPtycho keyword and calibration spellings retired with quantem.gpu's rename, kept readable for a transition.

The direct-method strengths are Tikhonov penalties of a damped least-squares solve, so they are
``phase_damping`` and ``amplitude_damping``; the output grid multiplier is ``upsampling``. Calls and
``quantem.ssb-calibration/6`` saves written with the older names keep working. This mirrors
``quantem.gpu.direct_ptycho._renamed`` so an array viewer needs no quantem.gpu.
"""

import warnings
from functools import wraps

# retired spelling -> current spelling
RENAMED = {
    "phase_regularization": "phase_damping",
    "amplitude_regularization": "amplitude_damping",
    "upsampling_factor": "upsampling",
}


def renamed_keywords(*retired: str):
    """Accept the ``retired`` keyword spellings of the decorated callable with a DeprecationWarning.

    Each is moved onto its current name before the call, so old and new calls run the same code. Passing both
    spellings of one keyword raises ``TypeError``, as passing a keyword twice would. Without this, a notebook
    written before the rename stops with an unexpected keyword error.
    """

    def decorate(function):
        @wraps(function)
        def accept(*args, **kwargs):
            for old in retired:
                if old in kwargs:
                    new = RENAMED[old]
                    if new in kwargs:
                        raise TypeError(f"{function.__qualname__}() got both {old}= and {new}=; pass {new}= only.")
                    warnings.warn(f"{old} is deprecated; use {new}= instead. {old} keeps working during the transition.",
                                  DeprecationWarning, stacklevel=2)
                    kwargs[new] = kwargs.pop(old)
            return function(*args, **kwargs)

        return accept

    return decorate


def current_keys(value):
    """``value`` with every retired key renamed, recursively: a ``quantem.ssb-calibration/6`` record as /7 reads it."""
    if isinstance(value, dict):
        return {RENAMED.get(key, key): current_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [current_keys(item) for item in value]
    return value
