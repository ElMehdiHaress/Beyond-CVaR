"""
beyond_cvar package
- Keeps imports lightweight (no eager submodule imports).
- Adds a small NumPy compatibility shim for code that referenced np.float / np.int / np.complex.
"""

import numpy as _np  # compatibility for environments with numpy>=1.24
for _alias, _type in (("float", float), ("int", int), ("complex", complex)):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _type)

__all__ = [
    "prediction_error_cpt",
    "prediction_error_srm",
    "excess_risk_over_t_srm",
    "excess_risk_over_n_srm",
    "cpt_value_estimation",
    "robust_mean_estimation",
]


