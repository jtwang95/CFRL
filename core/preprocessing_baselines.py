"""
this file contains the preprocessing baselines algorithms
including: fair through unawareness, full
"""

import numpy as np


class UnawarenessPreprocess:
    def __init__(self) -> None:
        pass

    def preprocess(self, z, xt, rtm1, **kwargs):
        """Preprocess the data

        Args:
            z (np.ndarray): sensitive variables, dim: (N, zdim) or (zdim,)
            xt (np.ndarray): states, dim: (N, xdim) or (xdim,)
        Returns:
            xt_new (np.ndarray): processed states, dim: (N, xdim) or (xdim,)
        """
        if rtm1 is None:
            return xt
        else:
            return xt, rtm1


class ConcatenateProprocess:
    def __init__(self) -> None:
        pass

    def preprocess(self, z, xt, rtm1, **kwargs):
        """Preprocess the data

        Args:
            z (np.ndarray): sensitive variables, dim: (N, zdim) or (zdim,)
            xt (np.ndarray): states, dim: (N, xdim) or (xdim,)
        Returns:
            xt_new (np.ndarray): processed states, dim: (N, xdim) or (xdim,)
        """

        if xt.ndim == 1:
            xt = xt[np.newaxis, :]
            z = z[np.newaxis, :]
            xt_new = np.concatenate([xt, z], axis=1)
            if rtm1 is None:
                return xt_new.flatten()
            else:
                return xt_new, rtm1.flatten()
        elif xt.ndim == 2:
            xt_new = np.concatenate([xt, z], axis=1)
            if rtm1 is None:
                return xt_new
            else:
                return xt_new, rtm1
