"""
Module for decoding spatial position from neuronal activity

(c) 2016 C. Schmidt-Hieber
GPLv3
"""

import numpy as np
from scipy.misc import factorial


def decodeML(ratemap, counts_time):
    """
    Decode spatial position from neuronal activity. Compute
    maximum likelihood assuming spikes are a Poisson process.
    Follows Dan Manson's code published here:
    https://d1manson.wordpress.com/2015/11/19/non-trivial-vectorizations/

    Parameters
    ----------
    ratemap : numpy.ndarray
        :math:`r`, Spatial firing rate map of shape (x, nrois) or (x, y, nrois)
    counts : numpy.ndarray
        :math:`c`, Spike counts per time bin, shape (ntimepoints, nrois)

    Returns
    -------
    L : numpy.ndarray
        Decoded spatial maximum likelihood map for each time bin,
        shape (x, y, ntimepoints).
        :math:`L = \\prod_{i=0}^{nrois}{\\frac{r_i(x,y)^{c_i(t)}e^{-r_i(x,y)}}{c_i(t)!}}`
    """

    if ratemap.ndim == 2:
        ratemap_new = ratemap[:, np.newaxis, :]
    elif ratemap.ndim == 3:
        ratemap_new = ratemap
    else:
        raise ValueError(
            "ratemap has to have shape (x, nrois) or (x, y, nrois)")

    if np.min(ratemap_new) < 0:
        raise ValueError(
            "ratemap has to be >= 0")
    elif np.min(ratemap_new) < 1e-9:
        ratemap_new += 1e-9

    # sum rates across cells/rois
    term_1 = np.sum(ratemap_new, axis=-1)

    term_2 = np.dot(np.log(ratemap_new), counts_time[:, :, np.newaxis])

    term_3 = np.sum(np.log(factorial(counts_time, exact=False)), axis=-1)

    return np.exp(-term_1[:, :, np.newaxis] + term_2[:, :, :, 0] -
                  term_3[np.newaxis, np.newaxis, :])
