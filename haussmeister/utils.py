"""
General utility functions

(c) 2015 C. Schmidt-Hieber
GPLv3
"""

import numpy as np
import bottleneck


def affine_transform_matrix(dx, dy):
    """
    Compute affine transformations

    Parameters
    ----------
    dx : int
        Shift in x
    dy : int
        Shift in y

    Returns
    -------
    matrix : numpy.ndarray
        2x3 numpy array to be used by roi.transform
    """
    return [np.array([
        [1, 0, dx],
        [0, 1, dy]])]


def zproject(stack):
    try:
        zproj = bottleneck.nanmax(stack, axis=0)
    except MemoryError:
        nframes = stack.shape[0]
        nseqs = 32
        nsubseqs = int(nframes)/nseqs
        zproj = bottleneck.nanmax(np.array([
            bottleneck.nanmax(
                stack[nseq*nsubseqs:(nseq+1)*nsubseqs, :, :],
                axis=0)
            for nseq in range(nseqs)
        ]), axis=0)

    return zproj
