"""
Module for decoding spatial position from neuronal activity

(c) 2016 C. Schmidt-Hieber
GPLv3
"""

import numpy as np
from scipy.misc import factorial


def decodeMLPoisson(ratemap, counts_time):
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

    if ratemap.shape[-1] != counts_time.shape[-1]:
        raise ValueError(
            "ratemap and counts_time must have same last dimension "
            "(nrois or ncells)")

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


def decodeMLNonparam(activity_map, activity_time, nentries=4):
    """
    Decode spatial position from neuronal activity. Compute
    maximum likelihood non-parametrically.

    Parameters
    ----------
    activity_map : 3D list
        :math:`r`, Spatial activity map of shape (nrois, ncrossings, x)
    activity_time : numpy.ndarray
        :math:`c`, Activity time series, shape (ntimepoints, nrois)
    nentries : int, optional
        Mean number of entries per bin in the histogram

    Returns
    -------
    L : numpy.ndarray
        Decoded spatial maximum likelihood map for each time bin,
        shape (x, y, ntimepoints).
    """

    # Compute normalized histogram of fluorescence at each position
    # for each roi/cell
    # TODO: Can probably be implemented more efficiently using histogramdd
    nbins = int(np.round(len(activity_map[0])/float(nentries)))
    if nbins < 2:
        nbins = 2

    histos = []
    for nroi in range(len(activity_map)):
        histos.append({})
        for ncrossing in range(len(activity_map[nroi])):
            for npos in range(len(activity_map[nroi][ncrossing])):
                if np.isfinite(activity_map[nroi][ncrossing][npos]):
                    if npos not in histos[nroi].keys():
                        histos[nroi][npos] = []
                    histos[nroi][npos].append(
                        activity_map[nroi][ncrossing][npos])

    for nroi, histo in enumerate(histos):
        # bins = np.linspace(0, np.max(
        #     np.concatenate([
        #         histopos
        #         for npos, histopos in histos[nroi].items()]).flatten())+1e-9,
        #                    nbins)
        for npos, histopos in histos[nroi].items():
            bins = np.linspace(
                np.min(histos[nroi][npos]),
                np.max(histos[nroi][npos]), nbins)
            bins = np.concatenate([bins, [1e15, ]])
            if bins[0] != 0:
                bins = np.concatenate([[0, ], bins])

            histos[nroi][npos] = np.histogram(
                histos[nroi][npos], bins=bins, density=True)

    # For each activity entry in activity_time, look up the corresponding
    # probabilities for each cell in histos, then use these probabilities
    # to compute the likelihood
    # TODO: vectorize!

    Pmaps = np.array([
        np.sum(np.log([
            [
                # Compute probability for each position:
                histo[0][np.where(activity[nroi] < histo[1])[0][0]-1]
                for npos, histo in histos[nroi].items()
            ]
            for nroi in range(activity_time.shape[-1])
        ]), axis=0)
        for activity in activity_time
    ])

    assert(
        Pmaps.ndim == 2 and
        Pmaps.shape[0] == activity_time.shape[0])

    return Pmaps.T


def load_for_keras(data):
    vrdict, haussio_data = syncfiles.read_files_2p(data)
    vrdict["posx"]
    vrdict["posy"]
    vrdict["frametvr"]
