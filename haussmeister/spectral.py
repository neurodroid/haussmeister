import sys
import time

import numpy as np
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    sys.stderr.write("pyfftw unavailable\n")

from stfio import plot as stfio_plot


def fgaussColqu(x, f_c):
    """
    Eq. 5 from Colquhoun & Sigworth, p. 486 of the blue book
    np.log(2.0)/2.0 = 0.34657359028

    Parameters
    ----------
    x : numpy.ndarray
        Frequencies
    f_c : Cutoff frequency (-3dB)

    Returns
    -------
    gauss : numpy.ndarray
        Transfer function to achieve -3dB at f_c
    """
    return np.exp(-0.34657359028*(x/f_c)*(x/f_c))


def convolve(x, transfer, arglist, verbose=True):
    """
    Convolves an array with a transfer function in the frequency domain

    Parameters
    ----------
    x : stfio_plot.Timeseries
        Input data
    transfer : function
        Transfer function
    arglist : list
        Additional arguments to transfer
    verbose : bool, optional
        Verbose output. Default: False

    Returns
    -------
    filtered : stfio_plot.Timeseries
        Filtered data
    """

    t0 = time.time()

    inputa = x.data.copy()
    outsize = int(len(inputa)/2.0 + 1)
    outputa = np.empty((outsize), dtype=np.complex)

    fft = pyfftw.FFTW(inputa, outputa, direction='FFTW_FORWARD',
                      flags=('FFTW_ESTIMATE',), threads=8)
    ifft = pyfftw.FFTW(outputa, inputa, direction='FFTW_BACKWARD',
                       flags=('FFTW_ESTIMATE',), threads=8)

    if verbose:
        sys.stdout.write("Computing frequencies... ")
        sys.stdout.flush()
    f = np.arange(0, len(outputa), dtype=np.float) / (len(inputa) * x.dt)
    try:
        assert(len(f) == len(outputa))
    except:
        sys.stderr.write("\nError in array lengths: %d != %d\n" % (
            len(f), len(outputa)))
        sys.exit(0)

    if verbose:
        sys.stdout.write("done\nForward fft (convolve)... ")
        sys.stdout.flush()
    fft()

    outputa *= transfer(f, *arglist)

    if verbose:
        sys.stdout.write("done\nReverse fft (convolve)... ")
        sys.stdout.flush()

    ifft(normalise_idft=False)

    # Scale
    inputa /= len(x.data)

    if verbose:
        sys.stdout.write("done (%.2f ms)\n" % ((time.time()-t0)*1e3))
        sys.stdout.flush()

    return stfio_plot.Timeseries(inputa, x.dt)


def gaussian_filter(x, f_c, verbose=True):
    """
    Gaussian filter

    Parameters
    ----------
    x : stfio_plot.Timeseries
        Input data
    f_c : float
        Cutoff frequency in kHz (-3 dB)
    verbose : bool, optional
        Verbose output. Default: False

    Returns
    -------
    x convolved with a Gaussian filter kernel.
    """

    return convolve(x, fgaussColqu, [f_c, ], verbose=verbose)


def lowpass(x, f_c, verbose=True):
    """
    Lowpass filter

    Parameters
    ----------
    x : stfio_plot.Timeseries
        Input data
    f_c : float
        Cutoff frequency in kHz (-3 dB)
    verbose : bool, optional
        Verbose output. Default: False

    Returns
    -------
    x convolved with a Gaussian filter kernel.
    """
    return gaussian_filter(x, f_c, verbose=verbose)


def highpass(x, f_c, verbose=True):
    """
    Highpass filter

    Parameters
    ----------
    x : stfio_plot.Timeseries
        Input data
    f_c : float
        Cutoff frequency in kHz (-3 dB)
    verbose : bool, optional
        Verbose output. Default: False

    Returns
    -------
    x convolved with a Gaussian filter kernel.
    """
    return convolve(
        x, lambda f, f_c: 1.0 - fgaussColqu(f, f_c), [f_c, ],
        verbose=verbose)
