import sys
import os
import time

import numpy as np
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import hilbert
from scipy.stats import zscore
from scipy.io import loadmat, savemat

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    sys.stderr.write("pyfftw unavailable\n")

try:
    from stfio import plot as stfio_plot
except ImportError:
    sys.stderr.write("stfio unavailable\n")


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


def remove_hum(data_raw, dt, humband=(36, 52), referenceband=(20, 30)):
    signal_mean = data_raw.mean()
    signal = data_raw-signal_mean
    W = rfftfreq(signal.size, d=dt)
    f_signal = rfft(signal)
    imax0 = np.where(W > humband[0])[0][0]
    imax1 = np.where(W > humband[1])[0][0]
    f_signal_new = f_signal.copy()
    iref0 = np.where(W > referenceband[0])[0][0]
    iref1 = np.where(W > referenceband[1])[0][0]
    refstd = np.std(f_signal[iref0:iref1])
    maxstd = np.std(f_signal[imax0:imax1])
    f_signal_new[imax0:imax1] /= ((np.abs(f_signal_new[imax0:imax1]/(1.0*refstd)))**1 + 1)
    signal_filtered = irfft(f_signal_new)+signal_mean

    return signal_filtered, W, f_signal, f_signal_new

def fhilbert(signal):
    padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
    tohilbert = np.hstack((signal, padding))
    
    result = hilbert(tohilbert)
    
    result = result[0:len(signal)]

    return result

def findRipples(signal_bp, signal_noise_bp, std_thresholds=(2, 10), durations=(30, 100), fn_hilbert=None):
    lowThresholdFactor, highThresholdFactor = std_thresholds
    minInterRippleInterval, maxRippleDuration = durations
    
    if fn_hilbert is not None and os.path.exists(fn_hilbert):
        f_hilbert = loadmat(fn_hilbert)
        signal_analytic = f_hilbert['signal'][0]
        noise_analytic = f_hilbert['noise'][0]
    else:
        sys.stdout.write("Computing Hilbert transform...")
        sys.stdout.flush()
        signal_analytic = fhilbert(signal_bp.data)
        noise_analytic = fhilbert(signal_noise_bp.data)

        if fn_hilbert is not None and not os.path.exists(fn_hilbert):
            savemat(fn_hilbert, {
                'signal': signal_analytic,
                'noise': noise_analytic
            })
        sys.stdout.write(" done\n")
    signal_envelope = np.abs(signal_analytic)
    noise_envelope = 3.0*np.abs(noise_analytic)
    
    zsignal = signal_envelope - noise_envelope
    zsignal[signal_envelope > noise_envelope] = zscore(zsignal[signal_envelope > noise_envelope])
    zsignal[signal_envelope <= noise_envelope] = 0
    
    thresholded = (zsignal > lowThresholdFactor).astype(int)
    start = np.where(np.diff(thresholded) > 0)[0]
    stop = np.where(np.diff(thresholded) < 0)[0]
    if len(stop) == len(start)-1:
        start = start[:-1]
    if len(stop)-1 == len(start):
        stop = stop[1:]
    if start[0] > stop[0]:
        start = start[:-1]
        stop = stop[1:]

    if not len(start):
        sys.stderr.write("No ripples detected\n")
        return

    minInterRippleSamples = int(np.round(minInterRippleInterval/signal_bp.dt))

    merged = True
    ripples = np.array([start, stop])
    while merged:
        merged = False
        tmpripples = [ripples[:, 0].tolist()]
        for ir, (r1, r2) in enumerate(zip(ripples[0, 1:], ripples[1, :-1])):
            if r1-r2 > minInterRippleSamples:
                tmpripples[-1][1] = r2
                tmpripples.append([r1, ripples[1, ir+1]])
            else:
                merged = True
        ripples = np.array(tmpripples).T.copy()
    durations = (ripples[1, :]-ripples[0, :]) * signal_bp.dt
    assert(np.all(durations > 0))
    ripples = ripples[:, durations < maxRippleDuration]
    
    ripplemaxs = np.array([np.max(zsignal[ripple[0]:ripple[1]]) for ripple in ripples.T])
    ripples = ripples[:, ripplemaxs > highThresholdFactor]
    rippleargmaxs = np.array([np.argmax(zsignal[ripple[0]:ripple[1]])+ripple[0] for ripple in ripples.T])

    return ripples, rippleargmaxs
