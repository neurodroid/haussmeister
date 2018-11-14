import sys
import os
import time

import numpy as np
import numpy.ma as ma
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.signal import hilbert
from scipy.stats import zscore
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    sys.stderr.write("pyfftw unavailable\n")


def save_ma(ftrunk, marr):
    if not isinstance(marr, ma.core.MaskedArray):
        marr = ma.array(marr, mask=False)
    data = np.array(marr)
    mask = np.array(marr.mask)
    np.save(ftrunk + ".data.npy", data)
    np.save(ftrunk + ".mask.npy", mask)


def load_ma(ftrunk):
    data = np.load(ftrunk + ".data.npy")
    mask = np.load(ftrunk + ".mask.npy")
    return ma.array(data, mask=mask)


class Timeseries(object):
    # it this is 2d, the second axis (shape[1]) is time
    def __init__(self, *args, **kwargs):
        if len(args) > 2:
            raise RuntimeError(
                "Timeseries accepts at most two non-keyworded arguments")
        fromFile = False
        # First argument has to be either data or file_trunk
        if isinstance(args[0], str):
            if len(args) > 1:
                raise RuntimeError(
                    "Timeseries accepts only one non-keyworded "
                    "argument if instantiated from file")
            if os.path.exists("%s_data.npy" % args[0]):
                self.data = np.load("%s_data.npy" % args[0])
            else:
                self.data = load_ma("%s_data.npy" % args[0])

            self.dt = np.load("%s_dt.npy" % args[0])

            fxu = open("%s_xunits" % args[0], 'r')
            self.xunits = fxu.read()
            fxu.close()

            fyu = open("%s_yunits" % args[0], 'r')
            self.yunits = fyu.read()
            fyu.close()
            fromFile = True
        else:
            self.data = args[0]
            self.dt = args[1]

        if len(kwargs) > 0 and fromFile:
            raise RuntimeError(
                "Can't set keyword arguments if Timeseries was "
                "instantiated from file")

        for key in kwargs:
            if key == "xunits":
                self.xunits = kwargs["xunits"]
            elif key == "yunits":
                self.yunits = kwargs["yunits"]
            elif key == "linestyle":
                self.linestyle = kwargs["linestyle"]
            elif key == "linewidth":
                self.linewidth = kwargs["linewidth"]
            elif key == "color":
                self.color = kwargs["color"]
            elif key == "colour":
                self.color = kwargs["colour"]
            else:
                raise RuntimeError("Unknown keyword argument: " + key)

        if "xunits" not in kwargs and not fromFile:
            self.xunits = "ms"
        if "yunits" not in kwargs and not fromFile:
            self.yunits = "mV"
        if "linestyle" not in kwargs:
            self.linestyle = "-"
        if "linewidth" not in kwargs:
            self.linewidth = 1.0
        if "color" not in kwargs and "colour" not in kwargs:
            self.color = 'k'

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __add__(self, other):
        if isinstance(other, Timeseries):
            result = self.data + other.data
        else:
            result = self.data + other

        return self.copy_attributes(result)

    def __mul__(self, other):
        if isinstance(other, Timeseries):
            result = self.data * other.data
        else:
            result = self.data * other

        return self.copy_attributes(result)

    def __sub__(self, other):
        if isinstance(other, Timeseries):
            result = self.data - other.data
        else:
            result = self.data - other

        return self.copy_attributes(result)

    def __div__(self, other):
        if isinstance(other, Timeseries):
            result = self.data / other.data
        else:
            result = self.data / other

        return self.copy_attributes(result)

    def __truediv__(self, other):
        return self.__div__(other)

    def copy_attributes(self, data):
        return Timeseries(
            data, self.dt, xunits=self.xunits, yunits=self.yunits,
            linestyle=self.linestyle, linewidth=self.linewidth,
            color=self.color)

    def x_trange(self, tstart, tend):
        return np.arange(int(tstart/self.dt), int(tend/self.dt), 1.0,
                         dtype=np.float) * self.dt

    def y_trange(self, tstart, tend):
        return self.data[int(tstart/self.dt):int(tend/self.dt)]

    def timearray(self):
        return np.arange(0.0, self.data.shape[-1]) * self.dt

    def duration(self):
        return self.data.shape[-1] * self.dt

    def interpolate(self, newtime, newdt):
        if len(self.data.shape) == 1:
            return Timeseries(np.interp(newtime, self.timearray(), self.data,
                                        left=np.nan, right=np.nan), newdt)
        else:
            # interpolate each row individually:
            # iparray = ma.zeros((self.data.shape[0], len(newtime)))
            # for nrow, row in enumerate(self.data):
            #     flin = \
            #         interpolate.interp1d(
            #            self.timearray(), row,
            #            bounds_error=False, fill_value=np.nan, kind=kind)
            #     iparray[nrow,:]=flin(newtime)
            iparray = ma.array([
                np.interp(
                    newtime, self.timearray(), row, left=np.nan, right=np.nan)
                for nrow, row in enumerate(self.data)])
            return Timeseries(iparray, newdt)

    def maskedarray(self, center, left, right):
        # check whether we have enough data left and right:
        if len(self.data.shape) > 1:
            mask = \
                np.zeros((self.data.shape[0], int((right+left)/self.dt)))
            maskedarray = \
                ma.zeros((self.data.shape[0], int((right+left)/self.dt)))
        else:
            mask = np.zeros((int((right+left)/self.dt)))
            maskedarray = ma.zeros((int((right+left)/self.dt)))
        offset = 0
        if center - left < 0:
            if len(self.data.shape) > 1:
                mask[:, :int((left-center)/self.dt)] = 1
            else:
                mask[:int((left-center)/self.dt)] = 1
            leftindex = 0
            offset = int((left-center)/self.dt)
        else:
            leftindex = int((center-left)/self.dt)
        if center + right >= len(self.data) * self.dt:
            endtime = len(self.data) * self.dt
            if len(self.data.shape) > 1:
                mask[:, -int((center+right-endtime)/self.dt):] = 1
            else:
                mask[-int((center+right-endtime)/self.dt):] = 1
            rightindex = int(endtime/self.dt)
        else:
            rightindex = int((center+right)/self.dt)
        for timest in range(leftindex, rightindex):
                if len(self.data.shape) > 1:
                    if timest-leftindex+offset < maskedarray.shape[1] and \
                       timest < self.data.shape[1]:
                        maskedarray[:, timest-leftindex+offset] = \
                            self.data[:, timest]
                else:
                    if timest-leftindex+offset < len(maskedarray):
                        maskedarray[timest-leftindex+offset] = \
                            self.data[timest]
        maskedarray.mask = ma.make_mask(mask)
        return Timeseries(maskedarray, self.dt)

    def save(self, file_trunk):
        if isinstance(self.data, ma.MaskedArray):
            save_ma("%s_data.npy" % file_trunk, self.data)
        else:
            np.save("%s_data.npy" % file_trunk, self.data)

        np.save("%s_dt.npy" % file_trunk, self.dt)

        fxu = open("%s_xunits" % file_trunk, 'w')
        fxu.write(self.xunits)
        fxu.close()

        fyu = open("%s_yunits" % file_trunk, 'w')
        fyu.write(self.yunits)
        fyu.close()

    def plot(self):
        fig = plt.figure(figsize=(8, 6))

        ax = StandardAxis(fig, 111, hasx=True)
        ax.plot(self.timearray(), self.data, '-k')

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
    x : Timeseries
        Input data
    transfer : function
        Transfer function
    arglist : list
        Additional arguments to transfer
    verbose : bool, optional
        Verbose output. Default: False

    Returns
    -------
    filtered : Timeseries
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

    return Timeseries(inputa, x.dt)


def gaussian_filter(x, f_c, verbose=True):
    """
    Gaussian filter

    Parameters
    ----------
    x : Timeseries
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
    x : Timeseries
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
    x : Timeseries
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
    """
    Find ripples in a signal

    Parameters
    ----------
    signal_bp : numpy.ndarray
        Signal bandpass-filtered in the ripple band
    signal_noise_bp : numpy.ndarray
        Signal bandpass-filtered in the noise band, used as a reference signal
        to avoid false positives
    std_thresholds : tuple of ints, optional
        Detection thresholds, expressed in standard deviations. First element
        is the minimal detection threshold, second element is the minimal amplitude
        of the largest point within the ripple
    durations : tuple of ints, optional
        Minimal and maximal duration thresholds for a ripple
    fn_hilbert : string, optional
        File name of a previously computed hilbert transform in MATLAB format.
        If the file does not exist, the hilbert transform will be computed and stored
        to this file name.

    Returns
    -------
    ripples, ripplesargmaxs:
        ripples are the starting and ending indices of the detected ripples.
        ripplesargmaxs are the indices of the maximal values within each ripple.
    """
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
            elif ir == ripples.shape[-2]:
                tmpripples[-1][1] = ripples[1][-1]
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


def xcorr(x, y, normed=True):
    correls = np.correlate(x, y, mode=2)

    if normed:
        correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    return correls
