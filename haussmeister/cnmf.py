"""
Apply CNMF to sima datasets
Adopted from
https://github.com/agiovann/Constrained_NMF/blob/master/demoCNMF.ipynb
by Andrea Giovannucci

(c) 2016 C. Schmidt-Hieber
GPLv3
"""

from __future__ import print_function

import sys
import os
import time
import subprocess
import multiprocessing as mp

import ipyparallel

import numpy as np
from scipy.io import savemat, loadmat

from matplotlib import _cntr

import sima
from sima.misc import tifffile
from sima.ROI import ROIList

try:
    from . import utils
except ValueError:
    import utils

sys.path.append('../SPGL1_python_port')
import ca_source_extraction as cse

NCPUS = int(mp.cpu_count()/2)


def tiffs_to_cnmf(haussio_data, mask=None, force=False):
    if not os.path.exists(haussio_data.dirname_comp + '_Y.npy') or force:
        sys.stdout.write('Converting to {0}... '.format(
            haussio_data.dirname_comp + '_Y*.npy'))
        sys.stdout.flush()

        if mask is None:
            filenames = haussio_data.filenames
        else:
            if len(haussio_data.filenames) > mask.shape[0]:
                mask_full = np.concatenate([
                    mask, np.ones((
                        len(haussio_data.filenames)-mask.shape[0])).astype(
                            np.bool)])
            else:
                mask_full = mask
            filenames = [fn for fn, masked in zip(
                haussio_data.filenames, mask_full) if not masked]
        t0 = time.time()
        tiff_sequence = tifffile.TiffSequence(filenames, pattern=None)
        tiff_data = tiff_sequence.asarray(memmap=True).astype(dtype=np.float32)
        tiff_data = np.transpose(tiff_data, (1, 2, 0))
        d1, d2, T = tiff_data.shape
        tiff_data_r = np.reshape(tiff_data, (d1*d2, T), order='F')
        np.save(haussio_data.dirname_comp + '_Y', tiff_data)
        np.save(haussio_data.dirname_comp + '_Yr', tiff_data_r)

        del tiff_data
        del tiff_data_r

        sys.stdout.write('took {0:.2f} s\n'.format(time.time()-t0))
        # 888s


def process_data(haussio_data, mask=None, p=2, nrois_init=200):
    fn_cnmf = haussio_data.dirname_comp + '_cnmf.mat'

    tiffs_to_cnmf(haussio_data, mask)
    sys.stdout.write('Loading from {0}... '.format(
        haussio_data.dirname_comp + '_Y*.npy'))
    Y = np.load(haussio_data.dirname_comp + '_Y.npy', mmap_mode='r')
    d1, d2, T = Y.shape

    if not os.path.exists(fn_cnmf):

        cse.utilities.stop_server()

        sys.stdout.flush()
        t0 = time.time()
        Yr = np.load(haussio_data.dirname_comp + '_Yr.npy', mmap_mode='r')
        sys.stdout.write('took {0:.2f} s\n'.format(time.time()-t0))

        # how to subdivide the work among processes
        n_pixels_per_process = d1*d2/NCPUS

        options = cse.utilities.CNMFSetParms(Y, K=nrois_init, p=p, gSig=[9, 9])
        options['preprocess_params']['n_processes'] = NCPUS
        options['preprocess_params'][
            'n_pixels_per_process'] =  n_pixels_per_process
        options['init_params']['nIter'] = 10
        options['init_params']['maxIter'] = 10
        options['init_params']['use_hals'] = False
        options['spatial_params']['n_processes'] = NCPUS
        options['spatial_params'][
            'n_pixels_per_process'] = n_pixels_per_process
        options['temporal_params']['n_processes'] = NCPUS
        options['temporal_params'][
            'n_pixels_per_process'] = n_pixels_per_process

        cse.utilities.start_server(NCPUS)

        t0 = time.time()
        sys.stdout.write("Preprocessing... ")
        sys.stdout.flush()
        Yr, sn, g = cse.preprocess_data(Yr, **options['preprocess_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 224.94s

        t0 = time.time()
        sys.stdout.write("Initializing components... ")
        sys.stdout.flush()
        Ain, Cin, b_in, f_in, center = cse.initialize_components(
            Y, **options['init_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 2281.37s

        t0 = time.time()
        sys.stdout.write("Updating spatial components... ")
        sys.stdout.flush()
        A, b, Cin = cse.update_spatial_components(
            Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 252.57s

        t0 = time.time()
        sys.stdout.write("Updating temporal components... ")
        sys.stdout.flush()
        C, f, S, bl, c1, neurons_sn, g, YrA = \
            cse.update_temporal_components(
                Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None,
                **options['temporal_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 455.14s

        t0 = time.time()
        sys.stdout.write("Merging ROIs... ")
        sys.stdout.flush()
        A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = \
            cse.merge_components(
                Yr, A, b, C, f, S, sn, options['temporal_params'],
                options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g,
                thr=0.7, mx=100, fast_merge=True)
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 702.55s

        t0 = time.time()
        sys.stdout.write("Updating spatial components... ")
        sys.stdout.flush()
        A2, b2, C2 = cse.update_spatial_components(
            Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 77.16s

        t0 = time.time()
        sys.stdout.write("Updating temporal components... ")
        sys.stdout.flush()
        C2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = \
            cse.update_temporal_components(
                Yr, A2, b2, C2, f, bl=None, c1=None, sn=None, g=None,
                **options['temporal_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 483.41s

        # A: spatial components (ROIs)
        # C: denoised [Ca2+]
        # YrA: residuals ("noise")
        # S: Spikes
        savemat(fn_cnmf, {"A": A2, "C": C2, "YrA": YrA, "S": S2, "bl": bl2})
    else:
        resdict = loadmat(fn_cnmf)
        A2 = resdict["A"]
        C2 = resdict["C"]
        YrA = resdict["YrA"]
        S2 = resdict["S"]
        bl2 = resdict["bl"]

    proj_fn = haussio_data.dirname_comp + "_proj.npy"
    if not os.path.exists(proj_fn):
        zproj = utils.zproject(np.transpose(Y, (2, 0, 1)))
        np.save(proj_fn, zproj)
    else:
        zproj = np.load(proj_fn)

    # DF_F, DF = cse.extract_DF_F(Y.reshape(d1*d2, T), A2, C2)

    t0 = time.time()
    sys.stdout.write("Ordering components... ")
    sys.stdout.flush()
    A_or, C_or, srt = cse.order_components(A2, C2)
    sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))

    cse.utilities.stop_server()

    polygons = contour(A2, d1, d2, thr=0.9)
    rois = ROIList([sima.ROI.ROI(polygons=poly) for poly in polygons])

    return rois, C2, haussio_data, zproj, S2, Y, YrA


def contour(A, d1, d2, thr=None):
    # Adopted from https://github.com/agiovann/Constrained_NMF
    from scipy.sparse import issparse
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    d, nr = np.shape(A)

    x, y = np.mgrid[:d1, :d2]

    coordinates = []
    for i in range(nr):
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, (d1, d2), order='F')
        cntr = _cntr.Cntr(y, x, Bmat)
        cs = cntr.trace(thr)
        if len(cs) > 0:
            coordinates.append(cs[0])
        else:
            coordinates.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return coordinates
