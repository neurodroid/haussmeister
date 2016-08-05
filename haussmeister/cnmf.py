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
import tempfile
import glob

import ipyparallel
from ipyparallel import Client

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
try:
    import ca_source_extraction as cse
except ImportError:
    sys.stderr.write("Could not find cse module")

NCPUS = mp.cpu_count()
NCPUS_PATCHES = 16


def get_mmap_name(basename, d1, d2, T):
    return basename.replace('_', '') + \
        '_d1_' + str(d1) + '_d2_' + str(d2) + '_d3_' + \
        str(1) + '_order_' + 'C' + '_frames_' + str(T) + '_.mmap'

def tiffs_to_cnmf(haussio_data, mask=None, force=False):
    mmap_files = glob.glob(
        haussio_data.dirname_comp.replace('_', '') + "*_.mmap")

    if len(mmap_files) == 0 or force:
        sys.stdout.write('Converting to {0}... '.format(
            haussio_data.dirname_comp + '_Y*.npy'))
        sys.stdout.flush()
        t0 = time.time()
        if haussio_data.rawfile is None or not os.path.exists(
                haussio_data.rawfile):
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
            tiff_sequence = tifffile.TiffSequence(filenames, pattern=None)
            tiff_data = tiff_sequence.asarray(memmap=True).astype(
                dtype=np.float32)
        else:
            if mask is not None:
                tiff_data = haussio_data.read_raw().squeeze().astype(
                    np.float32)[np.invert(mask), :, :]
            else:
                tiff_data = haussio_data.read_raw().squeeze().astype(
                    np.float32)

        tiff_data = np.transpose(tiff_data, (1, 2, 0))
        d1, d2, T = tiff_data.shape
        np.save(haussio_data.dirname_comp + '_Y', tiff_data)

        fname_tot = get_mmap_name(haussio_data.dirname_comp, d1, d2, T)
        big_mov = np.memmap(
            fname_tot, mode='w+',
            dtype=np.float32, shape=(d1*d2, T), order='C')
        big_mov[:] = np.reshape(tiff_data, (d1*d2, T), order='C')[:]
        big_mov.flush()

        del tiff_data
        del big_mov

        sys.stdout.write('took {0:.2f} s\n'.format(time.time()-t0))
        # 888s


def process_data(haussio_data, mask=None, p=2, nrois_init=400):
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
        fname_tot = get_mmap_name(haussio_data.dirname_comp, d1, d2, T)
        Yr, dm, Tm = cse.utilities.load_memmap(fname_tot)
        assert(dm[0] == d1)
        assert(dm[1] == d2)
        assert(Tm == T)
        sys.stdout.write('took {0:.2f} s\n'.format(time.time()-t0))

        # how to subdivide the work among processes
        n_pixels_per_process = d1*d2/NCPUS

        options = cse.utilities.CNMFSetParms(
            Y, NCPUS, K=nrois_init, p=p, gSig=[9, 9], ssub=1, tsub=1)
        options['preprocess_params']['n_processes'] = NCPUS
        options['preprocess_params'][
            'n_pixels_per_process'] =  n_pixels_per_process
        options['init_params']['nIter'] = 10
        options['init_params']['maxIter'] = 10
        options['init_params']['use_hals'] = True
        options['spatial_params'][
            'n_pixels_per_process'] = n_pixels_per_process
        options['temporal_params'][
            'n_pixels_per_process'] = n_pixels_per_process

        cse.utilities.start_server()

        t0 = time.time()
        sys.stdout.write("Preprocessing... ")
        sys.stdout.flush()
        Yr, sn, g, psx = cse.preprocess_data(Yr, **options['preprocess_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 224.94s
        # 2016-05-24: 146.30s

        t0 = time.time()
        sys.stdout.write("Initializing components... ")
        sys.stdout.flush()
        Ain, Cin, b_in, f_in, center = cse.initialize_components(
            Y, **options['init_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 2281.37s
        # 2016-05-24: 1054.72s

        t0 = time.time()
        sys.stdout.write("Updating spatial components... ")
        sys.stdout.flush()
        A, b, Cin = cse.update_spatial_components(
            Yr, Cin, f_in, Ain, sn=sn, dview=dview, **options['spatial_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 252.57s
        # 2016-05-24: 445.95s

        t0 = time.time()
        sys.stdout.write("Updating temporal components... ")
        sys.stdout.flush()
        C, f, S, bl, c1, neurons_sn, g, YrA = \
            cse.update_temporal_components(
                Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None,
                dview=dview, **options['temporal_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 455.14s
        # 2016-05-24: 86.10s

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
        # 2016-05-24: 11.75s

        t0 = time.time()
        sys.stdout.write("Updating spatial components... ")
        sys.stdout.flush()
        A2, b2, C2 = cse.update_spatial_components(
            Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 77.16s
        # 2016-05-24: 99.22s

        t0 = time.time()
        sys.stdout.write("Updating temporal components... ")
        sys.stdout.flush()
        C2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = \
            cse.update_temporal_components(
                Yr, A2, b2, C2, f, bl=None, c1=None, sn=None, g=None,
                **options['temporal_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 483.41s
        # 2016-05-24: 74.81s

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

    return rois, C2, zproj, S2, Y, YrA


def process_data_patches(
        haussio_data, mask=None, p=2, nrois_init=400, roi_iceberg=0.9):
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
        fname_new = get_mmap_name(haussio_data.dirname_comp, d1, d2, T)
        Yr, _, _ = cse.utilities.load_memmap(fname_new)
        sys.stdout.write('took {0:.2f} s\n'.format(time.time()-t0))

        # how to subdivide the work among processes
        n_pixels_per_process = d1*d2/NCPUS_PATCHES

        sys.stdout.flush()
        cse.utilities.stop_server()
        cse.utilities.start_server()
        cl = Client()
        dview = cl[:NCPUS_PATCHES]

        rf = int(np.ceil(np.sqrt(d1*d2/4/NCPUS_PATCHES)))  # half-size of the patches in pixels. rf=25, patches are 50x50
        sys.stdout.write("Patch size: {0} * {0} = {1}\n".format(rf*2, rf*rf*4))
        stride = int(rf/5)  # amounpl of overlap between the patches in pixels

        t0 = time.time()
        sys.stdout.write("CNMF patches... ")
        sys.stdout.flush()
        options_patch = cse.utilities.CNMFSetParms(
            Y, NCPUS_PATCHES, p=0, gSig=[16, 16], K=nrois_init/NCPUS_PATCHES,
            ssub=1, tsub=8, thr=0.8)
        A_tot, C_tot, b, f, sn_tot, opt_out = cse.map_reduce.run_CNMF_patches(
            fname_new, (d1, d2, T), options_patch, rf=rf, stride=stride,
            dview=dview, memory_fact=4.0)
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))

        options = cse.utilities.CNMFSetParms(
            Y, NCPUS_PATCHES, K=A_tot.shape[-1], p=p, gSig=[16, 16], ssub=1, tsub=1)
        pix_proc = np.minimum(
            np.int((d1*d2)/NCPUS_PATCHES/(T/2000.)),
            np.int((d1*d2)/NCPUS_PATCHES))  # regulates the amount of memory used
        options['spatial_params']['n_pixels_per_process'] = pix_proc
        options['temporal_params']['n_pixels_per_process'] = pix_proc

        t0 = time.time()
        sys.stdout.write("Merging ROIs... ")
        sys.stdout.flush()
        A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = \
            cse.merge_components(
                Yr, A_tot, [], np.array(C_tot), [], np.array(C_tot), [],
                options['temporal_params'],
                options['spatial_params'], dview=dview,
                thr=options['merging']['thr'], mx=np.Inf)
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))

        options['temporal_params']['p']=0
        options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
        options['temporal_params']['backend']='ipyparallel'
        
        t0 = time.time()
        sys.stdout.write("Updating temporal components... ")
        sys.stdout.flush()
        C_m, f_m, S_m, bl_m, c1_m, neurons_sn_m, g2_m, YrA_m = \
            cse.temporal.update_temporal_components(
                Yr, A_m, np.atleast_2d(b).T, C_m, f, dview=dview,
                bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 483.41s
        # 2016-05-24: 74.81s

        t0 = time.time()
        sys.stdout.write("Evaluating components... ")
        sys.stdout.flush()
        traces = C_m+YrA_m
        idx_components, fitness, erfc = cse.utilities.evaluate_components(
            traces, N=5, robust_std=False)
        idx_components = idx_components[np.logical_and(True ,fitness < -10)]
        A_m = A_m[:,idx_components]
        C_m = C_m[idx_components,:]
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))

        t0 = time.time()
        sys.stdout.write("Updating spatial components... ")
        sys.stdout.flush()
        A2, b2, C2 = cse.spatial.update_spatial_components(
            Yr, C_m, f, A_m, sn=sn_tot, dview=dview, **options['spatial_params'])
        sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))
        # 77.16s
        # 2016-05-24: 99.22s

        options['temporal_params']['p']=p
        options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
        C2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = \
            cse.temporal.update_temporal_components(
                Yr, A2, b2, C2, f, dview=dview, bl=None, c1=None, sn=None,
                g=None, **options['temporal_params'])

        quality_threshold = -20
        traces = C2+YrA
        idx_components, fitness, erfc = cse.utilities.evaluate_components(
            traces, N=5, robust_std=False)
        idx_components = idx_components[fitness < quality_threshold]

        A2 = A2.tocsc()[:, idx_components]
        C2 = C2[idx_components, :]
        YrA = YrA[idx_components, :]
        S2 = S2[idx_components, :]

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

    # t0 = time.time()
    # sys.stdout.write("Ordering components... ")
    # sys.stdout.flush()
    # A_or, C_or, srt = cse.order_components(A2, C2)
    # sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))

    cse.utilities.stop_server()

    polygons = contour(A2, d1, d2, thr=roi_iceberg)
    rois = ROIList([sima.ROI.ROI(polygons=poly) for poly in polygons])

    return rois, C2, zproj, S2, Y, YrA


def contour(A, d1, d2, thr=None):
    # Adopted from https://github.com/agiovann/Constrained_NMF
    from scipy.sparse import issparse
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    d, nr = np.shape(A)

    x, y = np.mgrid[:d1:1, :d2:1]

    coordinates = []
    for i in range(nr):
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, (d1, d2), order='C')
        cntr = _cntr.Cntr(y, x, Bmat)
        cs = cntr.trace(thr)
        if len(cs) > 0:
            coordinates.append(cs[0])
        else:
            coordinates.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return coordinates
