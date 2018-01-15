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

from haussmeister import haussio

try:
    from . import utils
except ValueError:
    import utils

if sys.version_info.major < 3:
    sys.path.append(os.path.expanduser("~/CaImAn/"))
    try:
        import caiman as cm
        from caiman.components_evaluation import evaluate_components
        from caiman.components_evaluation import estimate_components_quality_auto
        from caiman.utils.visualization import plot_contours,view_patches_bar
        from caiman.base.rois import extract_binary_masks_blob
        import caiman.source_extraction.cnmf as caiman_cnmf
    except ImportError:
        sys.stderr.write("Could not find caiman module")

NCPUS = mp.cpu_count()
NCPUS_PATCHES = 16


def get_mmap_name(basename, d1, d2, T, d0=1):
    bn = os.path.basename(os.path.normpath(basename))
    trunk = os.path.dirname(basename)
    new_path = os.path.join(trunk, bn.replace('_', ''))
    return new_path + \
        '_d1_' + str(d1) + '_d2_' + str(d2) + '_d3_' + \
        str(d0) + '_order_' + 'C' + '_frames_' + str(T) + '_.mmap'


def tiffs_to_cnmf(haussio_data, mask=None, force=False):
    mmap_files = glob.glob(haussio_data.dirname_comp + os.path.sep + "Yr*.mmap")

    if len(mmap_files) == 0 or force:
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
        # np.save(tmpdirname_comp + '_Y', tiff_data)
        fname_tot = get_mmap_name(haussio_data.dirname_comp + os.path.sep + 'Yr', d1, d2, T)

        big_mov = np.memmap(
            fname_tot, mode='w+',
            dtype=np.float32, shape=(d1*d2, T), order='C')
        big_mov[:] = np.reshape(tiff_data, (d1*d2, T), order='F')[:]
        big_mov.flush()

        del tiff_data
        del big_mov

        sys.stdout.write('took {0:.2f} s\n'.format(time.time()-t0))
        # 888s


def process_data(haussio_data, mask=None, p=2, nrois_init=400, roi_iceberg=0.9):
    if mask is not None:
        raise RuntimeError("mask not supported in cnmf.process_data")

    fn_cnmf = haussio_data.dirname_comp + '_cnmf.mat'
    shapefn = os.path.join(
        haussio_data.dirname_comp, haussio.THOR_RAW_FN[:-3] + "shape.npy")
    shape = np.load(shapefn)
    if len(shape) == 5:
        d1, d2 = shape[2], shape[3]
    else:
        d1, d2 = shape[1], shape[2]
    fn_mmap = get_mmap_name(haussio_data.dirname_comp + os.path.sep + 'Yr', d1, d2, shape[0])
    
    tiffs_to_cnmf(haussio_data)
    if not os.path.exists(fn_cnmf):

        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='multiprocessing', n_processes=None, single_thread=False)

        Yr, dims, T = cm.load_memmap(fn_mmap, 'r+')
        print(fn_mmap, dims, T)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        fr = 1.0/haussio_data.dt    # imaging rate in frames per second\n",
        decay_time = 0.4                    # length of a typical transient in seconds\n",

        # parameters for source extraction and deconvolution\n",
        bord_px_els = 32            # maximum shift to be used for trimming against NaNs
        p = 1                       # order of the autoregressive system\n",
        gnb = 2                     # number of global background components\n",
        merge_thresh = 0.8          # merging threshold, max correlation allowed\n",
        rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50\n",
        stride_cnmf = 6             # amount of overlap between the patches in pixels\n",
        K = nrois_init/n_processes  # number of components per patch\n",
        gSig = [8, 8]               # expected half size of neurons\n",
        init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')\n",
        is_dendrites = False        # flag for analyzing dendritic data\n",
        alpha_snmf = None           # sparsity penalty for dendritic data analysis through sparse NMF\n",

        # parameters for component evaluation\n",
        min_SNR = 2.5               # signal to noise ratio for accepting a component\n",
        rval_thr = 0.8              # space correlation threshold for accepting a component\n",
        cnn_thr = 0.8               # threshold for CNN based classifier"

        cnm = caiman_cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                        p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                        method_init=init_method, alpha_snmf=alpha_snmf,
                        only_init_patch = False, gnb = gnb, border_pix = bord_px_els)
        cnm = cnm.fit(images)

        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
            estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f,
                                             cnm.YrA, fr, decay_time, gSig, dims,
                                             dview = dview, min_SNR=min_SNR,
                                             r_values_min = rval_thr, use_cnn = False,
                                             thresh_cnn_lowest = cnn_thr)
        A_in, C_in, b_in, f_in = cnm.A[:,idx_components], cnm.C[idx_components], cnm.b, cnm.f
        cnm2 = caiman_cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                         merge_thresh=merge_thresh,  Ain=A_in, Cin=C_in, b_in = b_in,
                         f_in=f_in, rf = None, stride = None, gnb = gnb,
                         method_deconvolution='oasis', check_nan = True)
        cnm2 = cnm2.fit(images)
        
        A2 = cnm2.A.tocsc()
        C2 = cnm2.C
        YrA = cnm2.YrA
        S2 = cnm2.S

        # A: spatial components (ROIs)
        # C: denoised [Ca2+]
        # YrA: residuals ("noise", i.e. traces = C+YrA)
        # S: Spikes
        print(dir(cnm2))
        savemat(fn_cnmf, {"A": cnm2.A.tocsc(), "C": cnm2.C, "YrA": cnm2.YrA, "S": cnm2.S, "bl": cnm2.b})

    else:
        resdict = loadmat(fn_cnmf)
        A2 = resdict["A"]
        C2 = resdict["C"]
        YrA = resdict["YrA"]
        S2 = resdict["S"]
        bl2 = resdict["bl"]
        images = haussio_data.read_raw().squeeze()

    proj_fn = haussio_data.dirname_comp + "_proj.npy"
    if not os.path.exists(proj_fn):
        zproj = utils.zproject(images)
        np.save(proj_fn, zproj)
    else:
        zproj = np.load(proj_fn)

    # DF_F, DF = cse.extract_DF_F(Y.reshape(d1*d2, T), A2, C2)

    # t0 = time.time()
    # sys.stdout.write("Ordering components... ")
    # sys.stdout.flush()
    # A_or, C_or, srt = cse.order_components(A2, C2)
    # sys.stdout.write(' took {0:.2f} s\n'.format(time.time()-t0))

    cm.stop_server()

    logfiles = glob.glob("*LOG*")
    for logfile in logfiles:
        os.unlink(logfile)

    polygons = contour(A2, images.shape[1], images.shape[2], thr=roi_iceberg)
    rois = ROIList([sima.ROI.ROI(polygons=poly) for poly in polygons])

    return rois, C2, zproj, S2, images, YrA


def process_data_patches(
        haussio_data, mask=None, p=2, nrois_init=400, roi_iceberg=0.9):
    fn_cnmf = haussio_data.dirname_comp + '_cnmf.mat'

    tiffs_to_cnmf(haussio_data, mask)
    tmpdirname_comp = os.path.join(tempfile.gettempdir(),
        haussio_data.dirname_comp)
    try:
        os.makedirs(os.path.dirname(tmpdirname_comp))
    except OSError:
        pass
    sys.stdout.write('Loading from {0}... '.format(
        tmpdirname_comp + '_Y*.npy'))
    Y = np.load(tmpdirname_comp + '_Y.npy', mmap_mode='r')
    d1, d2, T = Y.shape

    if not os.path.exists(fn_cnmf):

        cse.utilities.stop_server()

        sys.stdout.flush()
        t0 = time.time()
        fname_new = get_mmap_name(tmpdirname_comp, d1, d2, T)
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
        A_tot, C_tot, YrA_tot, b, f, sn_tot, opt_out = cse.map_reduce.run_CNMF_patches(
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
        options['temporal_params']['fudge_factor'] = 0.96 #change ifdenoised traces time constant is wrong
        options['temporal_params']['backend'] = 'ipyparallel'

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

        Npeaks = 10
        final_frate = 1.0/haussio_data.dt
        tB = np.minimum(-2, np.floor(-5./30*final_frate))
        tA = np.maximum(5, np.ceil(25./30*final_frate))
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, sign_sam =\
            cse.utilities.evaluate_components(
                Y, traces, A_m, C_m, b, f_m,
                remove_baseline=True, N=5, robust_std=False,
                Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

        idx_components_r = np.where(r_values >= .5)[0]
        idx_components_raw = np.where(fitness_raw <- 20)[0]
        idx_components_delta = np.where(fitness_delta < -10)[0]

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)

        A_m = A_m[:, idx_components]
        C_m = C_m[idx_components, :]
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

        traces = C2+YrA
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, sign_sam =\
            cse.utilities.evaluate_components(
                Y, traces, A2, C2, b2, f2, remove_baseline=True, N=5,
                robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB,
                tA=tA, thresh_C=0.3)
        idx_components_r = np.where(r_values >= .6)[0]
        idx_components_raw = np.where(fitness_raw < -60)[0]
        idx_components_delta = np.where(fitness_delta < -20)[0]
        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)

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
        Bmat = np.reshape(Bvec, (d1, d2), order='F')
        cntr = _cntr.Cntr(y, x, Bmat)
        cs = cntr.trace(thr)
        if len(cs) > 0:
            coordinates.append(cs[0])
        else:
            coordinates.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return coordinates
