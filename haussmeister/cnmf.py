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

try:
    import ipyparallel
    from ipyparallel import Client
    HAS_IPYPARALLEL = True
except ImportError:
    print("ipyparallel unavailable")
    HAS_IPYPARALLEL = False

import numpy as np
from scipy.io import savemat, loadmat

from contours.core import shapely_formatter as shapely_fmt
from contours.quad import QuadContourGenerator

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
            tiff_data = tiff_sequence.asarray(memmap=True)
        else:
            if mask is not None:
                tiff_data = haussio_data.read_raw().squeeze()[
                    np.invert(mask), :, :]
            else:
                tiff_data = haussio_data.read_raw().squeeze()

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


def process_data(haussio_data, mask=None, p=2, nrois_init=400, roi_iceberg=0.9, merge_unconnected=None):
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
            backend='multiprocessing', n_processes=NCPUS, single_thread=False)

        Yr, dims, T = cm.load_memmap(fn_mmap, 'r+')
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        fr = 1.0/haussio_data.dt    # imaging rate in frames per second\n",
        decay_time = 0.4                    # length of a typical transient in seconds\n",

        # parameters for source extraction and deconvolution\n",
        bord_px_els = 32            # maximum shift to be used for trimming against NaNs
        p = 1                       # order of the autoregressive system\n",
        gnb = 2                     # number of global background components\n",
        merge_thresh = 0.8          # merging threshold, max correlation allowed\n",
        rf = int(np.round(np.sqrt(d1*d2)/nrois_init)) # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50\n",
        if rf < 16:
            rf = 16
        stride_cnmf = 6             # amount of overlap between the patches in pixels\n",
        npatches = np.round(d1/(rf*2) * d2/(rf*2))
        K = nrois_init/npatches     # number of components per patch\n",
        if K < 2:
            K = 2
        print(rf, npatches, K)
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
        
        if merge_unconnected is not None:
            idx_merge = []
            for nroi, ca_roi in enumerate(cnm2.C):
                for nroi_compare_counter, ca_roi_compare in enumerate(cnm2.C[nroi+1:]):
                    nroi_compare = nroi_compare_counter+nroi+1
                    if nroi_compare not in idx_merge:
                        correls = np.correlate(ca_roi, ca_roi_compare, mode='same')
                        correls /= np.sqrt(np.dot(ca_roi, ca_roi) * np.dot(ca_roi_compare, ca_roi_compare))
                        if correls.max() > merge_unconnected:
                            idx_merge.append(nroi_compare)
            idx_no_merge = [idx for idx in range(cnm2.C.shape[0]) if idx not in idx_merge]
        else:
            idx_no_merge = range(cnm2.C.shape[0])
        A2 = cnm2.A[:, idx_no_merge].tocsc()
        C2 = cnm2.C[idx_no_merge]
        YrA = cnm2.YrA[idx_no_merge]
        S2 = cnm2.S[idx_no_merge]

        # A: spatial components (ROIs)
        # C: denoised [Ca2+]
        # YrA: residuals ("noise", i.e. traces = C+YrA)
        # S: Spikes
        # f: temporal background
        savemat(fn_cnmf, {"A": A2, "C": C2, "YrA": YrA, "S": S2, "bl": cnm2.b, "f": cnm2.f})
        dview.terminate()
        cm.stop_server()
    else:
        resdict = loadmat(fn_cnmf)
        A2 = resdict["A"]
        C2 = resdict["C"]
        YrA = resdict["YrA"]
        S2 = resdict["S"]
        bl2 = resdict["bl"]
        f = resdict["f"]
        images = haussio_data.read_raw().squeeze()

    proj_fn = haussio_data.dirname_comp + "_proj.npy"
    if not os.path.exists(proj_fn):
        zproj = utils.zproject(images)
        np.save(proj_fn, zproj)
    else:
        zproj = np.load(proj_fn)

    logfiles = glob.glob("*LOG*")
    for logfile in logfiles:
        try:
            os.unlink(logfile)
        except OSError:
            pass

    print(images.shape[1], images.shape[2])
    polygons = contour(A2, images.shape[1], images.shape[2], thr=roi_iceberg)
    rois = ROIList([sima.ROI.ROI(polygons=poly) for poly in polygons])

    return rois, C2, zproj, S2, images, YrA


def contour(A, d1, d2, thr=None):
    # Adopted from https://github.com/agiovann/Constrained_NMF
    from scipy.sparse import issparse
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    d, nr = np.shape(A)

    # x, y = np.mgrid[:d1:1, :d2:1]
    x = np.arange(d1)
    y = np.arange(d2)
    coordinates = []
    for i in range(nr):
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, (d1, d2), order='F')
        Bmat = 1-Bmat
        c = QuadContourGenerator.from_rectilinear(y, x, Bmat, shapely_fmt)
        cs = c.filled_contour(min=1-thr, max=None)
        # cs = cntr.trace(thr)
        if len(cs) > 0:
            for csn in cs:
                print(list(csn.exterior.coords))
            coordinates.append(cs)
        else:
            sys.stdout.write("No polygon found\n")
            coordinates.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return coordinates
