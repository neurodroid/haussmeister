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
        from caiman.utils.visualization import plot_contours,view_patches_bar
        from caiman.base.rois import extract_binary_masks_blob
        import caiman.source_extraction.cnmf as caiman_cnmf
    except ImportError:
        sys.stderr.write("Could not find caiman module")

NCPUS = mp.cpu_count()
NCPUS_PATCHES = 16


def get_mmap_name(basename, d1, d2, T):
    bn = os.path.basename(os.path.normpath(basename))
    trunk = os.path.dirname(basename)
    new_path = os.path.join(trunk, bn.replace('_', ''))
    return new_path + \
        '_d1_' + str(1) + '_d2_' + str(d1) + '_d3_' + \
        str(d2) + '_order_' + 'C' + '_frames_' + str(T) + '_.mmap'


def tiffs_to_cnmf(haussio_data, mask=None, force=False):
    tmpdirname_comp = os.path.join(tempfile.gettempdir(),
                                   haussio_data.dirname_comp)
    try:
        os.makedirs(os.path.dirname(tmpdirname_comp))
    except OSError:
        pass
    mmap_files = glob.glob(
        tmpdirname_comp.replace('_', '') + "*_.mmap")

    if len(mmap_files) == 0 or force:
        sys.stdout.write('Converting to {0}... '.format(
            tmpdirname_comp + '_Y*.npy'))
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
        np.save(tmpdirname_comp + '_Y', tiff_data)

        fname_tot = get_mmap_name(tmpdirname_comp, d1, d2, T)
        big_mov = np.memmap(
            fname_tot, mode='w+',
            dtype=np.float32, shape=(d1*d2, T), order='F')
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
        fn_mmap = get_mmap_name('Yr', shape[2], shape[3], shape[0])
    else:
        d1, d2 = shape[1], shape[2]
        fn_mmap = get_mmap_name('Yr', shape[1], shape[2], shape[0])
    fn_mmap = os.path.join(haussio_data.dirname_comp, fn_mmap)
    print(fn_mmap, os.path.exists(fn_mmap), d1, d2)

    if not os.path.exists(fn_cnmf):
        # fn_raw = os.path.join(haussio_data.dirname_comp, haussio.THOR_RAW_FN)
        fn_sima = haussio_data.dirname_comp + '.sima'
        fnames = [fn_sima, ]
        fnames.sort()
        print(fnames)
        fnames = fnames

        final_frate = 1.0/haussio_data.dt
        downsample_factor = 1 # use .2 or .1 if file is large and you want a quick answer
        final_frate *= downsample_factor

        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)

        idx_xy = None
        base_name = 'Yr'
        name_new = cm.save_memmap_each(
            fnames, dview=dview, base_name=base_name,
            resize_fact=(1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy)
        name_new.sort()
        print(name_new)
        
        if len(name_new) > 1:
            fname_new = cm.save_memmap_join(
                name_new, base_name='Yr', n_chunks=12, dview=dview)
        else:
            sys.stdout.write('One file only, not saving\n')
            fname_new = name_new[0]

        print("fname_new: " + fname_new)

        Yr, dims, T = cm.load_memmap(fname_new)
        Y = np.reshape(Yr, dims+(T,), order='F')
        Cn = cm.local_correlations(Y)

        K = nrois_init # number of neurons expected per patch
        gSig = [15, 15] # expected half size of neurons
        merge_thresh = 0.8 # merging threshold, max correlation allowed
        p=2 #order of the autoregressive system
        options = caiman_cnmf.utilities.CNMFSetParms(
            Y, NCPUS, p=p, gSig=gSig, K=K, ssub=2, tsub=2)

        Yr, sn, g, psx = caiman_cnmf.pre_processing.preprocess_data(
            Yr, dview=dview, **options['preprocess_params'])
        Atmp, Ctmp, b_in, f_in, center = caiman_cnmf.initialization.initialize_components(
            Y, **options['init_params'])

        Ain, Cin = Atmp, Ctmp
        A,b,Cin = caiman_cnmf.spatial.update_spatial_components(
            Yr, Cin, f_in, Ain, sn=sn, dview=dview, **options['spatial_params'])

        options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
        C, f, S, bl, c1, neurons_sn, g, YrA = caiman_cnmf.temporal.update_temporal_components(
            Yr, A, b, Cin, f_in, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

        A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = caiman_cnmf.merging.merge_components(
            Yr, A, b, C, f, S, sn, options['temporal_params'],
            options['spatial_params'], dview=dview, bl=bl, c1=c1, sn=neurons_sn,
            g=g, thr=merge_thresh, mx=50, fast_merge=True)

        A2, b2, C2 = caiman_cnmf.spatial.update_spatial_components(
            Yr, C_m, f, A_m, sn=sn, dview=dview, **options['spatial_params'])
        options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
        C2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = caiman_cnmf.temporal.update_temporal_components(
            Yr, A2, b2, C2, f, dview=dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

        tB = np.minimum(-2,np.floor(-5./30*final_frate))
        tA = np.maximum(5,np.ceil(25./30*final_frate))
        Npeaks = 10
        traces = C2+YrA
        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
            evaluate_components(
                Y, traces, A2, C2, b2, f2, final_frate, remove_baseline=True, N=5,
                robust_std=False, Athresh=0.1, Npeaks=Npeaks, thresh_C=0.3)

        idx_components_r=np.where(r_values >= .6)[0]
        idx_components_raw=np.where(fitness_raw < -60)[0]
        idx_components_delta=np.where(fitness_delta < -20)[0]

        min_radius=gSig[0]-2
        masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
            A2.tocsc(), min_radius, dims, num_std_threshold=1,
            minCircularity= 0.6, minInertiaRatio = 0.2,minConvexity =.8)

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_blobs = np.intersect1d(idx_components, idx_blobs)
        idx_components_bad = np.setdiff1d(range(len(traces)), idx_components)

        A2 = A2.tocsc()[:, idx_components]
        C2 = C2[idx_components, :]
        YrA = YrA[idx_components, :]
        S2 = S2[idx_components, :]

        # A: spatial components (ROIs)
        # C: denoised [Ca2+]
        # YrA: residuals ("noise", i.e. traces = C+YrA)
        # S: Spikes
        savemat(fn_cnmf, {"A": A2, "C": C2, "YrA": YrA, "S": S2, "bl": bl2})

    else:
        resdict = loadmat(fn_cnmf)
        A2 = resdict["A"]
        C2 = resdict["C"]
        YrA = resdict["YrA"]
        S2 = resdict["S"]
        bl2 = resdict["bl"]
        Yr, dims, T = cm.load_memmap(fn_mmap)
        dims = dims[1:]
        Y = np.reshape(Yr, dims+(T,), order='F')

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

    cm.stop_server()

    polygons = contour(A2, Y.shape[0], Y.shape[1], thr=roi_iceberg)
    rois = ROIList([sima.ROI.ROI(polygons=poly) for poly in polygons])

    return rois, C2, zproj, S2, Y, YrA


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
