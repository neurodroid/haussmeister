"""
Motion correction using calblitz
https://github.com/agiovann/CalBlitz

(c) 2016 C. Schmidt-Hieber
GPLv3
"""

import time
import sys
import os
import numpy as np
import tempfile
import multiprocessing as mp
import glob
from sima import motion
try:
    sys.path.append(os.path.expanduser("~/CaImAn/"))
    import caiman as cb
    from caiman.motion_correction import MotionCorrect
except ImportError:
    print("CaiMan import failed")
from ipyparallel import Client


class CalBlitz(motion.MotionEstimationStrategy):
    """
    Motion correction using CalBlita. See
    https://github.com/agiovann/CalBlitz

    Parameters
    ----------
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. Default: None.
    fr : float
        frame rate. CalBlitz insists on knowing the frame rate. Default: 30
    verbose : bool, optional
        enable verbose mode. Default: False.
    """
    def __init__(self, max_displacement=None, fr=30.0, verbose=False):
        self._params = dict(locals())
        del self._params['self']

    def _estimate(self, dataset):
        """

        Parameters
        ----------

        Returns
        -------
        displacements : array
            (2, num_frames*num_cycles)-array of integers giving the
            estimated displacement of each frame
        """
        verbose = self._params['verbose']
        if self._params['max_displacement'] is None:
            max_displacement = (1e15, 1e15)
        else:
            max_displacement = self._params['max_displacement']

        frame_step = 1000
        displacements = []
        for sequence in dataset:
            num_frames = sequence.shape[0]
            num_planes = sequence.shape[1]
            num_channels = sequence.shape[4]
            if num_channels > 1:
                raise NotImplementedError("Error: only one colour channel \
                    can be used for DFT motion correction. Using channel 1.")

            # get results into a shape sima likes
            frame_shifts = np.zeros((num_frames, num_planes, 2))
            for plane_idx in range(num_planes):
                # load into memory... need to pass numpy array to dftreg.
                # could(should?) rework it to instead accept tiff array
                if verbose:
                    print('Loading plane ' + str(plane_idx+1) + ' of ' +
                          str(num_planes) + ' into numpy array')
                t0 = time.time()
                # reshape, one plane at a time
                memmap_file = tempfile.NamedTemporaryFile()
                frames = np.memmap(
                    memmap_file.name, dtype=np.float32, mode='w+', shape=(
                        sequence.shape[0], sequence.shape[2], sequence.shape[3]))
                for nframe in range(0, sequence.shape[0], frame_step):
                    frames[nframe:nframe+frame_step] = np.array(sequence[
                        nframe:nframe+frame_step, plane_idx, :, :, 0]).astype(np.float32).squeeze()
                e1 = time.time() - t0
                if verbose:
                    print('    Loaded in: ' + str(e1) + ' s')

                m = cb.movie(frames, fr=self._params['fr'])
                m, shifts, xcorrs, template = m.motion_correct(
                    max_shift_w=max_displacement[0],
                    max_shift_h=max_displacement[1],
                    num_frames_template=None,
                    template=None,
                    method='opencv')

                frame_shifts[:, plane_idx] = shifts
                memmap_file.close()

            displacements.append(np.round(frame_shifts).astype(np.int))

            total_time = time.time() - t0
            if verbose:
                print('    Total time for plane ' + str(plane_idx+1) + ': ' +
                      str(total_time) + ' s')

        return displacements


class NormCorr(motion.MotionEstimationStrategy):
    """
    Motion correction using NormCorr. See
    https://github.com/agiovann/CalBlitz

    Parameters
    ----------
    max_displacement : array of int, optional
        The maximum allowed displacement magnitudes in [y,x]. Default: None.
    fr : float
        frame rate. CalBlitz insists on knowing the frame rate. Default: 30
    verbose : bool, optional
        enable verbose mode. Default: False.
    """
    def __init__(self, max_displacement=None, fr=30.0, verbose=False, savedir=None):
        self._params = dict(locals())
        del self._params['self']

    def _estimate(self, dataset):
        """

        Parameters
        ----------

        Returns
        -------
        displacements : array
            (2, num_frames*num_cycles)-array of integers giving the
            estimated displacement of each frame
        """
        ncpus = mp.cpu_count()
        verbose = self._params['verbose']
        if self._params['max_displacement'] is None:
            max_displacement = (1e15, 1e15)
        else:
            max_displacement = self._params['max_displacement']

        displacements = []
        if 'dview' in locals():
            dview.terminate()
        # c, dview, n_processes = cb.cluster.setup_cluster(
        #     backend='local', n_processes=ncpus, single_thread=False)
        dview = None
        num_iter = 3 # number of times the algorithm is run
        splits = ncpus*2 # for parallelization split the movies in  num_splits chuncks across time
        shifts_opencv = True # apply shifts fast way (but smoothing results)
        save_movie_rigid = False # save the movies vs just get the template
        upsample_factor_grid = 4
        max_deviation_rigid = 3
        for sequence in dataset:
            t0 = time.time()
            num_frames = sequence.shape[0]
            num_planes = sequence.shape[1]
            num_channels = sequence.shape[4]
            strides = (int(sequence.shape[2]/10.0), int(sequence.shape[3]/10.0))
            overlaps = (int(sequence.shape[2]/20.0), int(sequence.shape[3]/20.0))
            if num_channels > 1:
                raise NotImplementedError("Error: only one colour channel \
                    can be used for DFT motion correction. Using channel 1.")

            # get results into a shape sima likes
            frame_shifts = np.zeros((num_frames, num_planes, 2))
            for plane_idx in range(num_planes):
                min_mov = np.min(sequence[:100,:,:,:,:])
                mc = MotionCorrect(
                    self._params['savedir'], min_mov, dview=dview,
                    max_shifts=max_displacement, niter_rig=num_iter,
                    splits_rig=splits, strides=strides,
                    overlaps=overlaps, splits_els=splits,
                    upsample_factor_grid=upsample_factor_grid,
                    max_deviation_rigid=max_deviation_rigid,
                    shifts_opencv=shifts_opencv, nonneg_movie=True)
                mc.motion_correct_rigid(save_movie=save_movie_rigid)
                # fname_tot_rig, total_template_rig, templates_rig, shifts_rig = \
                #     cb.motion_correction.motion_correct_batch_rigid(
                #         self._params['savedir'], max_displacement, dview = dview, splits = splits,
                #         num_splits_to_process = num_splits_to_process, num_iter = num_iter,
                #         template = None, shifts_opencv = shifts_opencv,
                #         save_movie_rigid = save_movie_rigid)

                frame_shifts[:, plane_idx] = mc.shifts_rig # (mc.x_shifts_els, mc.y_shifts_els)# shifts_rig

            displacements.append(np.round(frame_shifts).astype(np.int))

            total_time = time.time() - t0
            if verbose:
                print('    Total time for plane ' + str(plane_idx+1) + ': ' +
                      str(total_time) + ' s')

        cb.cluster.stop_server()
        return displacements
