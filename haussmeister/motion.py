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
from sima import motion
try:
    sys.path.append(os.path.expanduser("~/CaImAn/"))
    import caiman as cb
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
                frames = np.concatenate([
                    np.array(sequence[
                        nframe:nframe+1000, plane_idx, :, :, 0]).astype(
                            np.float32)
                    for nframe in range(0, sequence.shape[0], 1000)])
                frames = np.squeeze(frames)
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
        verbose = self._params['verbose']
        if self._params['max_displacement'] is None:
            max_displacement = (1e15, 1e15)
        else:
            max_displacement = self._params['max_displacement']

        displacements = []
        c, dview, n_processes = cb.cluster.setup_cluster(
            backend = 'local', n_processes = None, single_thread = False)
    
        num_iter = 1 # number of times the algorithm is run
        splits = 56 # for parallelization split the movies in  num_splits chuncks across time
        num_splits_to_process = None # if none all the splits are processed and the movie is saved
        shifts_opencv = True # apply shifts fast way (but smoothing results)
        save_movie_rigid = False # save the movies vs just get the template

        for sequence in dataset:
            t0 = time.time()
            num_frames = sequence.shape[0]
            num_planes = sequence.shape[1]
            num_channels = sequence.shape[4]
            if num_channels > 1:
                raise NotImplementedError("Error: only one colour channel \
                    can be used for DFT motion correction. Using channel 1.")

            # get results into a shape sima likes
            frame_shifts = np.zeros((num_frames, num_planes, 2))
            for plane_idx in range(num_planes):
                fname_tot_rig, total_template_rig, templates_rig, shifts_rig = \
                    cb.motion_correction.motion_correct_batch_rigid(
                        self._params['savedir'], max_displacement, dview = dview, splits = splits,
                        num_splits_to_process = num_splits_to_process, num_iter = num_iter,
                        template = None, shifts_opencv = shifts_opencv,
                        save_movie_rigid = save_movie_rigid)

                frame_shifts[:, plane_idx] = shifts_rig

            displacements.append(np.round(frame_shifts).astype(np.int))

            total_time = time.time() - t0
            if verbose:
                print('    Total time for plane ' + str(plane_idx+1) + ': ' +
                      str(total_time) + ' s')

        return displacements
