"""
Motion correction using calblitz
https://github.com/agiovann/CalBlitz

(c) 2016 C. Schmidt-Hieber
GPLv3
"""

import time
import numpy as np
from sima import motion
import calblitz as cb


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

            displacements.append(frame_shifts)

            total_time = time.time() - t0
            if verbose:
                print('    Total time for plane ' + str(plane_idx+1) + ': ' +
                      str(total_time) + ' s')

        return displacements
