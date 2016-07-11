"""
Module for computing spatial maps

(c) 2016 C. Schmidt-Hieber
GPLv3
"""

import numpy as np
from numpy import linalg as LA


def map2d(times, data, posx, posy, xbinsize=1, ybinsize=1):

    xdatarange = np.arange(np.min(posx), np.max(posy), xbinsize)
    ydatarange = np.arange(np.min(posy), np.max(posy), ybinsize)

    xgrid, ygrid = np.meshgrid(xdatarange, ydatarange, indexing='ij')
    xydata = data[:, 1::-1]
    xyflat = np.array([xgrid.flatten(), ygrid.flatten()]).T
    distances = np.array(
        [LA.norm(xydata-xypair, axis=1)
         for xypair in xyflat]).reshape(NPX, NPX, xydata.shape[0])
