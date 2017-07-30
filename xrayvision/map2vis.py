"""Converts map_data into visibilities

Each visibility structure corresponds to a single subcollimator, harmonic,
u, v, energy and time combination.
Reference: https://darts.jaxa.jp/pub/ssw/hessi/idl/atest/hsi_vis_map2vis.pro
"""
#
# TODO
#
__authors__ = ["Gábor Péterffy"]
__email__ = ["peterffy95@gmail.com"]
__license__ = "xxx"

import numpy as np
import copy
from .hsi_vis import *


def hsi_vis_map2vis(map_data: np.array, xy: np.array, uv: np.array):
    """
    Returns with hsi_vis visibility array with the computed
    visibilities for the given point.

    Parameters
    ----------
    map_data : np.array
        2D array with the flux (must be square)
    xy : np.array
        Determine the resolution of the data
    uv : np.array
        The list of the u, v points where we want to get the data.

    Returns
    -------
    visout : hsi_vis visibility array

    See Also
    --------

    Notes
    -----
    """
    twopi = 2 * np.pi
## The count of the displacements
    nxy = xy.shape[1]
## The count of the uv coordinates where we want the visibilities
    nuv = uv.shape[1]
    if (map_data.shape[0] * map_data.shape[1] != nxy ** 2.0):
        raise ValueError("Dimension mismatch between map_data and xy!"
                         "Please read the docs.")
    visout = []
    spatfreq2 = 0.0
    for i in range(nuv):
        visout.append(hsi_vis())
        visout[-1].u = uv[0, i]
        visout[-1].v = uv[1, i]
        spatfreq2 += uv[0, i] ** 2.0 + uv[1, i] ** 2.0
    visout = np.array(visout)
    spatfreq2 *= 2.33 ** 2.0
    scn = np.round(-np.log(spatfreq2) / np.log(3.0))
    mapcenterx = np.average(xy[0, :])
    mapcentery = np.average(xy[1, :])
    for i in visout:
        i.isc = scn-1
        i.harm = 1
        i.chi2 = 1.
        i.xyoffset = np.array([mapcenterx, mapcentery])
        i.obsvis = 0+0j

    ok = np.nonzero(map_data)
    nok = len(ok[0])
    if nok == 0:
        raise ValueError("Empty map")
    for j, i in enumerate(ok[0]):
        phase = twopi * (uv[0, :] * (xy[0, i] - mapcenterx)
                         + uv[1, :] * (xy[1, ok[1][j]] - mapcentery))
        for k, l in enumerate(visout):
            l.obsvis += map_data[i, ok[1][j]] * (np.cos(phase[k]) +
                                                 1j * np.sin(phase[k]))
    sigamp = 0.
    for i in visout:
        tmp = np.abs(i.obsvis)
        if (tmp > sigamp):
            sigamp = tmp
    for i in visout:
        i.sigamp = sigamp
    return visout
