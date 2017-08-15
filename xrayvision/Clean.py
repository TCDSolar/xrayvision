"""CLEAN algorithms"""

import numpy as np


class Hogbom(object):
    """
    Simplest Hogböm's CLEAN method (in case of a dirty beam with just 1s
    where there is a visibility value.

    Parameters
    ----------
    visibility: Visibility
        The processed visibility bag
    threshold: float
        If the highest value is smaller, it stops
    gain: The gain what is used for the iterations

    Examples
    --------

    Notes
    -----

    """
    def __init__(self, visibility, threshold, gain=0.25):
        self.vis = visibility
        self.thres = threshold
        self.gain = gain

    def iterate(self, gain=False):
        """
        The count of iterations depends on, how many times it has been called.
        It will not iterate if the threshold is reached.

        Parameters
        ----------
        gain: float
            If not provided it will use the one what was given at init

        Examples
        --------

        Notes
        -----

        """
        max_intesity = 0.0
        pos = 0
        summarized = 0.0
        for i, j in enumerate(np.absolute(self.vis.vis)):
            if j > max_intesity:
                max_intesity = j
                pos = i
                summarized += j
        abs_avg_value = summarized / float(self.vis.vis[pos])
        if abs_avg_value < self.thres:
            return True
        if not gain:
            gain = self.gain
        clean_vis = complex(gain*max_intesity)
        for i in self.vis.vis:
            i -= clean_vis
        return False
