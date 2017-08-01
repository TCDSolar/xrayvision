import numpy as np


__all__ = ['from_map', 'to_map']


class Visibility(object):
    """
    A set of visibility

    This frame has its origin at the solar centre and the north pole above the
    solar north pole, and the zero line on longitude pointing towards the
    Earth.

    Parameters
    ----------
    uv: `numpy.ndarray` The u, v coordinates of the visibilities
    vis: `numpy.ndarray` The complex visibility
    Examples
    --------

    Notes
    -----

    """

    def __init__(self, uv, vis):
        self.uv = np.array(uv)
        self.vis = np.array(vis, dtype=complex)

    def __repr__(self):
        print(self.uv, self.vis)

    def from_map(self, inmap):
        return self._dft_map(inmap)

    def to_map(self, outmap):
        return self._idft_map(outmap)

    def _dft_map(self, im):
        """
        Discrete Fourier Transform loops over a list of [x, y] pixel rather than looping over x and y separately

        Parameters
        ----------
        im :  ndarray
            Input image

        Returns
        -------
        vis : ndarray
            The visibilities

        """
        m, n = im.shape
        size = im.size
        xy = np.mgrid[0:m, 0:n].reshape(2, size)
        for i in range(self.uv.shape[1]):
            self.vis[i] = np.sum(
                im.reshape(size) * np.exp(-2j * np.pi * (self.uv[0, i] * xy[0, :] / m + self.uv[1, i] * xy[1, :] / n)))

        return self.vis

    def _idft_map(self, im):
        """
        Inverse Discrete Fourier Transform loops over a list of [x, y] pixel rather than looping over x and y separately

        Parameters
        ----------
        im :  ndarray
            Place holder image

        Returns
        -------
        out : ndarray
            The inverse transform or back projection

        """
        m, n = im.shape
        size = im.size
        out = np.zeros(m*n)
        xy = np.mgrid[0:m, 0:n].reshape(2, size)
        for i in range(im.size):
            out[i] = (1 / self.vis.size) * np.sum(
                self.vis * np.exp(2j * np.pi * (self.uv[0, :] * xy[0, i] / m + self.uv[1, :] * xy[1, i] / n)))

        return out.reshape(m, n)
