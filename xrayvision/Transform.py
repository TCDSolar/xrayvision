"""Transform class

"""

import numpy as np


class Transform(object):
    """
    Fourier transformation class

    Parameters
    ----------

    Examples
    --------

    Notes
    -----

    """
    @staticmethod
    def dft(im: np.array, uv: np.array, vis: np.array):
        """
        Discrete Fourier Transform loops over a list of [x, y] pixels rather than looping over
        x and y separately

        Parameters
        ----------
        im :  ndarray
            Input image
        uv : np.array
            The u, v coordinates where visibilities are needed
        vis: np.array
            An array where visibilities can be stored for u, v coordinates
        Returns
        -------
        vis : ndarray
            The visibilities

        """
        m, n = im.shape
        size = im.size
        xy = np.mgrid[0:m, 0:n].reshape(2, size)
        for i in range(uv.shape[1]):
            vis[i] = np.sum(
                im.reshape(size) * np.exp(
                    -2j * np.pi * (uv[0, i] * xy[0, :] / m + uv[1, i] * xy[1, :] / n)))

        return vis

    @staticmethod
    def idft(im: np.array, uv: np.array, vis: np.array):
        """
        Inverse Discrete Fourier Transform loops over a list of [x, y] pixels rather than looping
        over x and y separately

        Parameters
        ----------
        im :  ndarray
            Place holder image
        uv : np.array
            The u, v coordinates where visibilities are known
        vis: np.array
            An array which stores the visibilities for u, v coordinates

        Returns
        -------
        out : ndarray
            The inverse transform or back projection

        """
        m, n = im.shape
        size = im.size
        out = np.zeros(m * n)
        xy = np.mgrid[0:m, 0:n].reshape(2, size)
        for i in range(im.size):
            out[i] = (1 / vis.size) * np.sum(
                vis * np.exp(
                    2j * np.pi * (uv[0, :] * xy[0, i] / m + uv[1, :] * xy[1, i] / n)))

        return out.reshape(m, n)
