"""Visibility relates things.

Visibility.

"""

import numpy as np
import sunpy.map
from .Transform import Transform


class Visibility(object):
    """
    A set of visibilities.

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
        """

        Parameters
        ----------
        inmap

        Returns
        -------

        """
        return Transform.dft(inmap, self.uv, self.vis)

    def to_map(self, outmap):
        """

        Parameters
        ----------
        outmap

        Returns
        -------

        """
        return Transform.idft(outmap, self.uv, self.vis)

    def to_sunpy_map(self, outmap: sunpy.map):
        """
        Converts the visibility data into an actual image (specified by
        the given outmap sunpy map

        Parameters
        ----------
        outmap

        Returns
        -------

        """
        data = Transform.idft(outmap.data, self.uv, self.vis)
        outmap = sunpy.map.Map(data, outmap.meta)
        return outmap

    @staticmethod
    def generate_xy(number_pixels, pixel_size=1):
        """
        Generate the x or y image/map coordinates given the number of pixels and pixel size

        Parameters
        ----------
        number_pixels : int
            Number of pixels in the map

        pixel_size : float
            Size of pixel

        Returns
        -------
        `numpy.ndarray`
            The generated coordinates

        """
        if number_pixels % 2 == 0:
            x = np.linspace(-number_pixels * pixel_size / 2, (number_pixels / 2 - 1) * pixel_size,
                            number_pixels)
        else:
            x = np.linspace(-(number_pixels - 1) * pixel_size / 2,
                            (number_pixels - 1) * pixel_size / 2,
                            number_pixels)

        return x

    @staticmethod
    def dft_map(input_map, input_uv):
        """
        Calculate the visibilities for the given map using a discrete fourier transform

        Parameters
        ----------
        input_map : array-like
            Input map to be transformed

        input_uv : array-like
            The u, v coordinate to calculate the visibilities at

        Returns
        -------
        array-like
            The complex visibilities evaluated at the u, v coordinates

        """
        m, n = input_map.shape
        size = m * n
        vis = np.zeros(input_uv.shape[1], dtype=complex)

        x = Visibility.generate_xy(m, 1)
        y = Visibility.generate_xy(n, 1)

        x, y = np.meshgrid(x, y)
        x = x.reshape(size)
        y = y.reshape(size)

        for i in range(size):
            vis[i] = np.sum(
                input_map.reshape(size) * np.exp(
                    -2j * np.pi * (input_uv[0, i] * x + input_uv[1, i] * y)))

        return vis

    @staticmethod
    def idft_map(input_visibilities, output_map, input_uv):
        r"""
        Calculate a map from the given visibilities using a discrete fourier transform

        Parameters
        ----------
        input_visibilities : array-like
            The input visibilities to use

        output_map : array-like
            The u, v coordinate to calculate the visibilities at

        input_uv : array-like
            The corresponding u, v coordiante to the map

        Returns
        -------
        array-like
            The complex visibilities evaluated at the u, v coordinates

        """
        m, n = output_map.shape
        size = m * n

        x = Visibility.generate_xy(m, 1)
        y = Visibility.generate_xy(n, 1)

        x, y = np.meshgrid(x, y)
        x = x.reshape(size)
        y = y.reshape(size)

        im = np.zeros(size)

        for i in range(size):
            im[i] = (1 / input_visibilities.size) * np.sum(
                input_visibilities * np.exp(
                    2j * np.pi * (input_uv[0, :] * x[i] + input_uv[1, :] * y[i])))

        return im.reshape(m, n)
