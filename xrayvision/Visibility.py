"""Visibility relates things.

Visibility.

"""
from datetime import datetime

import numpy as np

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
    def dft_map(input_map, input_uv, center=(0.0, 0.0)):
        """
        Calculate the visibilities for the given map using a discrete fourier transform

        Parameters
        ----------
        input_map : array-like
            Input map to be transformed

        input_uv : array-like
            The u, v coordinate to calculate the visibilities at

        center: array-like
            Position of the center of the transformation. The center
            of the image is (0,0) and the direction of the x axis is ->
            and the direction of the y axis is ^

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
        x = x - center[0]
        y = y + center[1]

        for i in range(size):
            vis[i] = np.sum(
                input_map.reshape(size) * np.exp(
                    -2j * np.pi * (input_uv[0, i] * x + input_uv[1, i] * y)))

        return vis

    @staticmethod
    def idft_map(input_visibilities, output_map, input_uv, center=(0.0, 0.0)):
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

        center: array-like
            Position of the center of the transformation. The center
            of the result image is (0,0) and the direction of the x axis is ->
            and the direction of the y axis is ^

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
        x = x.reshape(size) - center[0]
        y = y.reshape(size) + center[1]

        im = np.zeros(size)

        for i in range(size):
            im[i] = (1 / input_visibilities.size) * np.sum(
                input_visibilities * np.exp(
                    2j * np.pi * (input_uv[0, :] * x[i] + input_uv[1, :] * y[i])))

        return im.reshape(m, n)


class RHESSIVisibility(Visibility):
    """
    A set of RHESSI visibilities.

    Parameters
    ----------
    uv: `numpy.ndarray`
        The u, v coordinates of the visibilities
    vis: `numpy.ndarray`
        The complex visibility
    isc: `int`
        Related to the grid/detector
    harm: `int`
        Harmonic used
    erange: `numpy.ndarray`
        Energy range
    trange: `numpy.ndarray`
        Time range
    totflux: `float`
        Total flux
    sigamp: `float`
        Sigma or error on visibility
    chi2: `float`
        Chi squared from fit
    xyoffset: `np.ndarray`
        Offset from Sun centre
    type_string: `str`
        count, photon, electron
    units: `str`
        If it is in idl format it will be converted
    atten_state: `int`
        State of the attenuator
    count: `float`
        detector counts
    Examples
    --------

    Notes
    -----

    """

    def __init__(self, uv, vis, isc: int=0, harm: int=1,
                 erange: np.array=np.array([0.0, 0.0]),
                 trange: np.array=np.array([datetime.now(), datetime.now()]),
                 totflux: float=0.0, sigamp: float=0.0,
                 chi2: float=0.0,
                 xyoffset: np.array=np.array([0.0, 0.0]),
                 type_string: str="photon",
                 units: str="Photons cm!u-2!n s!u-1!n",
                 atten_state: int=1,
                 count: float=0.0):
        super().__init__(uv, vis)
        self.isc = isc
        self.harm = harm
        self.erange = erange
        self.trange = trange
        self.totflux = totflux
        self.sigamp = sigamp
        self.chi2 = chi2
        self.xyoffset = xyoffset
        self.type_string = type_string
        self.units = RHESSIVisibility.convert_units_to_tex(units)
        self.atten_state = atten_state
        self.count = count

    @staticmethod
    def convert_units_to_tex(string: str):
        """
        String is converted from idl format to tex, if it alredy is there will be
        no conversation

        Parameters
        ----------
        string: str
            The string what should be converted

        Examples
        --------

        Notes
        -----
        """
        final_string = ""
        opened = 0
        check_for_instruction = False
        for i in range(len(string)):
            if check_for_instruction:
                if string[i] == 'n':
                    final_string += opened * "}"
                    opened = 0
                elif string[i] == 'u':
                    final_string += "^{"
                    opened += 1
                elif string[i] == 's':
                    final_string += "_{"
                    opened += 1
                check_for_instruction = False
            elif string[i] == '!':
                check_for_instruction = True
            else:
                final_string += string[i]
        final_string += opened * "}"
        return final_string
