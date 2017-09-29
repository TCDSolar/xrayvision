"""Visibility relates things.

Visibility.

"""
from datetime import datetime

import numpy as np
from sunpy.map import Map
from sunpy.io.fits import fits

from .Transform import Transform


class Visibility(object):
    """
    A set of visibilities.

    Parameters
    ----------
    uv: `numpy.ndarray` The u, v coordinates of the visibilities
    vis: `numpy.ndarray` The complex visibility
    xyoffset: array-like
        The center of the image is (0,0) and the direction of
        the x axis is -> and the direction of the y axis is ^.
        You have to give the displacement based on this.
    pixel_size: array-like
        What is the size of one pixel in the given direction (x, y)
    Examples
    --------

    Notes
    -----

    """

    def __init__(self, uv, vis, xyoffset=None, pixel_size=None):
        self.uv = np.array(uv)
        self.vis = np.array(vis, dtype=complex)
        self.xyoffset = xyoffset
        self.pixel_size = pixel_size

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

    def from_map_v2(self, inmap, center=None, pixel_size=None):
        """
        Parameters
        ----------
        inmap: np.array
            The input image
        center: array-like
            Where should be the origin in the Fourier-transformation
        pixel_size: array-like
            What is the size of one pixel in the given direction (x, y)
        Returns
        -------

        """
        if center is None:
            if self.xyoffset is None:
                center = (0., 0.)
            else:
                center = self.xyoffset
        else:
            self.xyoffset = center
        if pixel_size is None:
            if self.pixel_size is None:
                pixel_size = (1., 1.)
            else:
                pixel_size = self.pixel_size
        else:
            self.pixel_size = pixel_size
        self.vis = Visibility.dft_map(inmap, self.uv, center, pixel_size)
        return self.vis

    def from_sunpy_map(self, sunpy_map):
        """
        Parameters
        ----------
        sunpy_map: sunpy.map
            The data will be converted from the Sunpy Map and
            the center and pixel meta information will be taken
            into account.
        Returns
        -------

        """
        meta = sunpy_map.meta
        new_pos = [0., 0.]
        if "crval1" in meta:
            new_pos[0] = float(meta["crval1"])
        if "crval2" in meta:
            new_pos[1] = float(meta["crval2"])

        new_psize = [1., 1.]
        if "cdelt1" in meta:
            new_psize[0] = float(meta["cdelt1"])
        if "cdelt2" in meta:
            new_psize[1] = float(meta["cdelt2"])
        return self.from_map_v2(sunpy_map.data, new_pos, new_psize)

    def to_map(self, outmap):
        """
        Parameters
        ----------
        outmap

        Returns
        -------

        """
        return Transform.idft(outmap, self.uv, self.vis)

    def to_map_v2(self, outmap, center=None, pixel_size=None):
        """
        Parameters
        ----------
        outmap: np.array
            An empty array with the same dimensions as
            the output image should have
        center: array-like
            Where should be the origin in the Fourier-transformation
        pixel_size: array-like
            What is the size of one pixel in the given direction (x, y)

        Returns
        -------

        """
        if center is None:
            if self.xyoffset is None:
                center = (0., 0.)
            else:
                center = self.xyoffset
        if pixel_size is None:
            if self.pixel_size is None:
                pixel_size = (1., 1.)
            else:
                pixel_size = self.pixel_size

        return Visibility.idft_map(self.vis, outmap, self.uv, center, pixel_size)

    def to_sunpy_map(self, size=(33,33)):
        """

        Parameters
        ----------
        size: array-like
            (m, n) dimension of the output map

        Returns
        -------
        A Sunpy Map object with the map created from the visibilities
        and the meta data will contain the offset and the pixel's size

        """
        header = {'crval1': self.xyoffset[0],
                  'crval2': self.xyoffset[1],
                  'cdelt1': self.pixel_size[0],
                  'cdelt2': self.pixel_size[1]}
        data = np.zeros(size)
        data = self.to_map_v2(data)
        return Map((data, header))

    @staticmethod
    def generate_xy(number_pixels, center=0., pixel_size=1.):
        """
        Generate the x or y image/map coordinates given the number of pixels and pixel size

        Parameters
        ----------
        number_pixels : int
            Number of pixels in the map

        center: float
            The center of the image is (0,0) and the direction of
            the x axis is -> and the direction of the y axis is ^.
            You have to give the displacement based on this.

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

        x -= center
        return x

    @staticmethod
    def dft_map(input_map, input_uv, center=(0.0, 0.0), pixel_size=(1.0, 1.0)):
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

        pixel_size: array-like
            The size of a pixel in (x_size, y_size) format

        Returns
        -------
        array-like
            The complex visibilities evaluated at the u, v coordinates

        """
        m, n = input_map.shape
        size = m * n
        vis = np.zeros(input_uv.shape[1], dtype=complex)

        x = Visibility.generate_xy(m, center[0], pixel_size[0])
        y = Visibility.generate_xy(n, center[1], pixel_size[1])

        x, y = np.meshgrid(x, y)
        x = x.reshape(size)
        y = y.reshape(size)

        for i in range(input_uv.shape[1]):
            vis[i] = np.sum(
                input_map.reshape(size) * np.exp(
                    -2j * np.pi * (input_uv[0, i] * x + input_uv[1, i] * y)))

        return vis

    @staticmethod
    def idft_map(input_visibilities, output_map, input_uv,
                 center=(0.0, 0.0), pixel_size=(1.0, 1.0)):
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

        pixel_size: array-like
            The size of a pixel in (x_size, y_size) format

        Returns
        -------
        array-like
            The complex visibilities evaluated at the u, v coordinates

        """
        m, n = output_map.shape
        size = m * n

        x = Visibility.generate_xy(m, center[0], pixel_size[0])
        y = Visibility.generate_xy(n, center[1], pixel_size[1])

        x, y = np.meshgrid(x, y)
        x = x.reshape(size)
        y = y.reshape(size)

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
    isc: `int based array-like`
        Related to the grid/detector
    harm: `int`
        Harmonic used
    erange: `numpy.ndarray`
        Energy range
    trange: `numpy.ndarray`
        Time range
    totflux: `numpy.ndarray`
        Total flux
    sigamp: `numpy.ndarray`
        Sigma or error on visibility
    chi2: `numpy.ndarray`
        Chi squared from fit
    xyoffset: `np.ndarray`
        Offset from Sun centre
    type_string: `str`
        count, photon, electron
    units: `str`
        If it is in idl format it will be converted
    atten_state: `int`
        State of the attenuator
    count: `numpy.ndarray`
        detector counts
    pixel_size: `array-like`
        size of a pixel in arcseconds
    Examples
    --------

    Notes
    -----

    """

    def __init__(self, uv, vis, isc=None, harm: int=1,
                 erange: np.array=np.array([0.0, 0.0]),
                 trange: np.array=np.array([datetime.now(), datetime.now()]),
                 totflux=None, sigamp=None, chi2=None,
                 xyoffset: np.array=np.array([0.0, 0.0]),
                 type_string: str="photon",
                 units: str="Photons cm!u-2!n s!u-1!n",
                 atten_state: int=1, count=None,
                 pixel_size: np.array=np.array([1.0, 1.0])):
        super().__init__(uv, vis, xyoffset, pixel_size)
        if isc is None:
            self.isc = np.zeros(vis.shape)
        else:
            self.isc = isc
        self.harm = harm
        self.erange = erange
        self.trange = trange
        if totflux is None:
            self.totflux = np.zeros(vis.shape)
        else:
            self.totflux = totflux
        if sigamp is None:
            self.sigamp = np.zeros(vis.shape)
        else:
            self.sigamp = sigamp
        if chi2 is None:
            self.chi2 = np.zeros(vis.shape)
        else:
            self.chi2 = chi2
        self.type_string = type_string
        self.units = RHESSIVisibility.convert_units_to_tex(units)
        self.atten_state = atten_state
        if count is None:
            self.count = np.zeros(vis.shape)
        else:
            self.count = count

    @staticmethod
    def convert_units_to_tex(string: str):
        """
        String is converted from idl format to tex, if it already is,
        there will be no conversation

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

    @staticmethod
    def from_fits_file(path):
        """
        Creates RHESSIVisibility objects from compatible fits files

        Parameters
        ----------
        path: str
            Path where the fits file can be found

        Examples
        --------

        Notes
        -----
        It separates the Visibility data based on the time and energy
        ranges.
        """
        hudlist = fits.open(path)
        for i in hudlist:
            if i.name == "VISIBILITY":
                # Checking how many data structures we have
                data_sort = {}
                erange = i.data["erange"]
                erange_unique = np.unique(erange, axis=0)
                trange = i.data["trange"]
                trange_unique = np.unique(trange, axis=0)

                def find_erange(e):
                    for i, j in enumerate(erange_unique):
                        if np.allclose(j, e):
                            return i

                def find_trange(t):
                    for i, j in enumerate(trange_unique):
                        if np.allclose(j, t):
                            return i

                for j, k in enumerate(erange_unique):
                        data_sort[j] = {}
                for j, k in enumerate(trange):
                        eind = find_erange(erange[j])
                        tind = find_trange(k)
                        if not tind in data_sort[eind]:
                            data_sort[eind][tind] = [j]
                        else:
                            data_sort[eind][tind].append(j)

                # Creating the RHESSIVisibilities
                visibilities = []
                for j, k in data_sort.items():
                    for l, m in k.items():
                        visibilities.append(RHESSIVisibility(np.array([]), np.array([[], []]), erange=erange_unique[j], trange=trange_unique[l]))
                        u = np.take(i.data["u"], m)
                        v = np.take(i.data["v"], m)
                        visibilities[-1].uv = np.array([u, v])
                        if "XYOFFSET" in i.header.values():
                            visibilities[-1].xyoffset = i.data["xyoffset"][m[0]]
                        if "ISC" in i.header.values():
                            visibilities[-1].isc = np.take(i.data["isc"], m)
                        if "HARM" in i.header.values():
                            visibilities[-1].harm = i.data["harm"][m[0]]
                        if "OBSVIS" in i.header.values():
                            visibilities[-1].vis = np.take(i.data["obsvis"], m)
                        if "TOTFLUX" in i.header.values():
                            visibilities[-1].totflux = np.take(i.data["totflux"], m)
                        if "SIGAMP" in i.header.values():
                            visibilities[-1].sigamp = np.take(i.data["sigamp"], m)
                        if "CHI2" in i.header.values():
                            visibilities[-1].chi2 = np.take(i.data["chi2"], m)
                        if "TYPE" in i.header.values():
                            visibilities[-1].type_string = i.data["type"][m[0]]
                        if "UNITS" in i.header.values():
                            visibilities[-1].units = RHESSIVisibility.convert_units_to_tex(i.data["units"][m[0]])
                        if "ATTEN_STATE" in i.header.values():
                            visibilities[-1].atten_state = i.data["atten_state"][m[0]]
                        if "COUNT" in i.header.values():
                            visibilities[-1].count = np.take(i.data["count"], m)
                return visibilities
        return None
