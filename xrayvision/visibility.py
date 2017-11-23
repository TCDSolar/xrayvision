"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""
from datetime import datetime

import astropy.units as u
import numpy as np
from sunpy.map import Map
from sunpy.io.fits import fits

from .transform import dft_map, idft_map

__all__ = ['Visibility', 'RHESSIVisibility']


class Visibility(object):
    r"""
    A class to hold a set of visibilities and related information.

    Attributes
    ----------
    uv : `numpy.ndarray`
        Array of 2xN u, v coordinates where visibilities will be evaluated
    vis : `numpy.ndarray`
        Array of N complex visibilities at coordinates in `uv`
    xyoffset : `float` (x, y), optional
        The offset x, y offset of phase center
    pixel_size : `float` (dx, dy), optional
        Pixel size in x and y directions

    Methods
    -------

    Examples
    --------

    Notes
    -----

    """

    def __init__(self, uv, vis, xyoffset=(0., 0.)*u.arcsec, pixel_size=(1., 1.)*u.arcsec):
        r"""
        Initialise a new Visibility object.

        Parameters
        ----------
        uv : `numpy.ndarray`
            Array of 2xN u, v coordinates where visibilities will be evaluated
        vis : `numpy.ndarray`
            The complex visibilities
        xyoffset : `tuple` (x-center, y-center), optional
            The offset x, y offset of phase center

        pixel_size : `tuple` (x-size, y-size)
            Pixel in the given direction (x, y)

        """
        self.uv = uv
        self.vis = np.array(vis, dtype=complex)
        self.xyoffset = xyoffset
        self.pixel_size = pixel_size

    def __repr__(self):
        r"""
        Return a printable representation of the visibility.

        Returns
        -------
        `str`

        """
        return f"{self.uv}, {self.vis}"

    @classmethod
    def from_fits_file(cls, filename):
        r"""
        Create a new visibility object from a fits file.

        Parameters
        ----------
        filename : `basestring`
            The path/filename of the the fits file to read

        Returns
        -------
        `Visibility`
            The new visibilty object

        Raises
        ------
        TypeError
            If the fits file is not from a supported instrument

        """
        with fits.open(filename) as hdus:
            primary_header = hdus[0].header
            if primary_header.get('TELESCOP') == 'RHESSI' and \
                    primary_header.get('INSTRUME') == 'RHESSI':
                return RHESSIVisibility.from_fits(hdu_list=hdus)
            else:
                raise TypeError("Currently only support reading of RHESSI visibility files")

    @classmethod
    @u.quantity_input(center=u.arcsec, pixel_size=u.arcsec)
    def from_image(cls, image, uv, center=(0.0, 0.0)*u.arcsec, pixel_size=(1.0, 1.0)*u.arcsec):
        r"""
        Create a new Visibility object from the given image array.

        Parameters
        ----------
        image : `numpy.ndarray`
            The 2D input image
        uv : `numpy.ndarray`
            Array of 2xN u, v coordinates where the visibilities will be evaluated
        center : `float` (x, y)
            The coordinates of the center of the image
        pixel_size : `float` (dx, dy)
            The pixel size in  x and y directions

        Returns
        -------
        `Visibility`

            The new visibility object

        """
        vis = dft_map(image, uv, center=center, pixel_size=pixel_size)
        return Visibility(uv, vis, center, pixel_size)

    @classmethod
    @u.quantity_input(uv=1/u.arcsec)
    def from_map(cls, inmap, uv):
        r"""
        Create a new Visibility object from the given map.

        Parameters
        ----------
        inmap : `sunpy.map.Map`
            The input map
        uv : `numpy.ndarray`
            Array of 2xN u, v coordinates where the visibilities will be evaluated
        Returns
        -------
        `Visibility`
            The new visibility object

        """
        meta = inmap.meta
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

        return cls.from_image(inmap.data, uv, center=new_pos * u.arcsec,
                              pixel_size=new_psize * u.arcsec)

    @u.quantity_input(center=u.arcsec, pixel_size=u.arcsec)
    def to_image(self, shape, center=None, pixel_size=None):
        r"""
        Create a image by doing a back projection or inverse transform on the visibilities.

        Parameters
        ----------
        shape : `int`
            Shape of the output image to create (m, n)

        center : `float`, (x, y)
            Coordinates of the map center if given will override `self.xyoffset`

        pixel_size : `float` (dx, dy), optional
            Size of the pixels in x, y if only one give assumed same in both directions will \
            override `self.pixel_size`

        Returns
        -------
        `numpy.ndarray`
            Output image

        """
        offset = self.xyoffset
        if center:
            offset = center

        pixel = self.pixel_size
        if pixel_size:
            if pixel_size.ndim == 0:
                pixel = pixel_size.repeat(2)
            elif pixel_size.ndim == 1 and pixel_size.size == 2:
                pixel = pixel_size
            else:
                raise ValueError(f"Pixel_size must be scalar or of length of 2 not {pixel_size.shape}")  # noqa

        return idft_map(self.vis, shape, self.uv, center=offset, pixel_size=pixel)

    @u.quantity_input(center=u.arcsec, pixel_size=u.arcsec)
    def to_map(self, shape=(33, 33), center=None, pixel_size=None):

        r"""
        Create a map from doing a back projection or inverse transform on the visibilities.

        Parameters
        ----------
        shape : `int` (m, n)
            Shape of the output map in pixels
        center : `float` (x, y)
            Coordinates of the map center if given will override `self.xyoffset`
        pixel_size : `float` (dx, dy), optional
            Size of the pixels in x, y if only one give assumed same in both directions

        Returns
        -------
        `sunpy.map.Map`
            Map object with the map created from the visibilities and the meta data will contain the
            offset and the pixel size

        """
        header = {'crval1': self.xyoffset[0].value,
                  'crval2': self.xyoffset[1].value,
                  'cdelt1': self.pixel_size[0].value,
                  'cdelt2': self.pixel_size[1].value}
        if center:
            header['crval1'] = center[0].value
            header['crval2'] = center[1].value

        if pixel_size:
            if pixel_size.ndim == 0:
                header['cdelt1'] = pixel_size.value
                header['cdelt2'] = pixel_size.value
            elif pixel_size.ndim == 1 and pixel_size.size == 2:
                header['cdelt1'] = pixel_size[0].value
                header['cdelt2'] = pixel_size[1].value
            else:
                raise ValueError(f"pixel_size can have a length of 1 or 2 not {pixel_size.shape}")

        data = self.to_image(shape, center=center, pixel_size=pixel_size)
        return Map((data, header))

    def to_fits_file(self, path):
        """
        Write the visibilities to a fits file.

        Parameters
        ----------
        path : 'basestr'
            Path to fits file

        Returns
        -------

        """
        pass


class RHESSIVisibility(Visibility):
    """
    A set of RHESSI visibilities.

    Parameters
    ----------
    uv : `numpy.ndarray`
        The u, v coordinates of the visibilities
    vis : `numpy.ndarray`
        The complex visibility
    isc : `int based array-like`
        Related to the grid/detector
    harm : `int`
        Harmonic used
    energy_range : `numpy.ndarray`
        Energy range
    time_range : `numpy.ndarray`
        Time range
    total_flux : `numpy.ndarray`
        Total flux
    sigamp : `numpy.ndarray`
        Sigma or error on visibility
    chi2 : `numpy.ndarray`
        Chi squared from fit
    xyoffset : `np.ndarray`
        Offset from Sun centre
    type_string : `str`
        count, photon, electron
    units : `str`
        If it is in idl format it will be converted
    attenuator_state : `int`
        State of the attenuator
    count : `numpy.ndarray`
        detector counts
    pixel_size : `array-like`
        size of a pixel in arcseconds

    Examples
    --------

    Notes
    -----

    """

    def __init__(self, uv, vis, isc=None, harm: int=1,
                 energy_range: np.array=np.array([0.0, 0.0]),
                 time_range: np.array=np.array([datetime.now(), datetime.now()]),
                 total_flux=None, sigamp=None, chi2=None,
                 xyoffset: np.array=np.array([0.0, 0.0]),
                 type_string: str="photon",
                 units: str="Photons cm!u-2!n s!u-1!n",
                 attenuator_state: int=1, count=None,
                 pixel_size: np.array=np.array([1.0, 1.0])):
        r"""
        Initialise a new RHESSI visibility.

        Parameters
        ----------
        uv
        vis
        isc
        harm
        energy_range
        time_range
        total_flux
        sigamp
        chi2
        xyoffset
        type_string
        units
        attenuator_state
        count
        pixel_size

        """
        super().__init__(uv, vis, xyoffset, pixel_size)
        if isc is None:
            self.isc = np.zeros(vis.shape)
        else:
            self.isc = isc
        self.harm = harm
        self.erange = energy_range
        self.trange = time_range
        if total_flux is None:
            self.totflux = np.zeros(vis.shape)
        else:
            self.totflux = total_flux
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
        self.atten_state = attenuator_state
        if count is None:
            self.count = np.zeros(vis.shape)
        else:
            self.count = count

    @staticmethod
    def convert_units_to_tex(string: str):
        """
        Convert from idl format to latex, if it already is there will be no conversation.

        Parameters
        ----------
        string : `str`
            The IDL format string to be converted

        Returns
        -------
        `str`
            The LATEX equivalent of the IDL format string

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

    @classmethod  # noqa
    def from_fits(cls, hdu_list):
        """
        Create RHESSIVisibility from compatible fits hdus.

        Parameters
        ----------
        hdu_list : `list`
            List of RHESSI visibility hdus

        Returns
        -------
        `list`
            A list of `RHESSIVisibilty`

        Examples
        --------

        Notes
        -----
        It separates the Visibility data based on the time and energy
        ranges.

        """
        for i in hdu_list:
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
                        if np.allclose(j, t, rtol=1e-15):
                            return i

                for j, k in enumerate(erange_unique):
                    data_sort[j] = {}

                for j, k in enumerate(trange):
                    eind = find_erange(erange[j])
                    tind = find_trange(k)
                    if tind not in data_sort[eind]:
                        data_sort[eind][tind] = [j]
                    else:
                        data_sort[eind][tind].append(j)

                # Creating the RHESSIVisibilities
                visibilities = []
                for j, k in data_sort.items():
                    for l, m in k.items():
                        visibilities.append(RHESSIVisibility(np.array([]),
                                                             np.array([[], []]),
                                                             energy_range=erange_unique[j],
                                                             time_range=trange_unique[l]))
                        uu = np.take(i.data["u"], m)
                        vv = np.take(i.data["v"], m)
                        visibilities[-1].uv = np.array([uu, vv])
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
                            string = RHESSIVisibility.convert_units_to_tex(i.data["units"][m[0]])
                            visibilities[-1].units = string
                        if "ATTEN_STATE" in i.header.values():
                            visibilities[-1].atten_state = i.data["atten_state"][m[0]]
                        if "COUNT" in i.header.values():
                            visibilities[-1].count = np.take(i.data["count"], m)
                return visibilities

    def to_fits_file(self, path):
        """
        Write the visibility to a fits file.

        Parameters
        ----------
        path

        Returns
        -------

        """
        pass
