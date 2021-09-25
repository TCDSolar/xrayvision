"""
Modules contains visibility related classes.

This contains classes to hold general visibilities and specialised classes hold visibilities from
certain spacecraft or instruments
"""
from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.table import Table
from sunpy.io.fits import fits
from sunpy.map import Map
from sunpy.time import parse_time

from .transform import dft_map, idft_map

__all__ = ['Visibility', 'RHESSIVisibility']


class Visibility(object):
    r"""
    Hold a set of related visibilities and information.

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

    # TODO should really ensure vis has units to photons cm^-1 s^1 etc
    @u.quantity_input(uv=1/u.arcsec, center=u.arcsec, pixel_size=u.arcsec)
    def __init__(self, uv, vis, xyoffset=(0., 0.) * u.arcsec, pixel_size=(1., 1.) * u.arcsec):
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
        return f"{self.uv.size}, {self.vis}"

    def __eq__(self, other):
        r"""
        Equality for Visibility class

        Parameters
        ----------
        other : `Visibility`
            The other visibility to compare

        Returns
        -------
        `boolean`

        """
        props_equal = []
        for key in self.__dict__.keys():
                props_equal.append(np.array_equal(self.__dict__[key], other.__dict__[key]))

        if all(props_equal):
            return True
        else:
            return False

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
            The new visibility object

        Raises
        ------
        TypeError
            If the fits file is not from a supported instrument

        """
        with fits.open(filename) as hdu_list:
            primary_header = hdu_list[0].header
            if primary_header.get('source') == 'xrayvision':
                return Visibility.from_fits(hdu_list)
            elif primary_header.get('TELESCOP') == 'RHESSI' and \
                    primary_header.get('INSTRUME') == 'RHESSI':
                return RHESSIVisibility.from_fits_old(hdu_list=hdu_list)
            else:
                raise TypeError("This type of fits visibility file is not supported")

    @classmethod
    def from_fits(cls, hdu_list):
        """

        Parameters
        ----------
        hdu_list

        Returns
        -------

        """
        vis_hdu = hdu_list[1]
        spatial_unit = u.Unit(vis_hdu.header.get('unit', 'arcsec'))
        xyoffset = np.unique(vis_hdu.data['xyoffset'], axis=0)
        pixel_size = np.unique(vis_hdu.data['pixel_size'], axis=0)
        return Visibility(vis_hdu.data['uv'].T / spatial_unit, vis_hdu.data['vis'].T,
                          xyoffset.flatten() * spatial_unit, pixel_size.flatten() * spatial_unit)

    @classmethod
    @u.quantity_input(center=u.arcsec, pixel_size=u.arcsec)
    def from_image(cls, image, uv, center=(0.0, 0.0) * u.arcsec, pixel_size=(1.0, 1.0) * u.arcsec):
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
    @u.quantity_input(uv=1 / u.arcsec)
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
        new_pos = np.array([0., 0.])
        if "crval1" in meta:
            new_pos[0] = float(meta["crval1"])
        if "crval2" in meta:
            new_pos[1] = float(meta["crval2"])

        new_psize = np.array([1., 1.])
        if "cdelt1" in meta:
            new_psize[0] = float(meta["cdelt1"])
        if "cdelt2" in meta:
            new_psize[1] = float(meta["cdelt2"])

        return cls.from_image(inmap.data, uv, center=new_pos * u.arcsec,
                              pixel_size=new_psize * u.arcsec)

    @u.quantity_input(center=u.arcsec, pixel_size=u.arcsec)
    def to_image(self, shape, center=[0., 0.]*u.arcsec, pixel_size=None):
        r"""
        Create a image by performing a back projection or inverse transform on the visibilities.

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

        pixel = self.pixel_size
        if pixel_size:
            if pixel_size.ndim == 0:
                pixel = pixel_size.repeat(2)
            elif pixel_size.ndim == 1 and pixel_size.size == 2:
                pixel = pixel_size
            else:
                raise ValueError(
                    f"Pixel_size must be scalar or of length of 2 not {pixel_size.shape}")  # noqa

        return idft_map(self.vis, shape, self.uv, center=center, pixel_size=pixel)

    @u.quantity_input(center=u.arcsec, pixel_size=u.arcsec)
    def to_map(self, shape=(33, 33), center=None, pixel_size=None):

        r"""
        Create a map by performing a back projection or inverse transform on the visibilities.

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
        header = {'crval1': self.xyoffset[0, 0].value if self.xyoffset.ndim == 2
                  else self.xyoffset[0].value,
                  'crval2': self.xyoffset[0, 1].value if self.xyoffset.ndim == 2
                  else self.xyoffset[1].value,
                  'cdelt1': self.pixel_size[0].value,
                  'cdelt2': self.pixel_size[1].value,
                  'ctype1': 'HPLN-TAN',
                  'ctype2': 'HPLT-TAN',
                  'naxis': 2,
                  'naxis1': shape[0],
                  'naxis2': shape[1],
                  'cunit1': 'arcsec', 'cunit2': 'arcsec'}

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

        data = self.to_image(shape, pixel_size=pixel_size)
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
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['source'] = 'xrayvision'
        vis_table = Table([self.uv.value.T, self.vis,
                           np.repeat([self.xyoffset.value], self.vis.shape, axis=0),
                           np.repeat([self.pixel_size.value], self.vis.shape, axis=0)],
                          names=('uv', 'vis', 'xyoffset', 'pixel_size'))

        vis_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(vis_table.as_array()))
        if self.uv.unit.bases == self.xyoffset.unit.bases == self.pixel_size.unit.bases:
            vis_hdu.header.set('unit', str(self.uv.unit.bases[0]))
        else:
            raise ValueError(f'Units must have the same base unit  uv: {self.uv.unit}, xyoffset: '
                             f'{self.xyoffset.unit}, pixel_size: {self.pixel_size.unit}')

        hdul = fits.HDUList([primary_hdu, vis_hdu])
        try:
            hdul.writeto(path)
        except Exception as e:
            raise e


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
    erange : `numpy.ndarray`
        Energy range
    trange : `numpy.ndarray`
        Time range
    totflux : `numpy.ndarray`
        Total flux
    sigamp : `numpy.ndarray`
        Sigma or error on visibility
    chi2 : `numpy.ndarray`
        Chi squared from fit
    xyoffset : `np.ndarray`
        Offset from Sun centre
    type : `str`
        count, photon, electron
    units : `str`
        If it is in idl format it will be converted
    atten_state : `int`
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

    # For single time and energy ranges these data columns should be constant
    CONSTANT_DATA_COLUMNS = ['harm', 'erange', 'trange', 'xyoffset', 'type', 'units',
                             'atten_state', 'norm_ph_factor']
    DYANMIC_DATA_COLUMNS = ['isc', 'u', 'v', 'obsvis', 'totflux', 'sigamp', 'chi2', 'count']

    COLUMN_DEFS = {'ATTEN_STATE': 'I', 'CHI2': 'E', 'COUNT': 'E', 'ERANGE': '2E', 'HARM': 'I',
                   'ISC': 'I', 'NORM_PH_FACTOR': 'E', 'OBSVIS': 'C', 'SIGAMP': 'E', 'TOTFLUX': 'E',
                   'TRANGE': '2D', 'TYPE': '6A', 'U': 'E', 'UNITS': '24A', 'V': 'E',
                   'XYOFFSET': '2E'}

    def __init__(self, uv, vis, isc=None, harm: int = 1,
                 erange: np.array = np.array([0.0, 0.0]),
                 trange: np.array = np.array([datetime.now(), datetime.now()]),
                 totflux=None, sigamp=None, chi2=None,
                 xyoffset: np.array = np.array([0.0, 0.0]),
                 type: str = "photon",
                 units: str = "Photons cm!u-2!n s!u-1!n",
                 atten_state: int = 1, count=None,
                 pixel_size: np.array = np.array([1.0, 1.0])*u.arcsec,
                 norm_ph_factor=0,
                 *, meta):
        r"""
        Initialise a new RHESSI visibility.

        Parameters
        ----------
        uv
        vis
        isc
        harm
        erange
        trange
        totflux
        sigamp
        chi2
        xyoffset
        type
        units
        atten_state
        count
        pixel_size
        norm_ph_factor

        """
        super().__init__(uv=uv, vis=vis, xyoffset=xyoffset, pixel_size=pixel_size)
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
        self.type = type
        self.units = units
        self.atten_state = atten_state
        if count is None:
            self.count = np.zeros(vis.shape)
        else:
            self.count = count
        self.norm_ph_factor = norm_ph_factor
        self.meta = meta

    @staticmethod
    def exists_and_unique(hdu, column, indices):
        """
        Check if the data column exits have the same value for all indices

        Parameters
        ----------
        hdu : `astropy.io.fits.BinTableHDU` header data unit
            HDU to check

        column : `str`
            The data column name

        indices : `list`


        Returns
        -------

        Raises
        ------


        """
        if column.casefold() in [name.casefold() for name in hdu.data.columns.names]:
            column = column.lower()
            if np.all(hdu.data[column][indices] == hdu.data[column][indices[0]]):
                return hdu.data[column][indices[0]]
            else:
                raise ValueError(f"Column: {column} was not constant")
        else:
            raise ValueError(f"Column: {column} does not exist")

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

    @classmethod
    def from_fits(cls, hdu_list):
        """

        Parameters
        ----------
        hdu_list

        Returns
        -------

        """
        for hdu in hdu_list:
            if hdu.name == "VISIBILITY":
                rhessi_columns = cls.COLUMN_DEFS.copy()
                [rhessi_columns.pop(x) for x in ('U', 'V', 'OBSVIS')]
                data = {}

                for prop, _ in rhessi_columns.items():
                    if prop.casefold() in ['xyoffset', 'pixel_size']:
                        data[prop.casefold()] = hdu.data[prop] * u.arcsec
                    else:
                        data[prop.casefold()] = hdu.data[prop]

                data['meta'] = hdu_list[0].header

                return RHESSIVisibility(uv=np.vstack((hdu.data['u']*-1.0,
                                                      hdu.data['v']*-1.0))/u.arcsec,
                                        vis=hdu.data['obsvis'], **data)
        raise ValueError('Fits HDUs did not contain visibility extension')

    @classmethod
    def from_fits_old(cls, hdu_list):
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
        for hdu in hdu_list:
            if hdu.name == "VISIBILITY":
                energy_ranges = hdu.data["erange"]
                unique_energy_ranges = np.unique(energy_ranges, axis=0)
                time_ranges = hdu.data["trange"]
                unique_time_ranges = np.unique(time_ranges, axis=0)

                visibilities = np.zeros((unique_time_ranges.shape[0],
                                         unique_energy_ranges.shape[0]), dtype=object)

                # Creating the RHESSIVisibilities
                for i, time_range in enumerate(unique_time_ranges):
                    for j, energy_range in enumerate(unique_energy_ranges):
                        indices = np.argwhere((time_ranges[:, 0] == time_range[0]) &
                                              (time_ranges[:, 1] == time_range[1]) &
                                              (energy_ranges[:, 0] == energy_range[0]) &
                                              (energy_ranges[:, 1] == energy_range[1])).reshape(-1)

                        static = {name: cls.exists_and_unique(hdu, name, indices)
                                  for name in cls.CONSTANT_DATA_COLUMNS}

                        static['meta'] = hdu_list[0].header
                        static['xyoffset'] = static['xyoffset'] * u.arcsec

                        dynamic = {name: hdu.data[name][indices] for name in
                                   cls.DYANMIC_DATA_COLUMNS if name not in ['u', 'v', 'obsvis']}

                        cur_vis = RHESSIVisibility(uv=np.vstack((hdu.data['u'][indices] * -1.0,
                                                   hdu.data['v'][indices] * -1.0)) / u.arcsec,
                                                   vis=hdu.data['obsvis'][indices],
                                                   **{**static, **dynamic})

                        visibilities[i, j] = cur_vis

                if visibilities.size == 1:
                    return visibilities[0, 0]
                else:
                    # return RHESSIVisibilityList(visibilities)
                    return visibilities

    def to_map(self, shape=(33, 33), center=None, pixel_size=None):
        map = super().to_map(shape=shape, center=center, pixel_size=pixel_size)
        map.meta['wavelnth'] = self.erange
        map.meta['date_obs'] = parse_time(self.trange[0], format='utime').fits
        map.meta['date-obs'] = parse_time(self.trange[0], format='utime').fits
        map.meta['date_end'] = parse_time(self.trange[1], format='utime').fits
        map.meta['timesys'] = 'utc'

        for key, value in self.meta.items():
            if key.casefold() not in map.meta:
                map.meta[key.casefold()] = value

        return map

    def to_fits_file(self, path):
        """
        Write the visibility to a fits file.

        Parameters
        ----------
        path

        Returns
        -------

        """

        # TODO  Bit hacky need a better appraoch if the file orginally came from RHESSI should keep
        # all the orgial hdus for later writing back to fits. If a new file need to figure out what
        # minimal required headers and extensions are.
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header = self.meta

        modifed_colums = ('U', 'V', 'OBSVIS')
        orig_columns = self.COLUMN_DEFS.copy()
        modifed_colum_formats = [orig_columns.pop(x) for x in modifed_colums]

        fits_columns = []
        for name, format in orig_columns.items():
            value = getattr(self, name.casefold())
            if name.casefold() in self.CONSTANT_DATA_COLUMNS:
                value = np.tile(value, (self.vis.size, 1))

            fits_columns.append(fits.Column(name=name, array=value,
                                            format=format))

        fits_columns.append(fits.Column(name='U', array=self.uv[0, :]*-1.0,
                                        format=modifed_colum_formats[0]))

        fits_columns.append(fits.Column(name='V', array=self.uv[1, :]*-1.0,
                                        format=modifed_colum_formats[1]))

        fits_columns.append(fits.Column(name='OBSVIS', array=self.vis,
                                        format=modifed_colum_formats[2]))

        vis_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(fits_columns))

        hdu_list = fits.HDUList([primary_hdu, vis_hdu])
        hdu_list[1].name = 'VISIBILITY'
        hdu_list.writeto(path)
