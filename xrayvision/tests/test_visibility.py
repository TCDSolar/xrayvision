from pathlib import Path

import numpy as np
import pytest

import astropy.units as unit
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits

from sunpy.map import Map

from ..transform import generate_uv
from ..visibility import Visibility, RHESSIVisibility


@pytest.fixture
def test_data_dir():
    path = Path(__file__).parent.parent / 'data'
    return path


class TestVisibility(object):

    def test_from_image(self):
        m = n = 33
        size = m * n
        # Set up empty map
        image = np.zeros((m, n))

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv_in = np.array([u, v]).reshape(2, size) / unit.arcsec

        # For an empty map visibilities should all be zero (0+0j)
        empty_vis = Visibility.from_image(image, uv_in)
        assert np.array_equal(empty_vis.pixel_size, (1.0, 1.0) * unit.arcsec)
        assert np.array_equal(empty_vis.xyoffset, (0.0, 0.0) * unit.arcsec)
        assert np.array_equal(empty_vis.vis, np.zeros(n*m, dtype=complex))

    def test_from_image_with_center(self):
        m = n = 33
        size = m * n
        # Set up empty map
        image = np.zeros((m, n))

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv_in = np.array([u, v]).reshape(2, size) / unit.arcsec

        # For an empty map visibilities should all be zero (0+0j)
        empty_vis = Visibility.from_image(image, uv_in, center=(2.0, -3.0) * unit.arcsec)
        assert np.array_equal(empty_vis.pixel_size,  (1.0, 1.0) * unit.arcsec)
        assert np.array_equal(empty_vis.xyoffset,  (2.0, -3.0) * unit.arcsec)
        assert np.array_equal(empty_vis.vis, np.zeros(n * m, dtype=complex))

    def test_from_image_with_pixel_size(self):
        m = n = 33
        size = m * n
        # Set up empty map
        image = np.zeros((m, n))

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv_in = np.array([u, v]).reshape(2, size) / unit.arcsec

        empty_vis = Visibility.from_image(image, uv_in, pixel_size=(2.0, 3.0) * unit.arcsec)
        assert np.array_equal(empty_vis.pixel_size, (2.0, 3.0) * unit.arcsec)
        assert np.array_equal(empty_vis.xyoffset,  (0.0, 0.0) * unit.arcsec)
        assert np.array_equal(empty_vis.vis, np.zeros(n * m, dtype=complex))

    def test_from_image_with_center_and_pixel_size(self):
        m = n = 33
        size = m * n

        cen = (2.0, -3.0) * unit.arcsec
        pix = (2, 3) * unit.arcsec
        # Set up empty map
        image = Gaussian2DKernel(6, x_size=n, y_size=m).array

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m, center=cen[0], pixel_size=pix[0])
        v = generate_uv(n, center=cen[1], pixel_size=pix[1])
        u, v = np.meshgrid(u, v)
        uv_in = np.array([u, v]).reshape(2, size) / unit.arcsec

        vis = Visibility.from_image(image, uv_in, center=(2.0, -3.0) * unit.arcsec,
                                    pixel_size=(2.0, 3.0) * unit.arcsec)
        assert np.array_equal(vis.pixel_size, (2.0, 3.0) * unit.arcsec)
        assert np.array_equal(vis.xyoffset, (2.0, -3.0) * unit.arcsec)
        res = vis.to_image((m, n), center=(2.0, -3.0) * unit.arcsec,
                           pixel_size=(2.0, 3.0) * unit.arcsec)
        assert np.allclose(res, image)

    @pytest.mark.parametrize("pos,pixel", [((0.0, 0.0), (1.0, 1.0)),
                                           ((-12.0, 19.0), (2., 2.)),
                                           ((12.0, -19.0), (1., 5.)),
                                           ((0.0, 0.0), (1.0, 5.0))])
    def test_from_sunpy_map(self, pos, pixel):
        m = n = 33
        size = m * n

        pos = pos * unit.arcsec
        pixel = pixel * unit.arcsec

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m, pos[0])
        v = generate_uv(n, pos[1])
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size) / unit.arcsec

        header = {'crval1': pos[0].value, 'crval2': pos[1].value,
                  'cdelt1': pixel[0].value, 'cdelt2': pixel[1].value,
                  'cunit1': 'arcsec', 'cunit2': 'arcsec'}

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(6, x_size=n, y_size=m).array
        mp = Map((data, header))
        vis = Visibility.from_map(mp, uv)

        assert np.array_equal(vis.pixel_size, pixel)
        assert np.array_equal(vis.xyoffset, pos)

        res = vis.to_image((m, n), center=pos, pixel_size=pixel)
        assert np.allclose(res, data)

    def test_from_fits_file(self, test_data_dir):

        vis = Visibility.from_fits_file(test_data_dir / 'hsi_20020220_110600_1time_1energy.fits')
        assert np.array_equal(vis.pixel_size.value, [1, 1])
        assert np.array_equal(vis.xyoffset.value, np.float32([914.168396, 255.66218567]))
        assert np.array_equal(vis.erange, np.float32([6., 25.]))
        assert np.array_equal(vis.trange,  np.float64([730206360.0, 730206364.0]))

    def test_from_fits_file_invalid(self, tmpdir):
        data = np.zeros((100, 100))
        header = fits.Header()

        p = tmpdir.mkdir('fits').join('test.fits')

        fits.writeto(str(p), data, header=header, overwrite=True)

        with pytest.raises(TypeError):
            Visibility.from_fits_file(str(p))

    def test_to_image(self):
        m = n = 32
        size = m * n

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size) / unit.arcsec

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(6, x_size=n, y_size=m).array

        vis = Visibility.from_image(data, uv)
        res = vis.to_image((m, n))
        assert np.allclose(data, res)
        assert res.shape == (m, n)

    def test_to_image_single_pixel_size(self):
        m = n = 32
        size = m * n

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m, pixel_size=2. * unit.arcsec)
        v = generate_uv(n, pixel_size=2. * unit.arcsec)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size) / unit.arcsec

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(6, x_size=n, y_size=m).array

        vis = Visibility.from_image(data, uv, pixel_size=(2., 2.) * unit.arcsec)
        res = vis.to_image((m, n), pixel_size=2. * unit.arcsec)
        assert res.shape == (m, n)
        assert np.allclose(data, res)

    def test_to_image_invalid_pixel_size(self):
        m = n = 32
        size = m * n

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size) / unit.arcsec

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(6, x_size=n, y_size=m).array

        vis = Visibility.from_image(data, uv)
        with pytest.raises(ValueError):
            vis.to_image((m, n), pixel_size=[1, 2, 2] * unit.arcsec)

    @pytest.mark.parametrize("m,n,pos,pixel", [(33, 33, (10., -5.), (2., 3.)),
                                               (32, 32, (-12, -19), (1., 5.))])
    def test_to_sunpy_map(self, m, n, pos, pixel):
        pos = pos * unit.arcsec
        pixel = pixel * unit.arcsec
        u = generate_uv(m, pixel[0])
        v = generate_uv(n, pixel[1])
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n) / unit.arcsec

        header = {'crval1': pos[0].value, 'crval2': pos[1].value,
                  'cdelt1': pixel[0].value, 'cdelt2': pixel[1].value,
                  'cunit1': 'arcsec', 'cunit2': 'arcsec'}
        data = Gaussian2DKernel(2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)

        res = vis.to_map((m, n), pixel_size=pixel)
        # assert np.allclose(res.data, data)

        assert res.reference_coordinate.Tx == pos[0]
        assert res.reference_coordinate.Ty == pos[1]
        assert res.scale.axis1 == pixel[0] / unit.pix
        assert res.scale.axis2 == pixel[1] / unit.pix
        assert res.dimensions.x == m * unit.pix
        assert res.dimensions.y == n * unit.pix

    def test_to_sunpy_single_pixel_size(self):
        m = n = 32
        u = generate_uv(m, pixel_size=2. * unit.arcsec)
        v = generate_uv(m, pixel_size=2. * unit.arcsec)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n) / unit.arcsec

        header = {'crval1': 0, 'crval2': 0,
                  'cdelt1': 2, 'cdelt2': 2,
                  'cunit1': 'arcsec', 'cunit2': 'arcsec'}
        data = Gaussian2DKernel(2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)
        res = vis.to_map((m, n), pixel_size=2 * unit.arcsec)
        assert res.meta['cdelt1'] == 2.
        assert res.meta['cdelt1'] == 2.
        assert np.allclose(data, res.data)

    def test_to_sunpy_map_invalid_pixel_size(self):
        m = n = 32
        u = generate_uv(m)
        v = generate_uv(m)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n) / unit.arcsec

        header = {'crval1': 0, 'crval2': 0,
                  'cdelt1': 1, 'cdelt2': 1,
                  'cunit1': 'arcsec', 'cunit2': 'arcsec'}
        data = Gaussian2DKernel(2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)

        with pytest.raises(ValueError):
            vis.to_map((m, n), pixel_size=[1, 2, 3] * unit.arcsec)

    def test_to_fits_file(self, tmpdir):
        m = n = 32
        u = generate_uv(m)
        v = generate_uv(m)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n) / unit.arcsec

        header = {'crval1': 0, 'crval2': 0,
                  'cdelt1': 1, 'cdelt2': 1,
                  'cunit1': 'arcsec', 'cunit2': 'arcsec',
                  'ctype1': 'HPLN-TAN', 'ctype2': 'HPLT-TAN'}
        data = Gaussian2DKernel(2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)
        p = tmpdir.join('test.fits')
        vis.to_fits_file(p.strpath)
        assert vis == Visibility.from_fits_file(p.strpath)


class TestRHESSIVisibility(object):

    @pytest.mark.parametrize("N,M,isc,harm,erange,trange,totflux,sigamp,"
                             "chi2,xyoffset,type_string,units,"
                             "atten_state,count",
                             [(65, 65, 1, 3, [5.0, 10.0], [7.0, 19.0],
                               0.7, 0.3, 0.4, [10, 15], "photon", "test",
                               0, 72.5)])
    def test_from_map(self, N, M, isc, harm, erange, trange, totflux, sigamp,
                      chi2, xyoffset, type_string, units, atten_state, count):
        # Calculate uv and vis_in
        u, v = np.meshgrid(np.arange(M), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N*M)/unit.arcsec
        vis_in = np.zeros(N*M, dtype=complex)

        # Creating a RHESSI Visibility
        vis = RHESSIVisibility(uv_in, vis_in, isc, harm, erange, trange,
                               totflux, sigamp, chi2, xyoffset, type_string,
                               units, atten_state, count, meta={})

        assert vis.isc == isc
        assert vis.harm == harm
        assert vis.erange == erange
        assert vis.trange == trange
        assert vis.totflux == totflux
        assert vis.sigamp == sigamp
        assert vis.chi2 == chi2
        assert vis.xyoffset == xyoffset
        assert vis.type == type_string
        assert vis.units == units
        assert vis.atten_state == atten_state
        assert vis.count == count

    @pytest.mark.parametrize("in_str,out_str",
                             [("cm!u-2!n s!u-1!n", "cm^{-2} s^{-1}"),
                              ("m!s4", "m_{4}"),
                              ("m!n", "m"),
                              ("m!u1!u-2!n", "m^{1^{-2}}")])
    def test_unit_string_conversation(self, in_str, out_str):
        assert out_str == RHESSIVisibility.convert_units_to_tex(in_str)

    def test_fits_file_data_read_successful(self, test_data_dir):
        vis = RHESSIVisibility.from_fits_file(
            test_data_dir / "hsi_20020220_110600_1time_1energy.fits")
        assert isinstance(vis, RHESSIVisibility)

        # vis = RHESSIVisibility.from_fits_file(
        #   "xrayvision/data/hsi_20020220_110600_1time_4energies.fits")
        # assert isinstance(vis, RHESSIVisibility)

        # vis = RHESSIVisibility.from_fits_file(
        #   "xrayvision/data/hsi_20020220_110600_9times_1energy.fits")
        # assert isinstance(vis, RHESSIVisibility)

    def test_write_fits_file(self, tmpdir, test_data_dir):
        vis = RHESSIVisibility.from_fits_file(
            test_data_dir / "hsi_20020220_110600_1time_1energy.fits")

        filepath = tmpdir.join('rhessi.fits')
        vis.to_fits_file(filepath.strpath)

        read = RHESSIVisibility.from_fits_file(filepath.strpath)

        assert vis == read
