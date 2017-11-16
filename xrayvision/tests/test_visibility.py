import numpy as np
import pytest

import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits

from sunpy.map import Map

from ..transform import generate_uv
from ..visibility import Visibility, RHESSIVisibility


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
        uv_in = np.array([u, v]).reshape(2, size)

        # For an empty map visibilities should all be zero (0+0j)
        empty_vis = Visibility.from_image(image, uv_in)
        assert empty_vis.pixel_size == (1.0, 1.0)
        assert empty_vis.xyoffset == (0.0, 0.0)
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
        uv_in = np.array([u, v]).reshape(2, size)

        # For an empty map visibilities should all be zero (0+0j)
        empty_vis = Visibility.from_image(image, uv_in, center=(2.0, -3.0)*u.arcsec)
        assert empty_vis.pixel_size == (1.0, 1.0)
        assert empty_vis.xyoffset == (2.0, -3.0)
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
        uv_in = np.array([u, v]).reshape(2, size)

        empty_vis = Visibility.from_image(image, uv_in, pixel_size=(2.0, 3.0)*u.arcsec)
        assert empty_vis.pixel_size == (2.0, 3.0)
        assert empty_vis.xyoffset == (0.0, 0.0)
        assert np.array_equal(empty_vis.vis, np.zeros(n * m, dtype=complex))

    def test_from_image_with_center_and_pixel_size(self):
        m = n = 33
        size = m * n

        cen = (2.0, -3.0)
        pix = (2, 3)
        # Set up empty map
        image = Gaussian2DKernel(stddev=6, x_size=n, y_size=m).array

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m, center=cen[0], pixel_size=pix[0])
        v = generate_uv(n, center=cen[1], pixel_size=pix[1])
        u, v = np.meshgrid(u, v)
        uv_in = np.array([u, v]).reshape(2, size)

        vis = Visibility.from_image(image, uv_in, center=(2.0, -3.0)*u.arcsec,
                                    pixel_size=(2.0, 3.0)*u.arcsec)
        assert vis.pixel_size == (2.0, 3.0)
        assert vis.xyoffset == (2.0, -3.0)

        res = vis.to_image((m, n), center=(2.0, -3.0), pixel_size=(2.0, 3.0)*u.arcsec)
        assert np.allclose(res, image)

    @pytest.mark.parametrize("pos,pixel", [((0.0, 0.0), (1.0, 1.0)),
                                               ((-12.0, 19.0), (1., 2.)),
                                               ((12.0, -19.0), (1., 5.)),
                                               ((0.0, 0.0), (1.0, 5.0))])
    def test_from_sunpy_map(self, pos, pixel):
        m = n = 33
        size = m * n

        pos = pos * u.arcsec
        pixel = pos * u.arcsec

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size)

        header = {'crval1': pos[0], 'crval2': pos[1],
                  'cdelt1': pixel[0], 'cdelt2': pixel[1]}

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(stddev=6, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)

        assert vis.pixel_size == list(pixel)
        assert vis.xyoffset == list(pos)

        res = vis.to_image((m, n), center=pos, pixel_size=pixel)
        assert np.allclose(res, data)

    def test_from_fits_file(self):
        vis = Visibility.from_fits_file('xrayvision/data/hsi_20020220_110600_1time_1energy.fits')
        assert len(vis) == 1
        assert np.array_equal(vis[0].pixel_size, [1, 1])
        assert np.array_equal(vis[0].xyoffset, np.float32([914.168396, 255.66218567]))
        assert np.array_equal(vis[0].erange, np.float32([6., 25.]))
        assert np.array_equal(vis[0].trange,  np.float64([730206360.0, 730206364.0]))

    def test_from_fits_file_invalid(self, tmpdir):
        data = np.zeros((100, 100))
        header = fits.Header()

        p = tmpdir.mkdir('fits').join('test.fits')

        fits.writeto(str(p), data, header=header, overwrite=True)

        with pytest.raises(TypeError):
            vis = Visibility.from_fits_file(str(p))

    def test_to_image(self):
        m = n = 32
        size = m * n

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size)

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(stddev=6, x_size=n, y_size=m).array

        vis = Visibility.from_image(data, uv)
        res = vis.to_image((m, n))
        assert np.allclose(data, res)
        assert res.shape == (m, n)

    def test_to_image_single_pixel_size(self):
        m = n = 32
        size = m * n

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m, pixel_size=2.*u.arcsec)
        v = generate_uv(n, pixel_size=2.*u.arcsec)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size)

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(stddev=6, x_size=n, y_size=m).array

        vis = Visibility.from_image(data, uv, pixel_size=(2., 2.)*u.arcsec)
        res = vis.to_image((m, n), pixel_size=2.*u.arcsec)
        assert res.shape == (m, n)
        assert np.allclose(data, res)

    def test_to_image_invalid_pixel_size(self):
        m = n = 32
        size = m * n

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u = generate_uv(m)
        v = generate_uv(n)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, size)

        # Astropy index order is opposite to that of numpy, is 1st dim is across second down
        data = Gaussian2DKernel(stddev=6, x_size=n, y_size=m).array

        vis = Visibility.from_image(data, uv)
        with pytest.raises(ValueError):
            res = vis.to_image((m, n), pixel_size=[1,2,2])

    @pytest.mark.parametrize("m,n,pos,pixel", [(33, 33, (10., -5.), (2., 3.)),
                                               (32, 32, (-12, -19), (1., 5.))])
    def test_to_sunpy_map(self, m, n, pos, pixel):
        pos = pos * u.arcsec
        pixel = pixel * u.arcsec
        u = generate_uv(m, pos[0], pixel[0])
        v = generate_uv(m, pos[1], pixel[1])
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n)

        header = {'crval1': pos[0], 'crval2': pos[1],
                  'cdelt1': pixel[0], 'cdelt2': pixel[1]}
        data = Gaussian2DKernel(stddev=2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)

        res = vis.to_map((m, n), center=pos, pixel_size=pixel)
        assert np.allclose(res.data, data)
        assert res.meta['crval1'] == pos[0]
        assert res.meta['crval2'] == pos[1]
        assert res.meta['cdelt1'] == pixel[0]
        assert res.meta['cdelt2'] == pixel[1]
        assert res.meta['naxis1'] == m
        assert res.meta['naxis2'] == n

    def test_to_sunpy_single_pixel_size(self):
        m = n = 32
        u = generate_uv(m, pixel_size=2.*u.arcsec)
        v = generate_uv(m, pixel_size=2.*u.arcsec)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n)

        header = {'crval1': 0, 'crval2': 0,
                  'cdelt1': 2, 'cdelt2': 2}
        data = Gaussian2DKernel(stddev=2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)
        res = vis.to_map((m, n), pixel_size=2,)
        assert res.meta['cdelt1'] == 2.
        assert res.meta['cdelt1'] == 2.
        assert np.allclose(data, res.data)

    def test_to_sunpy_map_invalid_pixel_size(self):
        m = n = 32
        u = generate_uv(m)
        v = generate_uv(m)
        u, v = np.meshgrid(u, v)
        uv = np.array([u, v]).reshape(2, m * n)

        header = {'crval1': 0, 'crval2': 0,
                  'cdelt1': 1, 'cdelt2': 1}
        data = Gaussian2DKernel(stddev=2, x_size=n, y_size=m).array
        mp = Map((data, header))

        vis = Visibility.from_map(mp, uv)

        with pytest.raises(ValueError):
            res = vis.to_map((m, n), pixel_size=[1, 2, 3])


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
        uv_in = np.array([u, v]).reshape(2, N*M)
        vis_in = np.zeros(N*M, dtype=complex)

        # Creating a RHESSI Visibility
        vis = RHESSIVisibility(uv_in, vis_in, isc, harm, erange, trange,
                               totflux, sigamp, chi2, xyoffset, type_string,
                               units, atten_state, count)

        assert vis.isc == isc
        assert vis.harm == harm
        assert vis.erange == erange
        assert vis.trange == trange
        assert vis.totflux == totflux
        assert vis.sigamp == sigamp
        assert vis.chi2 == chi2
        assert vis.xyoffset == xyoffset
        assert vis.type_string == type_string
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

    def test_fits_file_data_read_successful(self):
        i = RHESSIVisibility.from_fits_file(
            "xrayvision/data/hsi_20020220_110600_1time_1energy.fits")
        assert len(i) == 1

        i = RHESSIVisibility.from_fits_file(
          "xrayvision/data/hsi_20020220_110600_1time_4energies.fits")
        assert len(i) == 4

        i = RHESSIVisibility.from_fits_file(
          "xrayvision/data/hsi_20020220_110600_9times_1energy.fits")
        assert len(i) == 9