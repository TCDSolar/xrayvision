import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from scipy.ndimage.interpolation import shift
from sunpy.map import Map

from ..Visibility import Visibility


class TestVisibility(object):

    # Typical map sizes even, odd
    @pytest.mark.parametrize("N,M", [(65, 65), [64, 64]])
    def test_from_map(self, N, M):
        # Set up empty map
        empty_map = np.zeros((N, M))

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u, v = np.meshgrid(np.arange(M), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N*M)
        vis_in = np.zeros(N*M, dtype=complex)

        vis = Visibility(uv_in, vis_in)

        # For an empty map visibilities should all be zero (0+0j)
        empty_vis = vis.from_map(empty_map)
        assert np.array_equal(empty_vis, np.zeros(N*M, dtype=complex))

        # Create a map with Gaussian and calculate FFT for comparision
        gaussian_map = Gaussian2DKernel(stddev=5, x_size=N, y_size=M)
        fft = np.fft.fft2(gaussian_map.array)

        # Our DFT should match inbuilt FFT
        gaussian_vis = vis.from_map(gaussian_map.array)
        assert np.allclose(fft.reshape(N*M), gaussian_vis)

    @pytest.mark.parametrize("N,M", [(65, 65), [64, 64]])
    def test_to_map(self, N, M):
        empty_map = np.zeros((N, M))

        # Calculate full u, v coverage so will be equivalent to a discrete Fourier transform (DFT)
        u, v = np.meshgrid(np.arange(N), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N * M)
        vis_in = np.zeros(N * M, dtype=complex)

        vis = Visibility(uv_in, vis_in)

        # For zero (0+0j) visibilities map should be zeros 0
        empty_bp = vis.to_map(empty_map)
        assert np.array_equal(empty_map, empty_bp)

        # Create a map with Gaussian and calculate FFT for comparision
        gaussian_map = Gaussian2DKernel(stddev=5, x_size=N, y_size=M)
        vis.from_map(gaussian_map.array)

        # The back projection should match exactly
        gaussian_bp = vis.to_map(empty_map)
        assert np.allclose(gaussian_map.array, gaussian_bp)

    @pytest.mark.parametrize("size, displace",
                             [(1.0, -7.0), (1.5, 8.5), (2.0, 9.5)])
    def test_generate_xy(self, size, displace):
        n_even = 64
        n_odd = 65

        even = np.arange(-32, 32) * size
        res_even = Visibility.generate_xy(n_even, pixel_size=size)
        assert np.array_equal(res_even, even)

        odd = np.arange(-32, 33) * size
        res_odd = Visibility.generate_xy(n_odd, pixel_size=size)
        assert np.array_equal(res_odd, odd)

        res_even_displaced = Visibility.generate_xy(n_even, displace, size)
        assert np.array_equal(even - displace, res_even_displaced)

        res_odd_displaced = Visibility.generate_xy(n_odd, displace, size)
        assert np.array_equal(odd - displace, res_odd_displaced)

    @pytest.mark.parametrize("xs,ys", [(65, 65), [64, 64]])
    def test_dftmap(self, xs, ys):
        m, n = xs, ys
        data = Gaussian2DKernel(stddev=5, x_size=m, y_size=n).array
        # data = np.zeros((M, N))
        # data[32, 32] = 1.0
        # Fake map

        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)

        visout = Visibility.dft_map(data, uv)
        imout = Visibility.idft_map(visout, np.zeros((m, n)), uv)

        assert np.allclose(data, imout)

    @pytest.mark.parametrize("m,n,pos", [(65, 65, (17, 10)),
                                         [64, 64, (12, 19)]])
    def test_dftmap_decentered(self, m, n, pos):
        data = Gaussian2DKernel(stddev=2, x_size=m, y_size=n).array
        data2 = shift(data, (pos[1], pos[0]))

        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)
        dft_data = Visibility.dft_map(data2, uv, center=pos)
        idft_data = Visibility.idft_map(dft_data, np.zeros((m, n)), uv)
        assert np.allclose(data, idft_data)

        dft_data2 = Visibility.dft_map(data, uv)
        idft_data2 = Visibility.idft_map(dft_data2, np.zeros((m, n)), uv,
                                         pos)
        assert np.allclose(idft_data2, data2)

    @pytest.mark.parametrize("m,n,pos,pixel",
                             [(65, 65, (17, 10), (2, 1)),
                              [64, 64, (12, 19), (1, 3)]])
    def test_dftmap_decentered_with_pixel(self, m, n, pos, pixel):
        data = Gaussian2DKernel(stddev=2, x_size=m, y_size=n).array
        data2 = shift(data, (pos[1], pos[0]))

        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)
        resized_pos = np.multiply(pos, pixel)
        dft_data = Visibility.dft_map(data2, uv, center=resized_pos,
                                      pixel_size=pixel)
        idft_data = Visibility.idft_map(dft_data, np.zeros((m, n)),
                                        uv, pixel_size=pixel)
        assert np.allclose(data, idft_data)

        dft_data2 = Visibility.dft_map(data, uv, pixel_size=pixel)
        idft_data2 = Visibility.idft_map(dft_data2, np.zeros((m, n)), uv,
                                         resized_pos, pixel_size=pixel)
        assert np.allclose(idft_data2, data2)

    @pytest.mark.parametrize("m,n", [(65, 65)])
    def test_v2_functions(self, m, n):
        data = Gaussian2DKernel(stddev=2, x_size=m, y_size=n).array
        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)
        vis = Visibility(uv, np.zeros(uv.shape[1], dtype=complex))
        vis.from_map_v2(data)
        res = vis.to_map_v2(np.zeros((m, n)))
        assert np.allclose(res, data)

    @pytest.mark.parametrize("m,n,pos,pixel", [(65, 65, (10., -5.), (2., 3.)),
                                               (64, 64, (-12, -19), (1., 5.))])
    def test_v2_functions_with_extra_params(self, m, n, pos, pixel):
        data = Gaussian2DKernel(stddev=2, x_size=m, y_size=n).array
        data2 = shift(data, (pos[1], pos[0]))
        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)
        vis = Visibility(uv, np.zeros(uv.shape[1], dtype=complex), xyoffset=pos)
        vis.from_map_v2(data2)
        res = vis.to_map_v2(np.zeros((m, n)))
        assert np.allclose(res, data2)
        res = vis.to_map_v2(np.zeros((m, n)), center=(0., 0.))
        assert np.allclose(res, data)

    @pytest.mark.parametrize("m,n,pos,pixel", [(65, 65, (10., -5.), (2., 3.)),
                                               (64, 64, (-12, -19), (1., 5.))])
    def test_from_sunpy_map(self, m, n, pos, pixel):
        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)

        header = {'crval1': pos[0], 'crval2': pos[1],
                  'cdelt1': pixel[0], 'cdelt2': pixel[1]}
        data = Gaussian2DKernel(stddev=2, x_size=m, y_size=n).array
        mp = Map((data, header))

        vis = Visibility(uv, np.zeros(uv.shape[1], dtype=complex))
        vis.from_sunpy_map(mp)

        res = vis.to_map_v2(np.zeros((m, n)))
        assert np.allclose(res, data)

    @pytest.mark.parametrize("m,n,pos,pixel", [(65, 65, (10., -5.), (2., 3.)),
                                               (64, 64, (-12, -19), (1., 5.))])
    def test_to_sunpy_map(self, m, n, pos, pixel):
        ut = (np.arange(m) - m / 2 + 0.5) * (1 / m)
        vt = -1.0 * (np.arange(n) - n / 2 + 0.5) * (1 / n)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, m * n)

        header = {'crval1': pos[0], 'crval2': pos[1],
                  'cdelt1': pixel[0], 'cdelt2': pixel[1]}
        data = Gaussian2DKernel(stddev=2, x_size=m, y_size=n).array
        mp = Map((data, header))

        vis = Visibility(uv, np.zeros(uv.shape[1], dtype=complex))
        vis.from_sunpy_map(mp)

        res = vis.to_sunpy_map((m, n))
        assert np.allclose(res.data, data)
        assert res.meta['crval1'] == pos[0]
        assert res.meta['crval2'] == pos[1]
        assert res.meta['cdelt1'] == pixel[0]
        assert res.meta['cdelt2'] == pixel[1]
