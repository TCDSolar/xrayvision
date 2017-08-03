from ..Visibillity import Visibility

import numpy as np
import pytest

from astropy.convolution import Gaussian2DKernel


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
        u, v = np.meshgrid(np.arange(M), np.arange(M))
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

    def test_dftmap(self):
        M, N = 64, 64
        data = Gaussian2DKernel(stddev=5, x_size=M, y_size=N).array
        # data = np.zeros((M, N))
        # data[32, 32] = 1.0
        # Fake map

        ut = (np.arange(M) - M / 2 + 0.5) * (1 / M)
        vt = -1.0 * (np.arange(M) - N / 2 + 0.5) * (1 / N)
        u, v = np.meshgrid(ut, vt)
        uv = np.array([u, v]).reshape(2, M * N)

        visout = Visibility.dft_map(data, uv)
        imout = Visibility.idft_map(visout, np.zeros((M, N)), uv)

        assert np.allclose(data, imout)
