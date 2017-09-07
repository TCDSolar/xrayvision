"""
import numpy as np
import pytest
from astropy.convolution import Gaussian2DKernel
from scipy.ndimage.interpolation import shift
from scipy import signal

from ..Clean import Hogbom
from ..Visibility import Visibility


class TestClean(object):
    # Typical map sizes even, odd and two point sources
    @pytest.mark.parametrize("N,M,pos1,pos2", [(65, 65, (15, 30), (40, 32)),
                                               (64, 64, (15, 30), (40, 32))])
    def test_hogbom_intensity_change(self, N, M, pos1, pos2):
        # Creating a "clean" map as a base with 2 point sources
        gaussian = Gaussian2DKernel(stddev=3, x_size=N, y_size=M).array
        clean_map = np.add(shift(gaussian, (pos1[0]-int(N/2),
                                            pos1[1]-int(M/2))),
                           shift(gaussian, (pos2[0]-int(N/2),
                                            pos2[1]-int(M/2))))
        psf = np.array([[0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                        [0., 0., 0., 0.5, 0., 0.5, 0., 0., 0.],
                        [0.5, 0.5, 0., 0., 1., 0., 0., 0.5, 0.5],
                        [0., 0., 0., 0.5, 0., 0.5, 0., 0., 0.],
                        [0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.5, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.5, 0., 0., 0., 0.]])
        dirty_map = signal.convolve2d(clean_map, psf)
        print(dirty_map)

        u, v = np.meshgrid(np.arange(M), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N * M)
        vis_in = np.zeros(N * M, dtype=complex)

        vis = Visibility(uv_in, vis_in)
        vis.from_map(dirty_map)
        clean = Hogbom(vis, 1., (N, M))
        while not clean.iterate():
            pass
        final_image = clean.finish(3)
        print(final_image)
        assert np.array_equal(final_image, clean_map)
"""
