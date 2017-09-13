"""
import numpy as np
import pytest
from scipy import signal
# import matplotlib.pyplot as plt

from ..Clean import Hogbom
from ..Visibility import Visibility


class TestClean(object):
    # Typical map sizes even, odd and two point sources
    @pytest.mark.parametrize("N,M,pos1,pos2", [(65, 65, (15, 30), (40, 32)),
                                               (64, 64, (15, 30), (40, 32))])
    def test_hogbom_intensity_change(self, N, M, pos1, pos2):
        # Creating a "clean" map as a base with 2 point sources
        clean_map = np.zeros((N, M), dtype=complex)
        clean_map[pos1[0], pos1[1]] = 10.
        clean_map[pos2[0], pos2[1]] = 7.
        psf = np.array([[0., 0., 0., 0., 10.0, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 10.0, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 10.0, 0., 0., 0., 0.],
                        [0., 0., 0., 10.0, 0., 10.0, 0., 0., 0.],
                        [10.0, 10.0, 0., 0., 20., 0., 0., 10.0, 10.0],
                        [0., 0., 0., 10.0, 0., 10.0, 0., 0., 0.],
                        [0., 0., 0., 0., 10.0, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 10.0, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 10.0, 0., 0., 0., 0.]])
        # plt.imshow(clean_map)
        # plt.show()
        dirty_map = signal.convolve2d(clean_map, psf, mode="same")
        # plt.imshow(dirty_map)
        # plt.show()

        u, v = np.meshgrid(np.arange(M), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N * M)
        vis_in = np.zeros(N * M, dtype=complex)

        vis = Visibility(uv_in, vis_in)
        vis.vis = vis.from_map(dirty_map)
        clean = Hogbom(vis, 1., (N, M))
        while not clean.iterate():
            pass
        final_image = clean.finish(3)
        assert np.array_equal(final_image, clean_map)
"""
