import numpy as np
# import pytest
from scipy import signal

from ..Clean import Hogbom
from ..Visibility import Visibility


class TestClean(object):
    # Typical map sizes even, odd and two point sources
    # @pytest.mark.parametrize("N,M,pos1,pos2", [(65, 65, (15, 30), (40, 32)),
    #                                            (64, 64, (15, 30), (40, 32))])
    # def test_hogbom_intensity_change(self, N, M, pos1, pos2):
    def test_hogbom_intensity_change(self):
        N = M = 65
        pos1 = [15, 30]
        pos2 = [40, 32]
        # Creating a "clean" map as a base with 2 point sources
        clean_map = np.zeros((N, M))
        clean_map[pos1[0], pos1[1]] = 1.
        clean_map[pos2[0], pos2[1]] = 0.8

        u, v = np.meshgrid(np.arange(N), np.arange(M))
        uv_in = np.array([u, v]).reshape(2, N * M)
        vis_in = np.zeros(N * M, dtype=complex)

        dirty_beam = np.zeros((N, M))
        dirty_beam[(N-1)//4:(N-1)//4 + (N-1)//2, (M-1)//2] = 0.75
        dirty_beam[(N-1)//2, (M-1)//4:(M-1)//4 + (M-1)//2, ] = 0.75
        dirty_beam[(N-1)//2, (M-1)//2] = 1.0

        dirty_map = signal.convolve2d(clean_map, dirty_beam, mode='same')

        vis = Visibility(uv_in, vis_in)
        vist = vis.from_map(dirty_map)
        vis.vis = vist

        clean = Hogbom(vis, dirty_beam, 1e-8, (N, M), gain=0.5)
        while not clean.iterate():
            pass
        final_image = np.add(clean.dirty_map, clean.point_source_map)
        assert np.allclose(final_image, clean_map)

    def test_clean2(self):
        N = M = 65
        pos1 = [15, 30]
        pos2 = [40, 32]

        clean_map = np.zeros((N, M))
        clean_map[pos1[0], pos1[1]] = 10.
        clean_map[pos2[0], pos2[1]] = 7.

        dirty_beam = np.zeros((N, M))
        dirty_beam[(N-1)//4:(N-1)//4 + (N-1)//2, (M-1)//2] = 0.75
        dirty_beam[(N-1)//2, (M-1)//4:(M-1)//4 + (M-1)//2, ] = 0.75
        dirty_beam[(N-1)//2, (M-1)//2] = 1.0

        dirty_map = signal.convolve2d(clean_map, dirty_beam, mode='same')

        # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(clean_map, label='Orig')
        # ax2.imshow(dirty_beam, label='Dirty Beam')
        # ax3.imshow(dirty_map, label='Dirty Map')
        # plt.show()

        out_map = Hogbom.clean(dirty_map, dirty_beam)

        # Within threshold set
        assert np.allclose(clean_map, out_map, atol=2*0.01)
        temp = np.argsort(out_map.ravel())[-2:]
        max_loccations = np.dstack(np.unravel_index(temp, out_map.shape))
        assert max_loccations[0][1].tolist() == pos1
        assert max_loccations[0][0].tolist() == pos2
