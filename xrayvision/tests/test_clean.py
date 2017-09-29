import numpy as np
# import pytest
from scipy import signal
import random
import copy

from ..Clean import Hogbom
from ..Clean import ReasonOfStop
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

        ut = (np.arange(N) - N / 2 + 0.5) * (1 / M)
        vt = -1.0 * (np.arange(M) - M / 2 + 0.5) * (1 / M)
        u, v = np.meshgrid(ut, vt)
        uv_in = np.array([u, v]).reshape(2, N * M)
        vis_in = np.zeros(N * M, dtype=complex)

        dirty_beam = np.zeros((N, M))
        dirty_beam[(N-1)//4:(N-1)//4 + (N-1)//2, (M-1)//2] = 0.75
        dirty_beam[(N-1)//2, (M-1)//4:(M-1)//4 + (M-1)//2, ] = 0.75
        dirty_beam[(N-1)//2, (M-1)//2] = 1.0

        dirty_map = signal.convolve2d(clean_map, dirty_beam, mode='same')

        vis = Visibility(uv_in, vis_in)
        vist = vis.from_map_v2(dirty_map)
        vis.vis = vist

        clean = Hogbom(vis, dirty_beam, 1e-8, (N, M), gain=0.5)
        while clean.iterate() == ReasonOfStop.NOT_FINISHED:
            pass
        final_image = np.add(clean.dirty_map, clean.point_source_map)
        assert np.allclose(final_image, clean_map)

    def test_clean_usecase(self):
        N = M = 64
        pos1 = (15, 30)
        pos2 = (40, 32)
        # Creating a "clean" map as a base with 2 point sources
        clean_map = np.zeros((N, M))
        clean_map[pos1[0], pos1[1]] = 1.
        clean_map[pos2[0], pos2[1]] = 0.8

        ut = (np.arange(N) - N / 2 + 0.5) * (1 / M)
        vt = -1.0 * (np.arange(M) - M / 2 + 0.5) * (1 / M)
        u, v = np.meshgrid(ut, vt)
        uv_in = np.array([u, v]).reshape(2, N * M)
        vis_in = np.zeros(N * M, dtype=complex)

        vis = Visibility(uv_in, vis_in)
        vist = vis.from_map_v2(clean_map)
        vis.vis = vist

        random.seed(0)

        indexes = []

        for i in range(N*M):
            delete_it = random.choice([True, False])
            if delete_it:
                indexes.append(i)

        vis.vis = np.delete(vis.vis, indexes)
        vis.uv = np.delete(vis.uv, indexes, 1)

        dirty_map = np.zeros((N, M))
        dirty_map = vis.to_map_v2(dirty_map)

        save_vis = copy.deepcopy(vis.vis)

        vis.vis = np.zeros(vis.vis.shape)
        input_delta = np.zeros((N, M))
        input_delta[N//2, M//2] = 1.
        vis.from_map_v2(input_delta)

        dirty_beam = np.zeros((N, M))
        dirty_beam = vis.to_map_v2(dirty_beam)

        vis.vis = save_vis
        clean = Hogbom(vis, dirty_beam, 1e-2, (N, M), gain=1.0)
        while clean.iterate() == ReasonOfStop.NOT_FINISHED:
            pass
        final_image = np.add(clean.dirty_map, clean.point_source_map)

        temp = np.argsort(final_image.ravel())[-2:]
        max_loccations = np.dstack(np.unravel_index(temp, final_image.shape))
        assert max_loccations[0][1].tolist() == list(pos1)
        assert max_loccations[0][0].tolist() == list(pos2)

        # Since we already checked for these coordinates, we can use them
        dirty_map[pos1] = 0
        dirty_map[pos2] = 0
        final_image[pos1] = 0
        final_image[pos2] = 0

        # Check if the background was succesfuly filtered or not
        dirty_avg_bgr = np.average(dirty_map)
        final_avg_bgr = np.average(final_image)
        assert final_avg_bgr < dirty_avg_bgr

    def test_clean_stop_reason(self):
        vis = Visibility([[0], [0]], [1])
        clean = Hogbom(vis, np.array([[]]), 0, (1, 1))
        clean.niter = 0
        assert clean.iterate() == ReasonOfStop.REACHED_NITER
        clean = Hogbom(vis, np.array([[]]), 2, (1, 1))
        clean.dirty_map = np.ones((1,1))
        assert clean.iterate() == ReasonOfStop.REACHED_THRESHOLD

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
