import numpy as np
import pytest

from ..Clean import Hogbom
from ..Visibility import Visibility


class TestClean(object):
    # Typical map sizes even, odd
    @pytest.mark.parametrize("", [()])
    def test_hogbom_intensity_change(self):
        # Create a visibility bag
        u = [5, 7, 15, 37, 45, 57, 60]
        v = [1, 3, 7, 27, 37, 1, 57]
        uv_in = np.array([u, v]).reshape(2, len(u))
        vis_in = [5+7j, 1+2j, 10-5j, -6+1j, 7j, -7-7j, 5j]

        # Test sanity check
        assert len(u) == len(v) and len(vis_in) == len(v)

        vis = Visibility(uv_in, vis_in)

        clean = Hogbom(vis, 0.01)
        max_abs = np.max(np.abs(vis_in))
        clean.iterate()
        max_abs_after_clean = np.max(np.abs(clean.vis.vis))
        # Clean should "lower the ceiling" with every iteration
        assert max_abs_after_clean < max_abs
