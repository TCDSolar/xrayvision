import numpy as np
from scipy import signal

from ..clean import clean


def test_clean():
    n = m = 65
    pos1 = [15, 30]
    pos2 = [40, 32]

    clean_map = np.zeros((n, m))
    clean_map[pos1[0], pos1[1]] = 10.
    clean_map[pos2[0], pos2[1]] = 7.

    dirty_beam = np.zeros((n, m))
    dirty_beam[(n-1)//4:(n-1)//4 + (n-1)//2, (m-1)//2] = 0.75
    dirty_beam[(n-1)//2, (m-1)//4:(m-1)//4 + (m-1)//2, ] = 0.75
    dirty_beam[(n-1)//2, (m-1)//2] = 1.0

    dirty_map = signal.convolve2d(clean_map, dirty_beam, mode='same')

    # Disable convolution of model with gaussian for testing
    out_map = clean(dirty_map, dirty_beam, clean_beam_width=0.0)

    # Within threshold default threshold
    assert np.allclose(clean_map, out_map, atol=0.01)
    temp = np.argsort(out_map.ravel())[-2:]
    max_loccations = np.dstack(np.unravel_index(temp, out_map.shape))
    # Position of max equal those set above
    assert max_loccations[0][1].tolist() == pos1
    assert max_loccations[0][0].tolist() == pos2
