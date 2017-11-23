import numpy as np
from scipy import signal

from ..clean import clean, ms_clean, component, radial_prolate_sphereoidal,\
    vec_radial_prolate_sphereoidal


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
    max_locations = np.dstack(np.unravel_index(temp, out_map.shape))
    # Position of max equal those set above
    assert max_locations[0][1].tolist() == pos1
    assert max_locations[0][0].tolist() == pos2


def test_component():
    comp = np.zeros((3, 3))
    comp[1,1] = 1.0

    res = component(scale=0, shape=(3, 3))
    assert np.array_equal(res, comp)

    res = component(scale=1, shape=(3, 3))
    assert np.array_equal(res, comp)

    res = component(scale=2, shape=(6,6))
    assert np.all(res[0, :] == 0.0)
    assert np.all(res[:, 0] == 0.0)
    assert np.all(res[2:4, 2:4] == res.max())

    res = component(scale=3, shape=(7,7))
    assert np.all(res[0, :] == 0.0)
    assert np.all(res[:, 0] == 0.0)
    assert res[3, 3] == 1


def test_radial_prolate_sphereoidal():
    amps = [radial_prolate_sphereoidal(r) for r in [-1.0, 0.0, 0.5, 1.0, 2.0]]
    assert amps[0] == 1.0
    assert amps[1] == 1.0
    assert amps[2] == 0.36106538453111797
    assert amps[3] == 0.0
    assert amps[4] == 0.0


def test_vec_radial_prolate_sphereoidal():
    radii = np.linspace(-0.5, 1.5, 1000)
    amps1 = [radial_prolate_sphereoidal(r) for r in radii ]
    amps2 = vec_radial_prolate_sphereoidal(radii)
    assert np.allclose(amps1, amps2)


def test_ms_clean():
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
    out_map = ms_clean(dirty_map, dirty_beam, scales=[0], clean_beam_width=0.0)

    # Within threshold default threshold
    assert np.allclose(clean_map, out_map, atol=0.01)
    temp = np.argsort(out_map.ravel())[-2:]
    max_locations = np.dstack(np.unravel_index(temp, out_map.shape))
    # Position of max equal those set above
    assert max_locations[0][1].tolist() == pos1
    assert max_locations[0][0].tolist() == pos2
