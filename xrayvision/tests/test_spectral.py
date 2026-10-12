import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as apu

from xrayvision.spectral import vis_spectral_components
from xrayvision.visibility import Visibilities

VUNIT = apu.ph / (apu.cm**2 * apu.s)


def _make_vis(visibilities, sigma, u, v):
    return Visibilities(
        visibilities * VUNIT,
        u=u,
        v=v,
        amplitude_uncertainty=sigma * VUNIT,
    )


@pytest.fixture
def uv():
    rng = np.random.default_rng(1234)
    n_vis = 6
    u = rng.uniform(0.01, 0.5, n_vis) / apu.arcsec
    v = rng.uniform(0.01, 0.5, n_vis) / apu.arcsec
    return u, v


@pytest.fixture
def two_components(uv):
    rng = np.random.default_rng(42)
    u, v = uv
    n_vis = u.shape[0]
    n_c = 2
    n_e = 5

    vis_comp_true = rng.uniform(1, 10, (n_c, n_vis)) + 1j * rng.uniform(-5, 5, (n_c, n_vis))
    fractions = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    sigma = np.full(n_vis, 0.01)

    vis_per_energy = [_make_vis(fractions[e] @ vis_comp_true, sigma, u, v) for e in range(n_e)]
    return vis_per_energy, fractions, vis_comp_true, u, v


def test_vis_spectral_components_recovers_known_components(two_components):
    vis_per_energy, fractions, vis_comp_true, u, v = two_components

    components = vis_spectral_components(vis_per_energy, fractions)

    assert len(components) == fractions.shape[1]
    for k, comp in enumerate(components):
        assert_allclose(comp.visibilities.to_value(VUNIT), vis_comp_true[k], atol=1e-6)
        assert apu.quantity.allclose(comp.u, u)
        assert apu.quantity.allclose(comp.v, v)
        assert comp.amplitude_uncertainty is not None


def test_vis_spectral_components_normalization(two_components):
    vis_per_energy, fractions, vis_comp_true, u, v = two_components

    n_e = len(vis_per_energy)
    normalization = np.linspace(1, 5, n_e)
    scaled_vis = [
        _make_vis(
            v_e.visibilities.to_value(VUNIT) * normalization[e],
            v_e.amplitude_uncertainty.to_value(VUNIT) * normalization[e],
            u,
            v,
        )
        for e, v_e in enumerate(vis_per_energy)
    ]

    components = vis_spectral_components(scaled_vis, fractions, normalization=normalization * VUNIT)

    for k, comp in enumerate(components):
        assert_allclose(comp.visibilities.to_value(VUNIT), vis_comp_true[k], atol=1e-6)


def test_vis_spectral_components_requires_amplitude_uncertainty(two_components):
    vis_per_energy, fractions, _, u, v = two_components
    no_uncertainty = [Visibilities(v_e.visibilities, u=u, v=v) for v_e in vis_per_energy]

    with pytest.raises(ValueError, match="amplitude_uncertainty"):
        vis_spectral_components(no_uncertainty, fractions)


def test_vis_spectral_components_requires_matching_uv(two_components):
    vis_per_energy, fractions, _, u, v = two_components
    mismatched = list(vis_per_energy)
    mismatched[-1] = _make_vis(
        mismatched[-1].visibilities.to_value(VUNIT),
        mismatched[-1].amplitude_uncertainty.to_value(VUNIT),
        u * 2,
        v,
    )

    with pytest.raises(ValueError, match="u, v coordinates"):
        vis_spectral_components(mismatched, fractions)


def test_vis_spectral_components_fractions_shape_mismatch(two_components):
    vis_per_energy, fractions, _, _, _ = two_components

    with pytest.raises(ValueError, match="one row per energy bin"):
        vis_spectral_components(vis_per_energy, fractions[:-1])


def test_vis_spectral_components_too_many_components(two_components):
    vis_per_energy, fractions, _, _, _ = two_components

    with pytest.raises(ValueError, match="At least as many energy bins"):
        vis_spectral_components(vis_per_energy[:1], fractions[:1])


def test_vis_spectral_components_singular_fractions(two_components):
    vis_per_energy, fractions, _, _, _ = two_components
    degenerate_fractions = np.repeat(fractions[:, [0]], 2, axis=1)

    with pytest.raises(ValueError, match="singular"):
        vis_spectral_components(vis_per_energy, degenerate_fractions)
