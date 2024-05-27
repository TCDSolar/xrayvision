import astropy.units as apu
import pytest
from astropy.coordinates import get_body
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from numpy.testing import assert_array_equal

import xrayvision.visibility as vm
from xrayvision.visibility import Visibility


def test_visibility():
    vis = Visibility(vis=1 * apu.ct, u=1 / apu.deg, v=1 / apu.deg)
    assert vis.vis == 1 * apu.ct
    assert vis.u == 1 / apu.deg
    assert vis.v == 1 / apu.deg
    assert_array_equal([0, 0] * apu.arcsec, vis.phase_centre)
    assert_array_equal([0, 0] * apu.arcsec, vis.offset)


@pytest.fixture
def vis_meta():
    return vm.VisMeta(
        {
            "observer_coordinate": get_body("Earth", Time("2000-01-01 00:00:00")),
            "energy_range": [6, 10] * apu.keV,
            "time_range": Time(["2000-01-01 00:00:00", "2000-01-01 00:00:01"]),
            "vis_labels": ["3a", "10b"],
            "instrument": "stix",
        }
    )


@pytest.fixture
def visibilities(vis_meta):
    visibilities = [1 + 2 * 1j, 3 + 3 * 1j] * apu.ct
    u = [0.023, -0.08] * 1 / apu.arcsec
    v = [-0.0013, 0.013] * 1 / apu.arcsec
    phase_center = [0, 0] * apu.arcsec
    unc = [0.01, 0.15] * apu.ct
    return vm.Visibilities(visibilities, u, v, phase_center, uncertainty=unc, meta=vis_meta)


def test_vis_u(visibilities):
    vis_base = visibilities
    output_u = vis_base.u
    expected_u = [0.023, -0.08] * 1 / apu.arcsec

    assert_quantity_allclose(output_u, expected_u)


def test_vis_v(visibilities):
    vis_base = visibilities
    output_v = vis_base.v
    expected_v = [-0.0013, 0.013] * 1 / apu.arcsec

    assert_quantity_allclose(output_v, expected_v)


def test_vis_amplitude(visibilities):
    vis_base = visibilities
    output_amplitude = vis_base.amplitude
    expected_amplitude = [2.236068, 4.2426407] * apu.ct

    assert_quantity_allclose(output_amplitude, expected_amplitude)


def test_vis_phase(visibilities):
    vis_base = visibilities
    output_phase = vis_base.phase
    expected_phase = [63.434949, 45] * apu.deg

    assert_quantity_allclose(output_phase, expected_phase)


def test_vis_amplitude_uncertainty(visibilities):
    vis_base = visibilities
    output_amplitude_uncertainty = vis_base.amplitude_uncertainty
    expected_amplitude_uncertainty = [0.004472136, 0.10606602] * apu.ct

    assert_quantity_allclose(output_amplitude_uncertainty, expected_amplitude_uncertainty)


def test_vis_phase_uncertainty(visibilities):
    vis_base = visibilities
    output_phase_uncertainty = vis_base.phase_uncertainty
    expected_phase_uncertainty = [0.22918312, 1.4323945] * apu.deg

    assert_quantity_allclose(output_phase_uncertainty, expected_phase_uncertainty)


def test_vis_eq(visibilities):
    vis = visibilities
    assert vis == vis


def test_meta_eq(vis_meta):
    meta = vis_meta
    assert meta == meta
    meta = vm.VisMeta(dict())
    assert meta == meta
