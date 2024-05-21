from pathlib import Path

import astropy.units as apu
import pytest
from astropy.tests.helper import assert_quantity_allclose

import xrayvision.visibility as vm


@pytest.fixture
def test_data_dir():
    path = Path(__file__).parent.parent / "data"
    return path


@pytest.fixture
def visibilities_base():
    visibilities = [1 + 2 * 1j, 3 + 3 * 1j] * apu.ct
    u = [0.023, -0.08] * 1 / apu.arcsec
    v = [-0.0013, 0.013] * 1 / apu.arcsec
    phase_center = [0, 0] * apu.arcsec
    unc = [0.01, 0.15] * apu.ct
    names = ["3a", "10b"]
    return vm.VisibilitiesBase(visibilities, u, v, phase_center, names=names, uncertainty=unc)


def test_vis_u(visibilities_base):
    vis_base = visibilities_base
    output_u = vis_base.u
    expected_u = [0.023, -0.08] * 1 / apu.arcsec

    assert_quantity_allclose(output_u, expected_u)


def test_vis_v(visibilities_base):
    vis_base = visibilities_base
    output_v = vis_base.v
    expected_v = [-0.0013, 0.013] * 1 / apu.arcsec

    assert_quantity_allclose(output_v, expected_v)


def test_vis_amplitude(visibilities_base):
    vis_base = visibilities_base
    output_amplitude = vis_base.amplitude
    expected_amplitude = [2.236068, 4.2426407] * apu.ct

    assert_quantity_allclose(output_amplitude, expected_amplitude)


def test_vis_phase(visibilities_base):
    vis_base = visibilities_base
    output_phase = vis_base.phase
    expected_phase = [63.434949, 45] * apu.deg

    assert_quantity_allclose(output_phase, expected_phase)


def test_vis_amplitude_uncertainty(visibilities_base):
    vis_base = visibilities_base
    output_amplitude_uncertainty = vis_base.amplitude_uncertainty
    expected_amplitude_uncertainty = [0.004472136, 0.10606602] * apu.ct

    assert_quantity_allclose(output_amplitude_uncertainty, expected_amplitude_uncertainty)


def test_vis_phase_uncertainty(visibilities_base):
    vis_base = visibilities_base
    output_phase_uncertainty = vis_base.phase_uncertainty
    expected_phase_uncertainty = [0.22918312, 1.4323945] * apu.deg

    assert_quantity_allclose(output_phase_uncertainty, expected_phase_uncertainty)


def test_vis_phase_uncertainty(visibilities_base):
    vis_base = visibilities_base
    output_phase_uncertainty = vis_base.phase_uncertainty
    expected_phase_uncertainty = [0.22918312, 1.4323945] * apu.deg

    assert_quantity_allclose(output_phase_uncertainty, expected_phase_uncertainty)
