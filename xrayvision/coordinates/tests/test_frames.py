import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from sunpy.coordinates import HeliographicStonyhurst

from xrayvision.coordinates.frames import Projective, projective_frame_to_wcs, projective_wcs_to_frame


@pytest.fixture
def projective_wcs():
    w = WCS(naxis=2)

    w.wcs.dateobs = "2024-01-01 00:00:00.000"
    w.wcs.crpix = [10, 20]
    w.wcs.cdelt = np.array([2, 2])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["PJLN-TAN", "PJLT-TAN"]

    w.wcs.aux.hgln_obs = 10
    w.wcs.aux.hglt_obs = 20
    w.wcs.aux.dsun_obs = 1.5e11

    return w


@pytest.fixture
def projective_frame():
    obstime = "2024-01-01"
    observer = HeliographicStonyhurst(10 * u.deg, 20 * u.deg, 1.5e11 * u.m, obstime=obstime)

    frame_args = {"obstime": obstime, "observer": observer, "rsun": 695_700_000 * u.m}

    frame = Projective(**frame_args)
    return frame


def test_projective_wcs_to_frame(projective_wcs):
    frame = projective_wcs_to_frame(projective_wcs)
    assert isinstance(frame, Projective)

    assert frame.obstime.isot == "2024-01-01T00:00:00.000"
    assert frame.rsun == 695700 * u.km
    assert frame.observer == HeliographicStonyhurst(
        10 * u.deg, 20 * u.deg, 1.5e11 * u.m, obstime="2024-01-01T00:00:00.000"
    )


def test_projective_wcs_to_frame_none():
    w = WCS(naxis=2)
    w.wcs.ctype = ["ham", "cheese"]
    frame = projective_wcs_to_frame(w)

    assert frame is None


def test_projective_frame_to_wcs(projective_frame):
    wcs = projective_frame_to_wcs(projective_frame)

    assert isinstance(wcs, WCS)
    assert wcs.wcs.ctype[0] == "PJLN-TAN"
    assert wcs.wcs.cunit[0] == "arcsec"
    assert wcs.wcs.dateobs == "2024-01-01 00:00:00.000"

    assert wcs.wcs.aux.rsun_ref == projective_frame.rsun.to_value(u.m)
    assert wcs.wcs.aux.dsun_obs == 1.5e11
    assert wcs.wcs.aux.hgln_obs == 10
    assert wcs.wcs.aux.hglt_obs == 20


def test_projective_frame_to_wcs_none():
    wcs = projective_frame_to_wcs(HeliographicStonyhurst())
    assert wcs is None
