import numpy as np

import astropy.units as u

__all__ = ["rhessi_like_uv_coverage", "stix_like_uv_coverage"]


def rhessi_like_uv_coverage():
    """
    Create a RHESSI-like u, v coverage distribution.

    Samples 9 circles, one per subcollimator, with radii given by the inverse of the approximate RHESSI grid
    resolutions (2.26 to 183.2 arcsec) and 32 evenly spaced rotation angles on each to mimic spacecraft rotation.

    Returns
    -------
    `dict`
        A dictionary with the following items, each an array of shape ``(9, 32)`` where the first axis is the
        subcollimator and the second the rotation angle:

        - ``"u"`` : `~astropy.units.Quantity`
            The u coordinates in 1/arcsec.
        - ``"v"`` : `~astropy.units.Quantity`
            The v coordinates in 1/arcsec.
        - ``"isc"`` : `~numpy.ndarray`
            The index of the subcollimator (``isc``), 0 to 8 corresponding to RHESSI detectors 1 to 9.
    """
    # Approximate RHESSI spatial frequencies (arcsec^-1) for subcollimators 1-9,
    # based on typical RHESSI grid pitches of 2.26, 3.92, 6.79, 11.76, 20.36, 35.27, 61.08, 105.8, 183.2 arcsec.
    isc = np.arange(0, 9)  # index subcollimator (isc) detectors 1-9
    resolutions = [2.26, 3.92, 6.79, 11.76, 20.36, 35.27, 61.08, 105.8, 183.2] * u.arcsec
    radii = 1 / resolutions

    # 32 evenly-spaced rotation angles per detector (simulates spacecraft rotation)
    n_angles = 32
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    u_comp = np.outer(radii, np.cos(angles))
    v_comp = np.outer(radii, np.sin(angles))

    isc = isc[:, None] * np.ones(u_comp.shape)

    return {"u": u_comp, "v": v_comp, "isc": isc}


def stix_like_uv_coverage():
    """
    Create a STIX-like u, v coverage distribution.

    Samples 10 circles, one per subcollimator, with radii given by the inverse of the STIX resolutions (7.1 to 178.6
    arcsec). Each subcollimator has three, a, b and c, at different orientations in the range 0 to 180 degrees.

    Returns
    -------
    `dict`
        A dictionary with the following items, each an array of shape ``(3, 10)`` where the first axis is the
        sub-collimator (a, b, c) and the second the subcollimator number 1 to 10:

        - ``"u"`` : `~astropy.units.Quantity`
            The u coordinates in 1/arcsec.
        - ``"v"`` : `~astropy.units.Quantity`
            The v coordinates in 1/arcsec.
        - ``"label"`` : `~numpy.ndarray`
            The subcollimator labels e.g. ``"1a"``, ``"1b"``, ``"1c"``, ``"2a"``.
    """
    resolutions = [7.1, 10.2, 14.6, 20.9, 29.8, 42.7, 61.0, 87.3, 124.9, 178.6] * u.arcsec
    radii = 1 / resolutions

    # Base orientations for sub-collimators a, b, and c
    base_angles = np.array([150, 90, 30])
    steps = np.array([-20, -20, -20])
    offsets = np.cumsum(np.repeat(steps.reshape(3, 1), 9, axis=1), axis=1)
    zeros = np.zeros((3, 1))
    offsets = np.hstack([zeros, offsets])
    angles_cts = base_angles[:, None] + offsets
    angles = np.mod(angles_cts, 180) * u.deg

    labels = np.vstack([f"{i}a,{i}b,{i}c".split(",") for i in range(1, 11)]).T

    u_comp = radii * np.cos(angles)
    v_comp = radii * np.sin(angles)

    return {"u": u_comp, "v": v_comp, "label": labels}
