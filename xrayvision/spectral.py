r"""
Implementation of spectral component imaging.

References
----------
Stiefel et al. 2025, "Spectral component imaging of solar X-ray flares", A&A, 704, A316
https://doi.org/10.1051/0004-6361/202557373
"""

import copy
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

import astropy.units as apu
from astropy.units import Quantity

from xrayvision.visibility import Visibilities

__all__ = ["vis_spectral_components"]


def vis_spectral_components(
    vis: Sequence[Visibilities],
    fractions: NDArray[np.floating] | Quantity,
    *,
    normalization: NDArray[np.floating] | Quantity | None = None,
) -> list[Visibilities]:
    r"""
    Decompose visibilities observed in multiple energy bins into visibilities of a number
    of spectral components.

    A visibility :math:`V(u, v, E)` measured at angular frequency :math:`(u, v)` and
    energy :math:`E` can be written as the total flux :math:`F(E)` at that energy times a
    relative visibility :math:`\nu(u, v, E) = V(u, v, E) / F(E)` describing the source
    morphology. If ``n_components`` spectral components (e.g. thermal and non-thermal)
    each have their own, energy-independent morphology :math:`\nu_k(u, v)` but
    energy-dependent fractional contribution :math:`f_k(E)` to the total flux (typically
    obtained from an independent spectral fit), then

    .. math::

        \nu(u, v, E) = \sum_{k=1}^{n_{components}} f_k(E)\, \nu_k(u, v).

    Given ``vis``, the fractions :math:`f_k(E)` and, optionally, :math:`F(E)`, this solves,
    independently at every :math:`(u, v)` point, the corresponding weighted linear least
    squares problem for the visibilities, :math:`\nu_k(u, v)`, of each spectral component,
    weighting by the visibility amplitude uncertainties. The component visibilities can
    then be imaged separately using e.g. `~xrayvision.clean.vis_clean` or
    `~xrayvision.mem.mem`.

    This implements the method described in Stiefel et al. 2025 (A&A, 704, A316,
    `doi:10.1051/0004-6361/202557373 <https://doi.org/10.1051/0004-6361/202557373>`__),
    based on Caspi et al. 2015 (ApJ, 811, 8, `doi:10.1088/0004-637X/811/1/8
    <https://doi.org/10.1088/0004-637X/811/1/8>`__).

    Parameters
    ----------
    vis :
        Visibilities, :math:`V(u, v, E)`, for each energy bin/channel, ordered to match
        the rows of `fractions`. All elements must share the same ``u``, ``v``
        coordinates and must provide
        `~xrayvision.visibility.Visibilities.amplitude_uncertainty`.
    fractions :
        Fractional contribution, :math:`f_k(E)`, of each spectral component to the total
        flux in each energy bin, shape ``(len(vis), n_components)``.
    normalization :
        Total flux, :math:`F(E)`, of each energy bin, shape ``(len(vis),)``, used to
        convert ``vis`` to relative visibilities, :math:`\nu(u, v, E)`, before solving.
        Defaults to no normalization, i.e. `vis` are assumed to already be relative
        visibilities.

    Returns
    -------
    :
        List of `~xrayvision.visibility.Visibilities`, one per spectral component in the
        order of the columns of `fractions`, sharing the ``u``, ``v`` coordinates,
        ``phase_center`` and metadata of ``vis[0]``.
    """
    n_e = len(vis)
    if n_e == 0:
        raise ValueError("vis must contain at least one Visibilities object.")

    if isinstance(fractions, apu.Quantity):
        fractions = cast(Quantity, fractions).to_value(apu.dimensionless_unscaled)
    fractions = np.atleast_2d(np.asarray(fractions, dtype=float))
    if fractions.shape[0] != n_e:
        raise ValueError(
            f"fractions must have one row per energy bin, got shape {fractions.shape} for {n_e} energy bins."
        )
    n_c = fractions.shape[1]
    if n_e < n_c:
        raise ValueError(f"At least as many energy bins ({n_e}) as spectral components ({n_c}) are required.")

    ref = vis[0]
    if any(v.amplitude_uncertainty is None for v in vis):
        raise ValueError("All input visibilities must provide `amplitude_uncertainty`.")
    amplitude_uncertainties = [cast(Quantity, v.amplitude_uncertainty) for v in vis]
    if any(
        v.u.shape != ref.u.shape or not apu.quantity.allclose(v.u, ref.u) or not apu.quantity.allclose(v.v, ref.v)
        for v in vis[1:]
    ):
        raise ValueError("All input visibilities must share the same u, v coordinates.")

    vis_unit = ref.visibilities.unit
    vis_array = np.stack([v.visibilities.to_value(vis_unit) for v in vis], axis=0)  # (n_e, n_vis)
    sigma_array = np.stack([sigma.to_value(vis_unit) for sigma in amplitude_uncertainties], axis=0)  # (n_e, n_vis)

    if normalization is not None:
        if isinstance(normalization, apu.Quantity):
            normalization = cast(Quantity, normalization).to_value(vis_unit)
        normalization = np.atleast_1d(np.asarray(normalization, dtype=float))
        if normalization.shape != (n_e,):
            raise ValueError(f"normalization must have shape ({n_e},), got {normalization.shape}.")
        vis_array = vis_array / normalization[:, np.newaxis]
        sigma_array = sigma_array / normalization[:, np.newaxis]

    if np.any(sigma_array <= 0):
        raise ValueError("amplitude_uncertainty must be positive for all visibilities.")

    # Batch the linear algebra over the uv points/baselines so u is the leading axis.
    vis_t = vis_array.T  # (n_vis, n_e)
    sigma_t = sigma_array.T  # (n_vis, n_e)
    weights = 1.0 / sigma_t**2  # (n_vis, n_e)

    # Weighted normal equations, per uv point: gram = fractions.T @ diag(weights) @ fractions.
    gram = np.einsum("ei,ue,ej->uij", fractions, weights, fractions)  # (n_vis, n_c, n_c)
    # rhs = fractions.T @ diag(weights), per uv point.
    rhs = np.einsum("ej,ue->uje", fractions, weights)  # (n_vis, n_c, n_e)
    # coeffs = gram^-1 @ rhs, maps energies -> components, per uv point. Solved directly rather
    # than via an explicit matrix inverse for better numerical stability and performance.
    try:
        coeffs = np.linalg.solve(gram, rhs)  # (n_vis, n_c, n_e)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Could not solve for the spectral component visibilities: the weighted normal "
            "equations are singular at one or more (u, v) points. This typically means "
            "`fractions` is not full rank, e.g. one energy bin's fractional contributions are a "
            "linear combination of the others, or there are fewer independent energy bins than "
            "spectral components."
        ) from err

    vis_comp = np.einsum("uie,ue->ui", coeffs, vis_t)  # (n_vis, n_c)
    sigma_comp = np.sqrt(np.einsum("uie,ue->ui", coeffs**2, sigma_t**2))  # (n_vis, n_c)

    components = []
    for k in range(n_c):
        components.append(
            Visibilities(
                vis_comp[:, k] * vis_unit,
                u=ref.u,
                v=ref.v,
                phase_center=ref.phase_center,
                meta=copy.deepcopy(ref.meta),
                amplitude_uncertainty=sigma_comp[:, k] * vis_unit,
            )
        )
    return components
