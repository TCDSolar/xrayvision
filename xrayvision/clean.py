"""
CLEAN algorithms.

The CLEAN algorithm solves the deconvolution problem by assuming a model for the true sky intensity
which is a collection of point sources or in the case of multiscale clean a collection of
appropriate component shapes at different scales.

"""

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.ndimage import shift

import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.units import Quantity

from sunpy.map.map_factory import Map
from sunpy.util.decorators import add_common_docstring

from xrayvision.imaging import generate_header, get_weights, validate_and_expand_kwarg, vis_psf_image, vis_to_map
from xrayvision.transform import idft_map
from xrayvision.utils import get_logger
from xrayvision.visibility import Visibilities

__all__ = ["clean", "ms_clean", "vis_clean", "vis_ms_clean"]


logger = get_logger(__name__, "DEBUG")


def _clean_parameters() -> dict[str, str]:
    """
    Return formatting fragments to use with `~sunpy.util.decorators.add_common_docstring` to
    populate the parts of the docstrings shared between the various clean functions.

    Every fragment is substituted via ``str.format``, not appended, so that the whole assembled
    docstring goes through a single `inspect.cleandoc` pass and ends up consistently indented
    (mixing `add_common_docstring`'s ``append`` and ``**kwargs`` on the same function leaves the
    appended text at its original source indentation while the rest gets dedented, which breaks
    RST parsing).
    """
    return {
        "scales": (
            "scales : array-like, optional\n"
            "    The scales to use, in ascending order, eg ``[1, 2, 4, 8]``. For grouped input these are\n"
            "    the basis function FWHM in pixels, one per group; by default each is measured from\n"
            "    the central peak of its group's dirty beam."
        ),
        "scale_biases": (
            "scale_biases : array-like, optional\n"
            "    Weights that favour small scales when searching for the next component, one per scale. They\n"
            "    only affect the search, never the flux of the component that is found. By default\n"
            "    ``1 - 0.6 * scale / max(scale)``, or for grouped input the ratio of the smallest scale's\n"
            "    FWHM to each scale's FWHM."
        ),
        "basis": (
            "basis :\n"
            "    Shape of the grouped components, ``'gaussian'`` or ``'paraboloid'``. Only used for grouped input."
        ),
        "groups": (
            "groups : sequence of array-like, optional\n"
            "    Indices into ``vis`` of the visibilities making up each group, eg the visibilities sampled on\n"
            "    one or more neighbouring circles of the u-v plane. Each group is cleaned at the scale of\n"
            "    its own dirty beam. Groups must not overlap, visibilities in no group are not used."
        ),
        "scheme": "scheme :\n    Weighting scheme, natural by default",
        "clean_beam_width": (
            "clean_beam_width :\n"
            "    The width of the gaussian to convolve the model with. If set to 0.0 \\\n"
            "    the gaussian to convolution is disabled"
        ),
        "gain": "gain :\n    The gain per loop or loop gain",
        "thres": "thres :\n    Terminates clean when ``residual.max() <= thres``",
        "niter": "niter :\n    Maximum number of iterations to perform",
        "convolve_residual": (
            "convolve_residual :\n"
            "    If `True`, convolve the residual with the normalised clean beam before adding\n"
            "    it to the restored image."
        ),
        "returns_clean": "Returns\n-------\n:\n    The CLEAN image 2D",
        "returns_ms_clean": "Returns\n-------\n:\n    Cleaned image",
        "notes_clean": (
            r"""Notes
-----
The CLEAN algorithm can be summarised in pesudo code as follows:

.. math::
   & \textrm{CLEAN} \left (I^{D}(l, m),\ B(l,m),\ \gamma,\ f_{Thresh},\ N \right ) \\
   & I^{Res} = I^{D},\ M = \{\},\ i=0 \\
   & \textbf{while} \ i \lt N \ \textbf{do:} \\
   & \qquad l_{max}, m_{max} = \underset{l,m}{\operatorname{argmax}} \left| I^{Res}(l,m) \right| \\
   & \qquad f_{max} = I^{Res}(l_{max}, m_{max}) \\
   & \qquad \textbf{if} \ f_{max} \lt 0 \ \textbf{break} \\
   & \qquad I^{Res} = I^{Res} - \alpha \cdot f_{max} \cdot \operatorname{shift} \left
   ( B(l,m), l_{max}, m_{max} \right ) \\
   & \qquad M = M + \{ l_{max}, m_{max}: \alpha \cdot f_{max} \} \\
   & \qquad i = i + 1 \\
   & \qquad \textbf{if} \ \operatorname{max} \left| I^{Res} \right| \le f_{Thresh} \ \textbf{break} \\
   & \textbf{done} \\
   & \textbf{return}\  M,\ I^{Res}"""
        ),
        "notes_ms_clean": (
            r"""Notes
-----
This is an implementation of the multiscale clean algorithm as outlined in [R1]_ adapted for \
x-ray Fourier observations.

It is based on the implementation in the CASA software which can be found here_ ~L956.

.. _here: https://github.com/casacore/casacore/blob/f4dc1c36287c766796ce3375cebdfc8af797a388/lattices/LatticeMath/LatticeCleaner.tcc

When the visibilities are split into groups (eg by the circles they sample in the u-v plane) the
dirty map and dirty beam become sums over groups, and the algorithm of [R2]_ is used instead: the
group dirty maps are cleaned in parallel, each with a basis function whose width is the FWHM of
its own dirty beam. The scale with the largest bias weighted peak (relative to the peak of its
initial dirty map) is chosen, the flux found at that scale's own peak is added to the model as one
basis function and its response is subtracted from every group's residual. The bias weighted peak
is only used to choose the scale and location, the flux added to the model is unbiased. The
residual left over is divided by the integral of the dirty beam and added to the model as an
estimate of the background."""
        ),
        # Only attached to `ms_clean` itself, not `vis_ms_clean` too: both docstrings can't
        # define the same `[R1]` citation without Sphinx raising a duplicate-citation warning,
        "references_ms_clean": (
            r"""References
----------
.. [R1] Cornwell, T. J., "Multiscale CLEAN Deconvolution of Radio Synthesis Images", IEEE Journal of Selected Topics in Signal Processing, vol 2, p793-801, Paper_ #noqa

.. _Paper: https://ieeexplore.ieee.org/document/4703304/

.. [R2] Volpara, A., Catalano, M., Piana, M., Massone, A. M., "Multi-scale CLEAN for Fourier-based hard x-ray solar imaging", Inverse Problems, vol 40, 125017, 2024, https://doi.org/10.1088/1361-6420/ad91db"""
        ),
    }


@add_common_docstring(**_clean_parameters())  # type: ignore[untyped-decorator]
@u.quantity_input  # type: ignore[untyped-decorator]
def clean(
    dirty_map: Quantity,
    dirty_beam: Quantity,
    pixel_size: Quantity[u.arcsec / u.pix] = None,
    clean_beam_width: Quantity[u.arcsec] | None = 4.0 * u.arcsec,
    gain: float | None = 0.1,
    thres: float | None = None,
    niter: int = 5000,
    convolve_residual: bool = False,
) -> Quantity | NDArray[np.float64]:
    r"""
    Clean the image using Hogbom's original method.

    CLEAN iteratively subtracts the PSF or dirty beam from the dirty map to create the residual.
    At each iteration, the location of the maximum residual is found and a shifted dirty beam is
    subtracted at that location. This process continues until either `niter` iterations are reached or
    the maximum residual <= `thres`.

    Parameters
    ----------
    dirty_map :
        The dirty map to be cleaned 2D
    dirty_beam :
        The dirty beam or point spread function (PSF) 2D must
    pixel_size :
        The pixel size in arcsec
    {clean_beam_width}
    {gain}
    {thres}
    {niter}
    {convolve_residual}

    {returns_clean}

    {notes_clean}
    """
    # Peak pixel of the (always odd-shaped, oversized) dirty beam
    beam_peak = (dirty_beam.shape[0] - 1) // 2, (dirty_beam.shape[1] - 1) // 2

    # Model for sources
    model = np.zeros(dirty_map.shape)
    for i in range(niter):
        # Find the largest deviation by absolute value, not just the largest positive value, so
        # a negative sidelobe bowl can be recognised as the residual's dominant feature.
        mx, my = np.unravel_index(np.abs(dirty_map).argmax(), dirty_map.shape)
        imax = dirty_map[mx, my]

        if imax < 0:
            logger.info("Largest residual is negative")
            break

        model[mx, my] += gain * imax

        if i % 25 == 0:
            logger.info(f"Iter: {i}, strength: {imax}, location: {mx, my}")

        # Window into the oversized dirty beam such that its peak lands exactly at (mx, my)
        xr = slice(beam_peak[0] - mx, beam_peak[0] - mx + dirty_map.shape[0])
        yr = slice(beam_peak[1] - my, beam_peak[1] - my + dirty_map.shape[1])

        shifted = dirty_beam[xr, yr]

        comp = imax * gain * shifted

        dirty_map = np.subtract(dirty_map, comp)

        if thres is not None:
            if np.abs(dirty_map).max() <= thres:
                logger.info("Threshold reached")
                break

    else:
        logger.info("Max iterations reached")

    if clean_beam_width is not None:
        # Convert from FWHM to StDev   FWHM = sigma*(8ln2)**0.5 = 2.3548200450309493
        x_stdev = (clean_beam_width / pixel_size[1]).to_value(u.pix) / 2.3548200450309493
        y_stdev = (clean_beam_width / pixel_size[0]).to_value(u.pix) / 2.3548200450309493
        clean_beam = Gaussian2DKernel(x_stdev, y_stdev, x_size=dirty_beam.shape[1], y_size=dirty_beam.shape[0]).array
        # Normalise beam
        clean_beam = clean_beam / clean_beam.max()

        # Convolve clean beam with model and scale
        clean_map = signal.convolve2d(model, clean_beam / clean_beam.sum(), mode="same") / (
            pixel_size[0].value * pixel_size[1].value
        )

        # Scale residual map with model and scale
        dirty_map = dirty_map / clean_beam.sum() / (pixel_size[0].value * pixel_size[1].value)
        if convolve_residual:
            dirty_map = signal.convolve2d(dirty_map, clean_beam / clean_beam.sum(), mode="same")
        return clean_map + dirty_map, model, dirty_map

    return model + dirty_map, model, dirty_map


@add_common_docstring(**_clean_parameters())  # type: ignore[untyped-decorator]
@u.quantity_input  # type: ignore[untyped-decorator]
def vis_clean(
    vis: Visibilities,
    shape: Quantity[u.pix],
    pixel_size: Quantity[u.arcsec / u.pix],
    clean_beam_width: Quantity[u.arcsec] | None = 4.0,
    niter: int = 5000,
    map: bool | None = True,  # noqa: A002
    gain: float | None = 0.1,
    convolve_residual: bool = False,
    **kwargs: Any,
) -> Any:
    r"""
    Clean the visibilities using Hogbom's original method.

    A wrapper around lower level `clean` which calculates the dirty map and psf

    Parameters
    ----------
    vis :
        The visibilities to clean
    shape :
        Size of map
    pixel_size :
        The pixel size in arcsec
    map :
        Return a `sunpy.map.Map` by default or array only if `False`
    {clean_beam_width}
    {gain}
    {niter}
    {convolve_residual}

    {returns_clean}

    {notes_clean}
    """
    dirty_map = vis_to_map(vis, shape=shape, pixel_size=pixel_size, **kwargs)
    dirty_beam_shape = [x.value * 3 + 1 if x.value * 3 % 2 == 0 else x.value * 3 for x in shape] * shape.unit
    dirty_beam = vis_psf_image(vis, shape=dirty_beam_shape, pixel_size=pixel_size, **kwargs)
    clean_map, model, residual = clean(
        dirty_map.data,
        dirty_beam.value,
        pixel_size=pixel_size,
        clean_beam_width=clean_beam_width,
        gain=gain,
        niter=niter,
        convolve_residual=convolve_residual,
    )
    if not map:
        return clean_map, model, residual

    return [Map((data, dirty_map.meta)) for data in (clean_map, model, residual)]


@add_common_docstring(**_clean_parameters())  # type: ignore[untyped-decorator]
@u.quantity_input  # type: ignore[untyped-decorator]
def ms_clean(
    dirty_map: Quantity,
    dirty_beam: Quantity,
    pixel_size: Quantity[u.arcsec / u.pix],
    scales: Sequence[float] | None = None,
    clean_beam_width: Quantity = 4.0 * u.arcsec,
    gain: float = 0.1,
    thres: float = 0.01,
    niter: int = 5000,
    convolve_residual: bool = False,
    scale_biases: Sequence[float] | None = None,
    basis: Literal["gaussian", "paraboloid"] = "gaussian",
) -> Quantity | NDArray[np.float64]:
    r"""
    Clean the map using a multiscale clean algorithm.

    Parameters
    ----------
    dirty_map :
        The 2D dirty map to be cleaned, or a stack of per group dirty maps with shape ``(groups, ny, nx)``
        which sum to the full dirty map
    dirty_beam :
        The 2D dirty beam should have the same dimensions as `dirty_map`, for grouped input a stack
        of per group dirty beams with shape ``(groups, Ny, Nx)`` which sum to the full dirty beam
    pixel_size :
        The pixel size in arcsec
    {scales}
    {clean_beam_width}
    {gain}
    {thres}
    {niter}
    {convolve_residual}
    {scale_biases}
    {basis}

    {returns_ms_clean}

    {notes_ms_clean}

    {references_ms_clean}
    """
    if np.ndim(dirty_map) == 3:
        return _ms_clean_groups(
            dirty_map,
            dirty_beam,
            pixel_size,
            scales=scales,
            scale_biases=scale_biases,
            basis=basis,
            clean_beam_width=clean_beam_width,
            gain=gain,
            thres=thres,
            niter=niter,
            convolve_residual=convolve_residual,
        )

    # Compute the number of dyadic scales, their sizes and scale biases
    number_of_scales: int = np.floor(np.log2(min(dirty_map.shape))).astype(int)
    scale_sizes: NDArray[np.int_] = 2 ** np.arange(number_of_scales)

    if scales:
        if list(scales) != sorted(scales):
            raise ValueError(f"scales must be in ascending order, got {list(scales)}")
        number_of_scales = len(scales)
        scale_sizes = np.array(scales)

    scale_sizes = np.where(scale_sizes == 0, 1, scale_sizes)

    biases = 1 - 0.6 * scale_sizes / scale_sizes.max()
    if scale_biases is not None:
        biases = np.asarray(scale_biases, dtype=float)
        if biases.shape != scale_sizes.shape:
            raise ValueError(f"scale_biases must have one entry per scale ({len(scale_sizes)}), got {biases.size}")

    model = np.zeros(dirty_map.shape)
    reference_residual: float | None = None

    # Integer center pixel of dirty_map, matching _component()'s
    # (and scipy.signal.convolve's `mode="same"` centering for an even-length kernel, which is
    # `(L - 1) // 2`, not `L // 2`), used as the reference for placing scale_kernels via shift()
    map_peak = (dirty_map.shape[0] - 1) // 2, (dirty_map.shape[1] - 1) // 2
    # Peak pixel of the (always odd-shaped, oversized) dirty beam / cross terms -- the one
    # well-defined integer reference pixel, regardless of whether dirty_map's even or odd size.
    beam_peak = (dirty_beam.shape[0] - 1) // 2, (dirty_beam.shape[1] - 1) // 2

    # Pre-compute scales, residual maps and dirty beams at each scale and dirty beam cross terms
    scale_kernels: NDArray[np.float64] = np.zeros((dirty_map.shape[0], dirty_map.shape[1], number_of_scales))
    scaled_residuals: NDArray[np.float64] = np.zeros((dirty_map.shape[0], dirty_map.shape[1], number_of_scales))
    scaled_dirty_beams: NDArray[np.float64] = np.zeros((dirty_beam.shape[0], dirty_beam.shape[1], number_of_scales))
    max_scaled_dirty_beams: NDArray[np.float64] = np.zeros(number_of_scales)
    cross_terms: dict[tuple[int, int], NDArray[np.float64]] = {}

    for i, scale in enumerate(scale_sizes):
        kernel = _component(scale=scale, shape=dirty_map.shape)
        scale_kernels[:, :, i] = kernel / kernel.sum()
        scaled_residuals[:, :, i] = signal.convolve(dirty_map, scale_kernels[:, :, i], mode="same")
        scaled_dirty_beams[:, :, i] = signal.convolve(dirty_beam, scale_kernels[:, :, i], mode="same")
        max_scaled_dirty_beams[i] = scaled_dirty_beams[:, :, i].max()

    # The peaks normalise each scale's response, so a non-positive one would flip signs or blow
    # up that scale's strengths and make it win the scale selection every iteration.
    if np.any(max_scaled_dirty_beams <= 0):
        raise ValueError("dirty_beam convolved with each scale kernel must have a positive peak")

    scale_factors = biases / max_scaled_dirty_beams

    # Cross terms need every scale kernel to already exist, so this must be a separate pass
    # from the one above (which is still populating scale_kernels one scale at a time).
    for i in range(number_of_scales):
        for j in range(i, number_of_scales):
            cross_terms[(i, j)] = cast(
                NDArray[np.float64],
                signal.convolve(scaled_dirty_beams[:, :, i], scale_kernels[:, :, j], mode="same"),
            )

    # Clean loop
    for i in range(niter):
        # Correct each scale's response before comparing peaks, so unit-sum kernels do not
        # systematically favour the point scale over extended components. The bias is only used
        # to choose the scale and location, the flux committed below is the unbiased one.
        responses = scaled_residuals * scale_factors
        max_index: int = np.argmax(np.abs(responses)).astype(int)
        max_x: int
        max_y: int
        max_scale: int
        max_x, max_y, max_scale = (int(x) for x in np.unravel_index(max_index, scaled_residuals.shape))

        response = responses[max_x, max_y, max_scale]
        strength = scaled_residuals[max_x, max_y, max_scale] / max_scaled_dirty_beams[max_scale]

        logger.info(f"Iter: {i}, max scale: {max_scale}, strength: {strength}")

        # Fixed reference response from the first iteration, used by the divergence check
        # below
        if reference_residual is None:
            reference_residual = float(np.abs(response))

        # Stop if the largest scale's found component is negative: a strong indication that
        # scale is fitting noise/sidelobes rather than real large-scale structure. With a single
        # scale this scale is trivially both the smallest and the largest, so the check reduces
        # to clean()'s unconditional "stop on negative" -- which is the behaviour we want here too.
        if max_scale == number_of_scales - 1 and strength < 0:
            logger.info("Reached negative on largest scale")
            break

        # Stop if this iteration's response has grown by more than 50% over the first
        # iteration's response: clean is diverging rather than converging
        if np.abs(response) - reference_residual > reference_residual / 2.0:
            logger.info("Diverging")
            break

        # Loop gain
        strength = strength * gain

        # Window into the oversized dirty beam / cross terms such that their peak lands
        # exactly at (max_x, max_y)
        xr = slice(beam_peak[0] - max_x, beam_peak[0] - max_x + dirty_map.shape[0])
        yr = slice(beam_peak[1] - max_y, beam_peak[1] - max_y + dirty_map.shape[1])

        comp = strength * shift(scale_kernels[:, :, max_scale], (max_x - map_peak[0], max_y - map_peak[1]), order=0)

        # Add this component to current model
        model = np.add(model, comp)

        # Update all images using precomputed terms
        for j, _ in enumerate(scale_sizes):
            if j > max_scale:
                cross_term = cross_terms[(max_scale, j)]
            else:
                cross_term = cross_terms[(j, max_scale)]

            comp = strength * cross_term[xr, yr]

            scaled_residuals[:, :, j] = np.subtract(scaled_residuals[:, :, j], comp)

        # End max(res(a)) or niter
        if np.abs(scaled_residuals[:, :, max_scale]).max() <= thres:
            logger.info("Threshold reached")
            break

    else:
        logger.info("Max iterations reached")

    return _restore(
        model, scaled_residuals.sum(axis=2), dirty_beam.shape, pixel_size, clean_beam_width, convolve_residual
    )


def _restore(
    model: NDArray[np.float64],
    residual: NDArray[np.float64],
    beam_shape: tuple[int, ...],
    pixel_size: Quantity[u.arcsec / u.pix],
    clean_beam_width: Quantity | None,
    convolve_residual: bool,
) -> Any:
    r"""
    Convolve the model with a gaussian clean beam and add the scaled residual, if there is a clean beam.

    Returns ``(clean_map, model, residual)`` with a clean beam otherwise just ``(model, residual)``.
    """
    if clean_beam_width is None:
        # Add residuals B_G * I^M + I^R
        return model, residual

    # Convolve model with clean beam B_G * I^M
    x_stdev = (clean_beam_width / pixel_size[0]).to_value(u.pix) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y_stdev = (clean_beam_width / pixel_size[1]).to_value(u.pix) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    clean_beam = Gaussian2DKernel(x_stdev, y_stdev, x_size=beam_shape[1], y_size=beam_shape[0]).array

    # Normalise beam
    clean_beam = clean_beam / clean_beam.max()

    clean_map = signal.convolve2d(model, clean_beam / clean_beam.sum(), mode="same") / (
        pixel_size[0].value * pixel_size[1].value
    )

    # Scale residual map with model and scale
    residual = residual / clean_beam.sum() / (pixel_size[0].value * pixel_size[1].value)
    if convolve_residual:
        residual = signal.convolve2d(residual, clean_beam / clean_beam.sum(), mode="same")

    return clean_map + residual, model, residual


def _radii_squared(shape: tuple[int, int]) -> NDArray[np.float64]:
    # Same integer centre, `(L - 1) // 2`, as `_component` and `signal.convolve(mode="same")`
    y, x = np.indices(shape)
    return cast(NDArray[np.float64], (y - (shape[0] - 1) // 2) ** 2 + (x - (shape[1] - 1) // 2) ** 2)


def _gaussian_basis(fwhm: float, shape: tuple[int, int]) -> NDArray[np.float64]:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = np.exp(-_radii_squared(shape) / (2.0 * sigma**2))
    return cast(NDArray[np.float64], kernel / kernel.sum())


def _paraboloid_basis(fwhm: float, shape: tuple[int, int]) -> NDArray[np.float64]:
    # 1 - (r / radius)**2 is at half maximum when r = radius / sqrt(2), so the FWHM is sqrt(2) * radius
    radius = fwhm / np.sqrt(2.0)
    kernel = np.clip(1.0 - _radii_squared(shape) / radius**2, 0.0, None)
    return cast(NDArray[np.float64], kernel / kernel.sum())


def _psf_fwhm(beam: NDArray[np.float64]) -> float:
    r"""
    Measure the FWHM, in pixels, of the central peak of a dirty beam.

    Follows the half maximum search of ``stx_vis_multiscale_pixon_gen``: along the row through the
    peak, between the nearest negative samples either side of it, take the samples closest to half
    of the peak value. Returns at least one pixel.
    """
    row, col = np.unravel_index(np.argmax(beam), beam.shape)
    cut = beam[row, :]
    half = cut[col] / 2.0

    negative = np.flatnonzero(cut < 0)
    before, after = negative[negative < col], negative[negative > col]
    left = before.max() if before.size else 0
    right = after.min() if after.size else cut.size - 1

    start = left + np.argmin(np.abs(cut[left : col + 1] - half))
    stop = col + np.argmin(np.abs(cut[col : right + 1] - half))
    return float(max(stop - start, 1))


def _ms_clean_groups(
    dirty_map: Any,
    dirty_beam: Any,
    pixel_size: Quantity[u.arcsec / u.pix],
    *,
    scales: Sequence[float] | None,
    scale_biases: Sequence[float] | None,
    basis: Literal["gaussian", "paraboloid"],
    clean_beam_width: Quantity | None,
    gain: float,
    thres: float,
    niter: int,
    convolve_residual: bool,
) -> Any:
    r"""
    Multi-scale CLEAN of dirty maps and beams split by visibility group, see `ms_clean`.
    """
    group_maps = np.asarray(Quantity(dirty_map).value, dtype=float)
    group_beams = np.asarray(Quantity(dirty_beam).value, dtype=float)
    n_groups, ny, nx = group_maps.shape
    if group_beams.ndim != 3 or group_beams.shape[0] != n_groups:
        raise ValueError("dirty_beam must be a stack with one beam per group of dirty_map")
    if basis not in ("gaussian", "paraboloid"):
        raise ValueError(f"basis must be 'gaussian' or 'paraboloid', got {basis!r}")

    measured = np.array([_psf_fwhm(beam) for beam in group_beams])
    if scales is None:
        # Finest to coarsest, so the first scale is the one that is favoured
        order = np.argsort(measured, kind="stable")
        widths = measured[order]
    else:
        if list(scales) != sorted(scales):
            raise ValueError(f"scales must be in ascending order, got {list(scales)}")
        order = np.arange(n_groups)
        widths = np.asarray(scales, dtype=float)
    if widths.shape != (n_groups,):
        raise ValueError(f"scales must have one entry per group ({n_groups}), got {widths.size}")
    group_maps, group_beams = group_maps[order], group_beams[order]

    if scale_biases is None:
        biases = widths[0] / widths
    else:
        biases = np.asarray(scale_biases, dtype=float)
        if biases.shape != (n_groups,):
            raise ValueError(f"scale_biases must have one entry per group ({n_groups}), got {biases.size}")
        biases = biases[order]

    map_peak = (ny - 1) // 2, (nx - 1) // 2
    beam_peak = (group_beams.shape[1] - 1) // 2, (group_beams.shape[2] - 1) // 2

    make_basis = _gaussian_basis if basis == "gaussian" else _paraboloid_basis
    kernels = np.stack([make_basis(width, (ny, nx)) for width in widths])

    # cross_terms[i, k] is the response of group k's dirty beam to a component of scale i
    cross_terms = np.stack([[signal.convolve(beam, kernel, mode="same") for beam in group_beams] for kernel in kernels])
    self_peaks = np.array([cross_terms[i, i].max() for i in range(n_groups)])
    if np.any(self_peaks <= 0):
        raise ValueError("dirty_beam convolved with each scale kernel must have a positive peak")

    # Rescale each group's dirty map by the peak of its own initial map so groups of very different
    # size are comparable, a group that is only sidelobes/noise (its largest value is negative) is
    # never searched.
    initial_max, initial_min = group_maps.max(axis=(1, 2)), group_maps.min(axis=(1, 2))
    usable = (initial_max > 0) & (-initial_min <= initial_max)
    search_weights = np.where(usable, biases / np.where(usable, initial_max, 1.0), 0.0)

    residuals = group_maps.copy()
    model = np.zeros((ny, nx))
    reference_score: float | None = None

    for i in range(niter):
        scores = np.abs(residuals) * search_weights[:, None, None]
        if scores.max() <= 0:
            logger.info("No scale left to clean")
            break
        max_scale, max_x, max_y = (int(x) for x in np.unravel_index(np.argmax(scores), scores.shape))
        score = scores[max_scale, max_x, max_y]
        peak = residuals[max_scale, max_x, max_y]

        logger.info(f"Iter: {i}, max scale: {max_scale}, strength: {peak}, location: {max_x, max_y}")

        if reference_score is None:
            reference_score = float(score)

        # Stop if the largest scale's found component is negative, see `ms_clean`.
        if max_scale == n_groups - 1 and peak < 0:
            logger.info("Reached negative on largest scale")
            break

        if score - reference_score > reference_score / 2.0:
            logger.info("Diverging")
            break

        # The bias was only for choosing the scale, use the true residual peak for the flux
        flux = gain * peak / self_peaks[max_scale]

        xr = slice(beam_peak[0] - max_x, beam_peak[0] - max_x + ny)
        yr = slice(beam_peak[1] - max_y, beam_peak[1] - max_y + nx)

        model += flux * shift(kernels[max_scale], (max_x - map_peak[0], max_y - map_peak[1]), order=0)
        residuals -= flux * cross_terms[max_scale][:, xr, yr]

        if np.abs(residuals.sum(axis=0)).max() <= thres:
            logger.info("Threshold reached")
            break

    else:
        logger.info("Max iterations reached")

    residual = residuals.sum(axis=0)
    if clean_beam_width is None:
        # Background estimate R / T, T being the integral of the full dirty beam over the map
        total_beam = group_beams.sum(axis=0)
        window = total_beam[beam_peak[0] - map_peak[0] :, beam_peak[1] - map_peak[1] :][:ny, :nx]
        return model, residual / window.sum()

    return _restore(model, residual, group_beams.shape[1:], pixel_size, clean_beam_width, convolve_residual)


@add_common_docstring(**_clean_parameters())  # type: ignore[untyped-decorator]
def vis_ms_clean(
    vis: Visibilities,
    shape: Quantity[u.pix],
    pixel_size: Quantity[u.arcsec / u.pix],
    scales: Sequence[float] | None,
    clean_beam_width: Quantity[u.arcsec] | Literal["auto"] | None = "auto",
    niter: int = 5000,
    map: bool | None = True,  # noqa: A002
    gain: float = 0.1,
    thres: float = 0.01,
    convolve_residual: bool = False,
    groups: Sequence[ArrayLike] | None = None,
    scale_biases: Sequence[float] | None = None,
    basis: Literal["gaussian", "paraboloid"] = "gaussian",
    scheme: str = "natural",
) -> Quantity | NDArray[np.float64]:
    r"""
    Clean the visibilities using a multiscale clean method.

    A wrapper around `ms_clean` which calculates the dirty map and psf.

    Parameters
    ----------
    vis :
        The visibilities to clean
    shape :
        Size of map
    pixel_size :
        The pixel size in arcsec
    map :
        Return a `sunpy.map.Map` by default or array only if `False`
    {scales}
    {clean_beam_width}
        If not set this is 4 arcsec, or without a clean beam if ``groups`` are used.
    {gain}
    {thres}
    {niter}
    {convolve_residual}
    {groups}
    {scale_biases}
    {basis}
    {scheme}

    {returns_ms_clean}

    {notes_ms_clean}
    """
    # The paper's algorithm adds the residual to the model directly, there is no clean beam to restore with
    if isinstance(clean_beam_width, str):
        clean_beam_width = 4.0 * u.arcsec if groups is None else None

    if groups is None:
        dirty_map = vis_to_map(vis, shape=shape, pixel_size=pixel_size, scheme=scheme)
        dirty_beam = vis_psf_image(vis, shape=shape * 3, pixel_size=pixel_size, scheme=scheme)
        meta = dirty_map.meta
        dirty_map, dirty_beam = dirty_map.data, dirty_beam
    else:
        dirty_map, dirty_beam = _group_images(vis, groups, shape, pixel_size, scheme)
        meta = generate_header(
            vis,
            shape=validate_and_expand_kwarg(shape, "shape"),
            pixel_size=validate_and_expand_kwarg(pixel_size, "pixel_size"),
        )
    result = ms_clean(
        dirty_map,
        dirty_beam,
        pixel_size=pixel_size,
        scales=scales,
        clean_beam_width=clean_beam_width,
        gain=gain,
        thres=thres,
        niter=niter,
        convolve_residual=convolve_residual,
        scale_biases=scale_biases,
        basis=basis,
    )
    # ms_clean only returns (clean_map, model, residual) when clean_beam_width is set; with
    # clean_beam_width=None it returns just (model, residual), so clean_map has to be derived.
    if clean_beam_width is not None:
        clean_map, model, residual = result
    else:
        model, residual = result
        clean_map = model + residual
    if not map:
        return clean_map, model, residual

    return [Map((data, meta)) for data in (clean_map, model, residual)]


def _group_images(
    vis: Visibilities,
    groups: Sequence[ArrayLike],
    shape: Quantity[u.pix],
    pixel_size: Quantity[u.arcsec / u.pix],
    scheme: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Back project each group of visibilities into a dirty map and dirty beam.

    The weights are normalised over every visibility that is in a group, so the maps (and the
    beams) of the groups sum to the dirty map (and beam) of all of them. For natural weights this
    is the ``countvis / nvis`` rescaling of the group's share of the visibilities.
    """
    shape = validate_and_expand_kwarg(shape, "shape").to(u.pix)
    pixel_size = validate_and_expand_kwarg(pixel_size, "pixel_size")
    # Odd, as for `vis_psf_image`, so the peak of the beam is in exactly one pixel
    beam_shape = [s // 2 * 2 + 1 for s in (shape * 3).to_value(u.pix)] * u.pix

    indices = [np.asarray(group, dtype=int) for group in groups]
    if any(index.size == 0 for index in indices):
        raise ValueError("every group must contain at least one visibility")
    used = np.concatenate(indices)
    if np.unique(used).size != used.size:
        raise ValueError("groups must not overlap")

    weights = get_weights(vis, scheme=scheme)
    weights = weights / weights[used].sum()

    maps, beams = [], []
    for index in indices:
        kwargs = {"u": vis.u[index], "v": vis.v[index], "weights": weights[index], "pixel_size": pixel_size}
        dirty_map = idft_map(vis.visibilities[index], shape=shape, phase_center=[0.0, 0.0] * u.arcsec, **kwargs)
        dirty_beam = idft_map(np.ones(index.size) * vis.visibilities.unit, shape=beam_shape, **kwargs)
        maps.append(Quantity(dirty_map).value)
        beams.append(Quantity(dirty_beam).value)

    return np.stack(maps), np.stack(beams)


def _radial_prolate_sphereoidal(nu: float) -> float:  # pyright: ignore[reportUnusedFunction]
    r"""
    Calculate prolate spheroidal wave function approximation.

    Parameters
    ----------
    nu : `float`
        The radial value to evaluate the function at

    Returns
    -------
    `float`
        The amplitude of the the prolate spheroid function at `nu`

    Notes
    -----
    Note this is a direct translation of the on the implementation the CASA code reference by [1] \
    and can be found here Link_

    .. _Link: https://github.com/casacore/casacore/blob/f4dc1c36287c766796ce3375cebdfc8af797a388/lattices/LatticeMath/LatticeCleaner.tcc#L956 #noqa

    """
    if nu <= 0:
        return 1.0
    elif nu >= 1.0:
        return 0.0
    else:
        n_p = 5
        n_q = 3

        p = np.zeros((n_p, 2))
        q = np.zeros((n_q, 2))

        p[0, 0] = 8.203343e-2
        p[1, 0] = -3.644705e-1
        p[2, 0] = 6.278660e-1
        p[3, 0] = -5.335581e-1
        p[4, 0] = 2.312756e-1
        p[0, 1] = 4.028559e-3
        p[1, 1] = -3.697768e-2
        p[2, 1] = 1.021332e-1
        p[3, 1] = -1.201436e-1
        p[4, 1] = 6.412774e-2

        q[0, 0] = 1.0000000e0
        q[1, 0] = 8.212018e-1
        q[2, 0] = 2.078043e-1
        q[0, 1] = 1.0000000e0
        q[1, 1] = 9.599102e-1
        q[2, 1] = 2.918724e-1

        part = 0
        nuend = 0.0

        if 0.0 <= nu < 0.75:
            part = 0
            nuend = 0.75
        elif 0.75 <= nu <= 1.00:
            part = 1
            nuend = 1.0

        top = p[0, part]
        delnusq = np.power(nu, 2.0) - np.power(nuend, 2.0)

        for k in range(1, n_p):
            top += p[k, part] * np.power(delnusq, k)

        bot = q[0, part]
        for k in range(1, n_q):
            bot += q[k, part] * np.power(delnusq, k)

        if bot != 0.0:
            return float(top / bot)
        else:
            return 0


def _vec_radial_prolate_sphereoidal(nu: ArrayLike) -> NDArray[np.float64]:
    r"""
    Calculate prolate spheroidal wave function approximation.

    Parameters
    ----------
    nu : `float` array
        The radial value to evaluate the function at

    Returns
    -------
    `float`
        The amplitude of the the prolate spheroid function at `nu`

    Notes
    -----
    Note this is based on the implementation the CASA code reference by [1] and can be found here
    Link_

    .. _Link: https://github.com/casacore/casacore/blob/f4dc1c36287c766796ce3375cebdfc8af797a388/lattices/LatticeMath/LatticeCleaner.tcc#L956 #noqa

    """
    nu = np.array(nu)

    n_p = 5
    n_q = 3

    p = np.zeros((n_p, 2))
    q = np.zeros((n_q, 2))

    p[0, 0] = 8.203343e-2
    p[1, 0] = -3.644705e-1
    p[2, 0] = 6.278660e-1
    p[3, 0] = -5.335581e-1
    p[4, 0] = 2.312756e-1
    p[0, 1] = 4.028559e-3
    p[1, 1] = -3.697768e-2
    p[2, 1] = 1.021332e-1
    p[3, 1] = -1.201436e-1
    p[4, 1] = 6.412774e-2

    q[0, 0] = 1.0000000e0
    q[1, 0] = 8.212018e-1
    q[2, 0] = 2.078043e-1
    q[0, 1] = 1.0000000e0
    q[1, 1] = 9.599102e-1
    q[2, 1] = 2.918724e-1

    lower = np.where((nu >= 0.0) & (nu < 0.75))  # part = 0, nuend = 0.75
    upper = np.where((nu >= 0.75) & (nu <= 1.00))  # part = 1, nuend = 1.0

    delnusq = np.zeros_like(nu)
    delnusq[lower] = np.power(nu[lower], 2.0) - np.power(0.75, 2.0)
    delnusq[upper] = np.power(nu[upper], 2.0) - np.power(1.00, 2.0)

    top = np.zeros_like(nu, dtype=float)
    top[lower] = p[0, 0]
    top[upper] = p[0, 1]

    k = np.arange(1, n_p)
    top[lower] += np.sum(p[k, 0, np.newaxis] * np.power(delnusq[lower], k[..., np.newaxis]), axis=0)
    top[upper] += np.sum(p[k, 1, np.newaxis] * np.power(delnusq[upper], k[..., np.newaxis]), axis=0)

    bot = np.zeros_like(nu, dtype=float)
    bot[lower] = q[0, 0]
    bot[upper] = q[0, 1]

    j = np.arange(1, n_q)
    bot[lower] += np.sum(q[j, 0, np.newaxis] * np.power(delnusq[lower], j[..., np.newaxis]), axis=0)
    bot[upper] += np.sum(q[j, 1, np.newaxis] * np.power(delnusq[upper], j[..., np.newaxis]), axis=0)

    out = np.zeros(nu.shape)
    out[bot != 0] = top[bot != 0] / bot[bot != 0]
    out = np.where(nu <= 0, 1.0, out)
    out = np.where(nu >= 1, 0.0, out)

    return out


def _component(scale: float, shape: tuple[int, ...]) -> NDArray[np.float64]:
    r"""

    Parameters
    ----------
    scale

    Returns
    -------

    """
    # Integer center pixel (not the previous `(shape - 1) / 2.0`, which is a half-integer for
    # even `shape` and left the kernel with no single, well-defined peak pixel to align with a
    # target location). This must match scipy.signal.convolve's `mode="same"` centering
    # convention for an even-length kernel, which is `(L - 1) // 2`, not `L // 2`.
    # `(shape - 1) // 2` is unchanged from the previous convention for odd `shape`, so this only
    # changes behaviour for even shapes.
    refx, refy = (shape[0] - 1) // 2, (shape[1] - 1) // 2

    if scale == 0.0:
        wave_amp = np.zeros(shape)
        wave_amp[int(refx), int(refy)] = 1
        return wave_amp

    xy = np.mgrid[0 : shape[0] : 1, 0 : shape[1] : 1]
    radii_squared = ((xy[0, :, :] - refx) / scale) ** 2 + ((xy[1, :, :] - refy) / scale) ** 2

    rad_zeros_indices = radii_squared <= 0.0
    amp_zero_indices = radii_squared >= 1.0

    wave_amp = _vec_radial_prolate_sphereoidal(np.sqrt(radii_squared.reshape(radii_squared.size)))
    wave_amp = wave_amp.reshape(shape)
    wave_amp[rad_zeros_indices] = _vec_radial_prolate_sphereoidal([0])[0]

    wave_amp = wave_amp * (1 - radii_squared)

    wave_amp[amp_zero_indices] = 0.0

    return wave_amp
