from typing import Optional

import numpy as np
from astropy.visualization import quantity_support
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from xrayvision.visibility import Visibilities


def plot_vis(vis: Visibilities, fig: Optional[Figure] = None, **mplkwargs: dict) -> Figure:
    r"""
    Plot visibilities amplitude and phase.

    Plot as a function of the resolution (r) :math:`\sqrt{u^2 + v^2` and angle (theta) :math:`\mathrm{arctan2}(u, v)`.
    Theta is represented as a rotation of the plot symbol.

    Parameters
    ----------
    vis :
        Visibilities
    fig :
        Figure to use if given will use the first and second axes to plot the amplitude and phase.
    mplkwargs :
        Keyword arguments passed to matplotlib

    Returns
    -------

    """
    if fig is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    else:
        axes = fig.get_axes()

    fig.subplots_adjust(hspace=0)

    angles = np.arctan2(vis.u, vis.v)
    size = 1 / np.sqrt(vis.u**2 + vis.v**2)

    with quantity_support():
        for i, _ in enumerate(vis.visibilities):
            transform = Affine2D().rotate(angles[i].to_value("deg"))
            axes[0].scatter(
                size[i], np.absolute(vis.visibilities[i]), marker=MarkerStyle("|", transform=transform), **mplkwargs
            )
            axes[1].scatter(
                size[i],
                np.angle(vis.visibilities[i]).to("deg"),
                marker=MarkerStyle("|", transform=transform),
                **mplkwargs,
            )

    axes[0].set_ylabel(f"Amplitude [{vis.visibilities.unit}]")
    axes[1].set_ylabel("Phase [deg]")
    axes[1].set_xlabel(f"Resolution [{size.unit}]")

    return fig, axes


def plot_vis_diff(visa: Visibilities, visb: Visibilities, fig=None, **mplkwargs):
    r"""
    Plot the difference between amplitude and phase of the visibilities.

    Plot as a function of the resolution :math:`\sqrt{u^2 + v^2` and angle :math:`\mathrm{arctan2}(u, v)`.
    The resolution is used as the x-axis while the angle is displayed as a rotation of the plot symbol.

    Parameters
    ----------
    visa
        Visibilities to plot
    visb
        Visibilities to plot
    fig :
        Figure to use
    mplkwargs
        Keyword arguments passed to matplotlib
    Returns
    -------

    """
    if not (np.all(visa.u == visb.u) and np.all(visb.v == visb.v)):
        raise ValueError("The visibilities must have the same u, v coordinates.")

    if fig is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    else:
        axes = fig.get_axes()

    fig.subplots_adjust(hspace=0)

    angles = np.arctan2(visa.u, visa.v)
    size = 1 / np.sqrt(visa.u**2 + visa.v**2)
    vis_diff = visa.visibilities - visb.visibilities

    for i, _ in enumerate(vis_diff):
        transform = Affine2D().rotate(angles[i].to_value("deg"))
        axes[0].scatter(size[i], np.absolute(vis_diff[i]), marker=MarkerStyle("|", transform=transform), **mplkwargs)
        axes[1].scatter(
            size[i], np.angle(vis_diff[i]).to("deg"), marker=MarkerStyle("|", transform=transform), **mplkwargs
        )

    axes[0].set_ylabel(f"Amplitude Diff [{vis_diff.unit}]")
    axes[1].set_ylabel("Phase Diff [deg]")
    axes[1].set_xlabel(f"Resolution [{size.unit}]")

    return fig, axes
