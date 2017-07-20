"""Converts map_data into visibilities

Each visibility structure corresponds to a single subcollimator, harmonic,
u, v, energy and time combination.
Reference: https://darts.jaxa.jp/pub/ssw/hessi/idl/atest/hsi_vis_map2vis.pro
"""
#
# TODO
#
__authors__ = ["Gábor Péterffy"]
__email__ = ["peterffy95@gmail.com"]
__license__ = "xxx"

import numpy as np

def vis_spatial_frequency_weighting(vis, spatial_frequency_weight = 1.0, uniform_weighting = True):
    """
    This function returns the spatial_frequency_weighting factor for
    each visibility in the input vis bag. The weights are not normalized. That's
    left for the task that uses them

    Parameters
    ----------
    vis : np.array
        np.array of hsi_vis visibility structure (Visibility bag)
        
    spatial_frequency_weight : np.array, float
        If uniform_weighting is not true., and if this is a single scalar it will use natural weighting,
        if this is an array of values the dimensions of the visibilities and weights must agree.
        
        If uniform_weighting is True, the spatial weight will be computed based on sqrt(vis.u^2 + vis.v^2))
        
        The result will be stored in this variable
        
    uniform_weighting : bool
        
    Returns
    -------
    
    See Also
    --------

    Notes
    -----
    
    Reference
    ----------
    | https://darts.isas.jaxa.jp/pub/ssw/gen/idl/image/vis/vis_spatial_frequency_weighting.pro
    """
    nvis = vis.shape[0]
    u = []
    v = []
    for i in vis:
        u.append(i.u)
        v.append(i.v)
    u = np.array(u)
    v = np.array(v)
    if uniform_weighting:
        spatial_frequency_weight = np.sqrt(np.square(u)+np.square(v))
    nsfw = spatial_frequency_weight.shape[0]
    if nsfw == 1:
        spatial_frequency_weight = np.repeate(spatial_frequency_weight, nvis)
    nsfw = spatial_frequency_weight.shape[0]
    if nsfw != nvis:
        raise ValueError("Visibility bag, vis, and SPATIAL_FREQUENCY_WEIGHTs cannot be reconciled.")

def vis_bpmap_get_spatial_weights(visin = np.ones((visin.shape[0],)), spatial_frequency_weight, uniform_weighting = False):
    """
    Calculate sthe spatial weights for the visibilities

    Parameters
    ----------
    visin : np.array
        np.array of hsi_vis visibility structure (Visibility bag)
    
        
    Returns
    -------
    
    See Also
    --------

    Notes
    -----
    
    Reference
    ----------
    | https://darts.isas.jaxa.jp/pub/ssw/gen/idl/image/vis/vis_bpmap.pro
    """
    spatial_frequency_weight = np.di
    vide(spatial_frequency_weight, np.array(np.sum(spatial_frequency_weight)))
    vis_spatial_frequency_weighting.

def vis_bpmap_get_xypi( npx, pixel, verbose = False):
        """
    Calculate sthe spatial weights for the visibilities

    Parameters
    ----------
    visin : np.array
        np.array of hsi_vis visibility structure (Visibility bag)
    
        
    Returns
    -------
    Spatial Frequency weight
    
    See Also
    --------

    Notes
    -----
    
    Reference
    ----------
    | https://darts.isas.jaxa.jp/pub/ssw/gen/idl/image/vis/vis_bpmap.pro
    """
    global xypi_com
    global xypi
    global npx_sav
    global pixel_sav
    if !(npx != npx_sav) or !(pixel_sav != pixel):
        xypi = 
    
def vis_bpmap():
    
