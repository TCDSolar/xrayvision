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
    each visibility in the input vis visibility bag. The weights are not normalized. That's
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

def vis_bpmap_get_spatial_weights(visin, spatial_frequency_weight = 0, uniform_weighting = False):
    """
    Calculate sthe spatial weights for the visibilities

    Parameters
    ----------
    visin : np.array
        np.array of hsi_vis visibility structure (Visibility bag)
    
        
    Returns
    -------
        spatial_frequency_weight: np.array
            The value of spatial_frequency_weight will be modified, but it will be returned also
    
    See Also
    --------

    Notes
    -----
    
    Reference
    ----------
    | https://darts.isas.jaxa.jp/pub/ssw/gen/idl/image/vis/vis_bpmap.pro
    """
    if type(spatial_frequency_weight) == int:
         spatial_frequency_weight = np.ones((visin.shape[0],))
    spatial_frequency_weight = np.divide(spatial_frequency_weight, np.sum(spatial_frequency_weight))
    return spatial_frequency_weight

def pixel_coord(image_dim = (64,64)):
    """
    This function converts image dimensions into a 2d array
    of x and y coordinates in pixel units relative to the center of the image.

    Parameters
    ----------
    image_dim : tuple
        (Number of pixels in x, number of pixels in y), default is (64, 64)
        
    Returns
    -------
    mapindex : np.array
        Two dimensional numpy array which is containing x, y coordinates
        relative to the image center
    
    See Also
    --------

    Notes
    -----
    
    Reference
    ----------
    | https://darts.isas.jaxa.jp/pub/ssw/gen/idl/image/pixel_coord.pro
    """
    data = np.ones(image_dim)
    mapindex = np.asarray(np.nonzero(data))
    npixel = np.count_nonzero(data)
    mapindex[0] = np.subtract(mapindex[0], float(image_dim[0]-1.) / 2.)
    mapindex[1] = np.subtract(mapindex[1], float(image_dim[1]-1.) / 2.)
    return mapindex

def vis_bpmap_get_xypi(npx : int, pixel):
    """
    Calculate sthe spatial weights for the visibilities

    Parameters
    ----------
    npx : int
        x size of the image
    pixel : float
        
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
    if 'xypi_com' not in globals():
        xypi_com = 0.
        xypi = 0.
        npx_sav = 0.
        pixel_sav = 0.
    if (npx != npx_sav) or (pixel_sav != pixel):
        xypi = np.reshape(pixel_coord((npx, npx)), (2, npx, npx))
        xypi = np.multiply(xypi, 2. * np.pi * pixel)
        pixel_sav = float(pixel)
        npx_sav = int(npx)
    return xypi
    
def vis_bpmap(visin, bp_fov = 80., pixel = 0., uniform_weighting = False,
              spatial_frequency_weight = 0):
    """
    This procedure makes a backprojection map from a visibility bag

    Parameters
    ----------
    visin : np.array
        np.array of hsi_vis visibility structure (Visibility bag)
    bp_fov : float
        field of view (arcsec) Default = 80. arcsec
    pixel : float
        size of a pixel
    uniform_weighting : bool
        If true, changes subcollimator weighting from default (NATURAL) to UNIFORM
        Default if false.
    spatial_frequency_weight : weighting for each collimator, set by
        UNIFORM_WEIGHTING if used. The number of weights should either equal the number of
        unique sub-collimators or the number of visibilities
        
    Returns
    -------
        Backprojection map of the given visibility bag
    
    See Also
    --------

    Notes
    -----
    
    Reference
    ----------
    | https://darts.isas.jaxa.jp/pub/ssw/gen/idl/image/vis/vis_spatial_frequency_weighting.pro
    """
    if type(spatial_frequency_weight) == int:
        spatial_frequency_weight = np.ones((visin.shape[0],))
    fov = float(bp_fov)
    pixel = fov / 200.
    npx = int(fov / pixel)
    MAP = np.zeros((npx, npx))
    # For RHESSI case, preserve 9 spatial weights if they are passed
    spatial_frequency_weight = vis_bpmap_get_spatial_weights(visin,
                                                             spatial_frequency_weight,
                                                             uniform_weighting)
    xypi = vis_bpmap_get_xypi(npx, pixel)
    ic = 1.0+1.0j
    nvis = visin.shape[0]
    for nv in range(nvis):
        uv = np.add(np.multiply(xypi[0, :, :], visin[nv].u ),
                    np.multiply(xypi[1, :, :], visin[nv].v ))
        MAP = np.add(MAP, np.multiply(np.add(np.cos(uv), np.multiply(np.sin(uv), (0+1j))),
                           spatial_frequency_weight[nv] * visin[nv].obsvis))
    return MAP
    
