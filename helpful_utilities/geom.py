"""
Tools for mapping and geometry
"""
import numpy as np


def get_regrid_country(country_name, country_folder, lats, lons, dilate=False):
    """
    Rasterize a Greenland shape file, and then regrid to lats, lons of a desired grid.

    Parameters
    ----------
    country_name : str
        Name of country (e.g. 'Greenland' or 'United States of America')
    country_folder : str
        Path to shape files with country outlines
    lats : numpy.array
        Vector of latitudes to interpolate to
    lons : numpy.array
        Vector of longitudes to interpolate to
    dilate : bool
        Whether to dilate boundary after regridding. Can be helpful for masking

    Returns
    -------
    da_country : xr.DataArray
        Mask (1/0) of desired country
    """
    import geopandas
    from helpful_utilities import ncutils
    from scipy.ndimage.morphology import binary_dilation

    # Get Greenland mask
    countries = geopandas.read_file(country_folder)
    country = countries.query("ADMIN == '%s'" % country_name).reset_index(drop=True)
    da_country = ncutils.rasterize(country['geometry'],
                                   {'lon': np.linspace(-179.5, 180, 1000),
                                    'lat': np.linspace(-90, 90, 1000)})
    da_country = ncutils.lon_to_360(da_country)
    da_country = da_country.fillna(0)
    da_country = da_country.interp({'lat': lats, 'lon': lons})
    if dilate:
        # regridded mask may miss boundary, so do dilation to catch it
        expanded_country = binary_dilation(binary_dilation((da_country > 0).values))
        da_country = da_country.copy(data=expanded_country)

    return da_country
