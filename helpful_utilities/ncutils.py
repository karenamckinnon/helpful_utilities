import xarray as xr
import numpy as np


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def lon_to_360(da, lon_name='lon'):
    """Returns the xr.DataArray with longitude changed from -180, 180 to 0, 360"""
    da = da.assign_coords({lon_name: (da[lon_name] + 360) % 360})
    return da.sortby(lon_name)


def lon_to_180(da, lon_name='lon'):
    """Returns the xr.DataArray with longitude changed from 0, 360 to -180, 180"""
    da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)})
    return da.sortby(lon_name)


# Via Stephan Hoyer, https://github.com/pydata/xarray/issues/501
def transform_from_latlon(lat, lon):
    from affine import Affine
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, fill=np.nan, **kwargs):
    from rasterio import features

    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))


def regrid(da, new_lat, new_lon, wgt_file=''):
    """
    Regrid either an xarray dataset or dataarray

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        Data to be regridded. Needs to be in lat/lon
    new_lat : numpy.array
        New latitudes after regridding, must be increasing
    new_lon : numpy.array
        New longitudes after regridding, must be increasing
    wgt_file : str
        If desired, the weight file to be reused

    Returns
    -------
    da_rg : xr.Dataset or xr.DataArray
        Regridded datr
    """

    import xesmf as xe
    import os
    if os.path.isfile(wgt_file):
        reuse_weights = True
    else:
        reuse_weights = False

    if 'latitude' in da.coords:
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    if (da.lon.min() < 0) & (np.min(new_lon) > 0):
        da = lon_to_360(da)
    if (da.lon.min() > 0) & (np.min(new_lon) < 0):
        da = lon_to_180(da)

    # Latitude and longitude should be increasing
    assert (new_lat[0] < new_lat[1])
    assert (new_lon[0] < new_lon[1])
    da = da.sortby('lat')
    da = da.sortby('lon')

    regridder = xe.Regridder({'lat': da.lat, 'lon': da.lon},
                             {'lat': new_lat, 'lon': new_lon},
                             'bilinear',
                             periodic=True, reuse_weights=reuse_weights,
                             filename=wgt_file)

    da_rg = regridder(da)
    return da_rg
