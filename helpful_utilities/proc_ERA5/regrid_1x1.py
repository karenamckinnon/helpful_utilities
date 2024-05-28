import xesmf as xe
import xarray as xr
from glob import glob
import os
import numpy as np
from subprocess import check_call
import click


@click.command()
@click.option('--era5_var_name', help='official ERA5 variable name, or with _n and _x for min/max', type=click.STRING)
@click.option('--time_resolution', help='time resolution of the data to regrid e.g. day', type=click.STRING)
def regrid_1x1(era5_var_name, time_resolution, era5_dir='/home/data/ERA5'):

    lat1x1 = np.arange(-89.5, 90)
    lon1x1 = np.arange(0.5, 360)

    base_dir = '%s/%s/%s' % (era5_dir, time_resolution, era5_var_name)
    files = sorted(glob('%s/%s_*.nc' % (base_dir, era5_var_name)))

    # make 1x1 dir if not already present
    era5_dir_1x1 = '%s/1x1' % (base_dir)
    cmd = 'mkdir -p %s' % era5_dir_1x1
    check_call(cmd.split())

    for f in files:
        print(f)
        f_new = f.replace('.nc', '_1x1.nc').split('/')[-1]
        f_new = '%s/%s' % (era5_dir_1x1, f_new)

        if os.path.isfile(f_new):
            continue
        wgt_file = '%s/xe_weights_1x1.nc' % (base_dir)
        if os.path.isfile(wgt_file):
            reuse_weights = True
        else:
            reuse_weights = False

        da = xr.open_dataarray(f)
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
        da = da.sortby('lat')

        regridder = xe.Regridder({'lat': da.lat, 'lon': da.lon},
                                 {'lat': lat1x1, 'lon': lon1x1},
                                 'bilinear',
                                 periodic=True, reuse_weights=reuse_weights,
                                 filename=wgt_file)

        da = regridder(da)
        da = da.rename(era5_var_name)
        da.to_netcdf(f_new)


if __name__ == '__main__':
    regrid_1x1()
