from subprocess import check_call
import click
import xarray as xr
from glob import glob
import numpy as np


@click.command()
@click.option('--year', help='Year of data to process', type=click.INT)
@click.option('--era5_var_name', help='official ERA5 variable name', type=click.STRING)
@click.option('--calc_max', help='Save daily max?', type=click.BOOL)
@click.option('--calc_min', help='Save daily min?', type=click.BOOL)
@click.option('--delete_hourly', help='Delete original hourly data?', type=click.BOOL)
def aggregate_hourly(year, era5_var_name, calc_max, calc_min, delete_hourly, era5_dir='/home/data/ERA5'):

    files = sorted(glob('%s/hourly/%s/%s_%04i_*hourly.nc' % (era5_dir, era5_var_name, era5_var_name, year)))
    print(files)
    if len(files) > 1:  # case of the last year, when the last month is in a different file
        da = []
        for f in files:
            this_da = xr.open_dataarray(f)

            if 'expver' in this_da.dims:
                missing_val = 261.3532
                # where to switch from ERA5 to ERA5T
                start5 = np.where((this_da[:, 1, 0, 0] != missing_val) & ~np.isnan(this_da[:, 1, 0, 0]))[0][0]
                tmp = this_da.sel(expver=1)
                tmp[start5:, ...] = this_da.sel(expver=5)[start5:, ...]
                this_da = tmp

            da.append(this_da)
        da = xr.concat(da, dim='time')
    else:
        da = xr.open_dataarray(files[0])
        if 'expver' in da.dims:
            da = da.sel(expver=1)

    era5_daily_dir = '%s/day' % era5_dir
    # calculate daily average
    da_daily_average = da.resample(time='D').mean()
    savedir = '%s/%s' % (era5_daily_dir, era5_var_name)
    cmd = 'mkdir -p %s' % savedir
    check_call(cmd.split())
    da_daily_average.to_netcdf('%s/%s_%04i.nc' % (savedir, era5_var_name, year))
    del da_daily_average

    # calculate daily max
    if calc_max:
        da_daily_max = da.resample(time='D').max()
        savedir = '%s/%s_x' % (era5_daily_dir, era5_var_name)
        cmd = 'mkdir -p %s' % savedir
        check_call(cmd.split())
        da_daily_max.to_netcdf('%s/%s_x_%04i.nc' % (savedir, era5_var_name, year))
        del da_daily_max

    # calculate daily min
    if calc_min:
        da_daily_min = da.resample(time='D').min()
        savedir = '%s/%s_n' % (era5_daily_dir, era5_var_name)
        cmd = 'mkdir -p %s' % savedir
        check_call(cmd.split())
        da_daily_min.to_netcdf('%s/%s_n_%04i.nc' % (savedir, era5_var_name, year))
        del da_daily_min

    if delete_hourly:
        # delete original file of hourly data
        cmd = 'rm -f %s/hourly/%s/%s_%04i_*hourly.nc' % (era5_dir, era5_var_name, era5_var_name, year)
        check_call(cmd.split())


if __name__ == '__main__':
    aggregate_hourly()
