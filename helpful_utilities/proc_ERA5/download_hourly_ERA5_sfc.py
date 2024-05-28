import cdsapi
import click
from datetime import datetime, timedelta
from subprocess import check_call


@click.command()
@click.option('--year', help='Year of data to process', type=click.INT)
@click.option('--era5_var_name', help='official ERA5 variable name', type=click.STRING)
def download_ERA5_vars_sfc(year, era5_var_name, era5_dir='/home/data/ERA5/hourly'):
    print(year)
    print(era5_var_name)

    # make hourly directory if not already there
    cmd = 'mkdir -p %s/%s' % (era5_dir, era5_var_name)
    check_call(cmd.split())

    # Get date to avoid error of trying to download near-realtime
    today = datetime.now()
    maxdate = today - timedelta(days=6)

    if maxdate.year == year:
        maxmonth = maxdate.month
        maxday = maxdate.day
    else:
        maxmonth = 12
        maxday = 31

    months = ['%02i' % i for i in range(1, maxmonth + 1)]
    days = ['%02i' % i for i in range(1, 32)]
    days_short = ['%02i' % i for i in range(1, maxday + 1)]

    c = cdsapi.Client()

    if maxdate.year == year:
        # get all but current month
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '%s' % era5_var_name,
                'year': '%04i' % year,
                'month': months[:-1],
                'day': days,
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                'grid': [1.0, 1.0],
            },
            '%s/%s/%s_%04i_hourly.nc' % (era5_dir, era5_var_name, era5_var_name, year))

        # then get current month
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '%s' % era5_var_name,
                'year': '%04i' % year,
                'month': months[-1],
                'day': days_short,
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                'grid': [1.0, 1.0],
            },
            '%s/%s/%s_%04i_last_month_hourly.nc' % (era5_dir, era5_var_name, era5_var_name, year))

    else:  # not current year
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': '%s' % era5_var_name,
                'year': '%04i' % year,
                'month': months,
                'day': days,
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
                'grid': [1.0, 1.0],
            },
            '%s/%s/%s_%04i_hourly.nc' % (era5_dir, era5_var_name, era5_var_name, year))


if __name__ == '__main__':
    download_ERA5_vars_sfc()
