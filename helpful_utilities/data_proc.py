"""
Tools for common data preprocessing steps
"""
import numpy as np
from helpful_utilities.stats import lowpass_butter
import xarray as xr
import os


def get_smooth_clim(data):
    """
    Estimate a smoothed climatology using a lowpass Butterworth filter with a frequency of 1/30d

    The data is mirrored on either side to address edge issues with the filter.
    """
    idx_data = ~np.isnan(data)
    if idx_data.any():
        vals = data[idx_data]
        nt = len(vals)
        tmp = np.hstack((vals[::-1], vals, vals[::-1]))
        filtered = lowpass_butter(1, 1/30, 3, tmp)
        smooth_data = data.copy()
        smooth_data[idx_data] = filtered[nt:2*nt]

        return smooth_data
    else:
        return data


def get_trend_array(ds, var_name, trend_normalizer=1, dim='year'):
    """
    Get the linear trend of a variable in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing values for the the trend estimate
    var_name : str
        Name of the variable for which we want the trend
    trend_normalizer : int / float
        Multiply the trend by this value (e.g. if we want it per X years)
    dim : str
        The dimension along which to calculate the trend

    Returns
    -------
    da_trend : xr.DataArray
        Trend (with normalization) for the specific variable
    """

    beta = ds.polyfit(deg=1, dim=dim)
    return trend_normalizer*(beta.sel(degree=1)['%s_polyfit_coefficients' % var_name])


def get_day_idx_temperature_percentiles_doy(da, dataname, percentile_width, percentile_base,
                                            start_year, end_year, procdir, detrend=True):
    """
    Given a dataset, identify days that fit into extremal percentile categories. Percentiles are
    a function of the day of year, so results are independent of whether data is deseasonalized first.

    Parameters
    ----------
    da : xr.DataArray
        Data for which to identify percentile categories
    dataname : str
        Name of dataset for saving
    percentile_width : int
        The half-width of the percentile categories (0-100)
    percentile_base : numpy.array
        The middle percentiles to define the categories (0-100)
    start_year : int
        First year of the dataset
    end_year : int
        Last year of the dataset
    procdir : str
        Where to save the output
    detrend : bool
        Whether to linearly detrend the data before identifying the percentiles

    Returns
    -------
    idx_all : xr.Dataset
        Contains boolean indicator of whether something was within a percentile bin or not
    """

    savename = '%s/%s_idx_w%02i_%04i-%04i.nc' % (procdir, dataname, percentile_width,
                                                 start_year, end_year)
    if os.path.isfile(savename):
        idx_all = xr.open_dataset(savename)
    else:
        idx_all = []
        for counter, p in enumerate(percentile_base):

            if counter == 0:  # Process data

                this_da = da.copy().sel(time=slice('%04i' % start_year, '%04i' % end_year))
                if detrend:
                    da_beta = this_da.polyfit(dim='time', deg=1)
                    da_yhat = xr.polyval(this_da['time'], da_beta)['polyfit_coefficients']
                    da_anom = this_da - da_yhat
                else:
                    da_anom = this_da

            if (p + percentile_width) == 100:
                qcut1 = da_anom.groupby('time.dayofyear').quantile((p - percentile_width)/100)
                idx = da_anom.groupby('time.dayofyear') > qcut1
                del qcut1

            elif (p - percentile_width) == 0:
                qcut2 = da_anom.groupby('time.dayofyear').quantile((p + percentile_width)/100)
                idx = da_anom.groupby('time.dayofyear') < qcut2
                del qcut2
            else:
                qcut1 = da_anom.groupby('time.dayofyear').quantile((p - percentile_width)/100)
                qcut2 = da_anom.groupby('time.dayofyear').quantile((p + percentile_width)/100)
                idx = ((da_anom.groupby('time.dayofyear') > qcut1) &
                       (da_anom.groupby('time.dayofyear') < qcut2))
                del qcut1, qcut2
            idx = idx.rename('base_p_%02i' % p)
            if 'quantile' in list(idx.coords):
                idx = idx.drop('quantile')
            if 'expver' in list(idx.coords):
                idx = idx.drop('expver')

            idx_all.append(idx)
        idx_all = xr.merge(idx_all)
        idx_all.to_netcdf(savename)

    return idx_all
