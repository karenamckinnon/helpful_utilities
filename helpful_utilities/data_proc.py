"""
Tools for common data preprocessing steps
"""
import numpy as np
from helpful_utilities.stats import lowpass_butter
import xarray as xr


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
