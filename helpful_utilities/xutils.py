import numpy as np
import xarray as xr


def annual_average_from_monthly(da):
    """
    Calculate the annual average of a DataArray with monthly data,
    weighted by the number of days in each month and handling NaNs.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dimensions ('time', 'lat', 'lon'), where 'time'
        contains monthly average data.

    Returns
    -------
    annual_mean : xr.DataArray
        DataArray with annual averages, reduced along the 'time' dimension.
    """
    # Ensure time coordinate exists
    if 'time' not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension.")

    # Calculate the number of days in each month
    days_in_month = da['time'].dt.days_in_month

    # Mask the weights where the data is NaN
    valid_mask = da.notnull()
    masked_days_in_month = days_in_month.where(valid_mask, other=0)

    # Weight the data by the number of valid days in each month
    weighted_da = da * masked_days_in_month

    # Group by year and sum the weighted data
    weighted_sum = weighted_da.groupby('time.year').sum(dim='time')

    # Group by year and sum the valid number of days
    days_per_year = masked_days_in_month.groupby('time.year').sum(dim='time')

    # Calculate the weighted annual average
    annual_mean = weighted_sum / days_per_year

    return annual_mean


def pearsonr_xr(da1, da2, dim='time'):
    """
    Return pearson correlation coefficient and p-value from correlating two xarray dataarrays
    """
    from scipy.stats import pearsonr

    def _pearsonr(a, b):
        # Mask NaNs
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) < 3:
            return np.nan, np.nan
        r, p = pearsonr(a[mask], b[mask])
        return r, p

    r, p = xr.apply_ufunc(
        _pearsonr, da1, da2,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )

    return r, p


def xr_linregress_pval(da, time_dim='year'):
    """
    Compute linear trend and p-value for an xarray DataArray over the specified time dimension.

    Parameters:
        da (xr.DataArray): DataArray with a time dimension (e.g., 'year').
        time_dim (str): Name of the time dimension (default: 'year').

    Returns:
        slope (xr.DataArray): Linear slope at each grid point
        pval (xr.DataArray): Two-sided p-value for the slope
    """
    from scipy.stats import linregress
    time_vals = da[time_dim].values

    def linreg_slope_pval(y, x):
        mask = np.isfinite(y) & np.isfinite(x)
        if np.sum(mask) < 3:
            return np.nan, np.nan
        x = x[mask]
        y = y[mask]
        x = x - np.mean(x)
        slope, _, _, pval, _ = linregress(x[mask], y[mask])
        return slope, pval

    slope, pval = xr.apply_ufunc(
        linreg_slope_pval,
        da,
        xr.DataArray(time_vals, dims=[time_dim]),
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    )

    return slope, pval


def xr_weighted_corr(x, y, *, weights=None, dims=('lat', 'lon'), eps=1e-12):
    """
    Weighted Pearson correlation across `dims`.
    - x, y: DataArray with matching dims (e.g., ('lat','lon') or ('time','lat','lon'))
    - weights: DataArray broadcastable to x,y (e.g., cell area). If None, uses cos(lat).
    - dims: dims to reduce over (the spatial dims)
    Returns a DataArray over the remaining dims (e.g., 'time' or scalar).
    """

    # build weights if not provided
    if weights is None:
        if 'lat' not in x.dims:
            raise ValueError('Provide `weights` if there is no "lat" dimension.')
        w = np.cos(np.deg2rad(x['lat']))
        w = xr.broadcast(w, x)[0]
    else:
        w = weights

    # common mask
    comm_mask = ~np.isnan(x) & ~np.isnan(y)
    w = w.where(comm_mask)
    x = x.where(comm_mask)
    y = y.where(comm_mask)

    # normalized weights sum over dims
    wsum = w.sum(dims, skipna=True)
    w = w / (wsum + eps)

    # weighted means
    mx = (w * x).sum(dims, skipna=True)
    my = (w * y).sum(dims, skipna=True)

    # centered fields
    xc = x - mx
    yc = y - my

    # weighted (co)variances
    cov = (w * xc * yc).sum(dims, skipna=True)
    vx  = (w * xc * xc).sum(dims, skipna=True)
    vy  = (w * yc * yc).sum(dims, skipna=True)

    r = cov / np.sqrt((vx + eps) * (vy + eps))
    return r
