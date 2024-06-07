"""
Tools for common data preprocessing steps
"""
import numpy as np
from helpful_utilities.stats import lowpass_butter


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
