def nan_argmax(arr):
    """
    Replicates the behavior of np.argmax but returns NaN if the slice is all NaN.

    Parameters:
    arr (numpy array): The input array to perform the argmax on.

    Returns:
    int or float: Index of the maximum value, or NaN if all values are NaN.

    Written by chatGPT
    """
    if np.all(np.isnan(arr)):
        return np.nan
    else:
        return np.nanargmax(arr)  # This returns the index of the max ignoring NaNs
