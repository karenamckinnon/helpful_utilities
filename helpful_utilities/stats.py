def fit_OLS(X, y, add_intercept=True, remove_mean=True):
    """Calculate the MLE of the OLS linear trend in data, y, given covariates, X.

    Parameters
    ----------
    X : numpy.ndarray
        Covariates/predictors
    y : numpy.ndarray
        Predictand (one-dim)
    add_intercept : bool
        Include intercept term?
    remove_mean : bool
        Remove mean of each column of X?

    Returns
    -------
    beta : numpy.array
        OLS coefficients
    yhat : numpy.array
        Estimated trend in data.
    """
    import numpy as np
    from numpy.linalg import multi_dot

    # Make y into a matrix of correct size
    n = len(y)
    ymat = np.matrix(y)

    if np.shape(ymat)[0] != n:
        ymat = ymat.T

    Xmat = np.matrix(X)
    if np.shape(Xmat)[0] != n:
        Xmat = Xmat.T

    if remove_mean:
        Xmat = Xmat.astype(float)
        Xmat -= np.mean(Xmat, axis=0)

    if add_intercept:
        Xmat = np.hstack((np.ones((n, 1)), Xmat))

    beta = (multi_dot((np.dot(Xmat.T, Xmat).I, Xmat.T, ymat)))

    yhat = np.array(np.dot(Xmat, beta)).flatten()
    beta = np.array(beta).flatten()

    return beta, yhat


def lowpass_butter(fs, lowcut, order,  data, axis=-1):
    """Perform a lowpass butterworth filter on data using a forward and backward digital filter.

    Parameters
    ----------
    fs : float
        Sampling frequency of data (example: 12 for monthly data)
    lowcut : float
        Critical frequency for Butterworth filter. See scipy docs.
    order : int
        Order of filter. Note that filtfilt doubles the original filter order.
    data : numpy array
        1D vector or 2D array to be filtered
    axis : int
        Axis along which filtering is performed.

    Returns
    -------
    data_filtered : numpy array
        Filtered data

    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')  # Coefficients for Butterworth filter
    filtered = filtfilt(b, a, data, axis=axis)

    return filtered
