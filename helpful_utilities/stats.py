"""
Functions for statistical modeling.
"""

import numpy as np
from numpy.linalg import multi_dot
from scipy.signal import butter, filtfilt


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

    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')  # Coefficients for Butterworth filter
    filtered = filtfilt(b, a, data, axis=axis)

    return filtered


def pmtm(x, dt, nw=3, cl=0.95):
    """Returns Thomson’s multitaper power spectral density (PSD) estimate, pxx, of the input signal, x.
    Slightly modified from Peter Huybers's matlab code, pmtmPH.m
    Parameters
    ----------
    x : numpy array
        Time series to analyze
    dt : float
        Time step
    nw : float
        The time-halfbandwidth product
    cl : float
        Confidence interval to calculate and display
    Returns
    -------
    P : numpy array
        PSD estimate
    s : numpy array
        Associated frequencies
    ci : numpy array
        Associated confidence interval
    """
    from scipy.signal import windows

    nfft = np.shape(x)[0]

    nx = np.shape(x)[0]
    k = min(np.round(2.*nw), nx)
    k = int(max(k-1, 1))
    s = np.arange(0, 1/dt, 1/(nfft*dt))

    # Compute the discrete prolate spheroidal sequences
    [E, V] = windows.dpss(nx, nw, k, return_ratios=True)
    E = E.T

    # Compute the windowed DFTs.
    Pk = np.abs(np.fft.fft(E*x[:, np.newaxis], nfft, axis=0))**2

    if k > 1:
        sig2 = np.dot(x[np.newaxis, :], x[:, np.newaxis])[0][0]/nx
        # initial spectrum estimate
        P = ((Pk[:, 0] + Pk[:, 1])/2)[:, np.newaxis]
        Ptemp = np.zeros((nfft, 1))
        P1 = np.zeros((nfft, 1))
        tol = .0005*sig2/nfft
        a = sig2*(1-V)

        while (np.sum(np.abs(P - P1)/nfft) > tol):
            b = np.repeat(P, k, axis=-1)/(P*V[np.newaxis, :] + np.ones((nfft, 1))*a[np.newaxis, :])
            wk = (b**2) * (np.ones((nfft, 1))*V[np.newaxis, :])
            P1 = (np.sum(wk*Pk, axis=-1)/np.sum(wk, axis=-1))[:, np.newaxis]

            Ptemp = np.empty_like(P1)
            Ptemp[:] = P1
            P1 = np.empty_like(P)
            P1[:] = P
            P = np.empty_like(Ptemp)
            P[:] = Ptemp

        # Determine equivalent degrees of freedom, see Percival and Walden 1993.
        v = ((2*np.sum((b**2)*(np.ones((nfft, 1))*V[np.newaxis, :]), axis=-1)**2) /
             np.sum((b**4)*(np.ones((nfft, 1))*V[np.newaxis, :]**2), axis=-1))

    else:
        P = np.empty_like(Pk)
        P[:] = Pk
        v = 2*np.ones((nfft, 1))

    select = (np.arange(0, (nfft + 1)/2.)).astype('int')
    P = P[select].flatten()
    s = s[select].flatten()
    v = v[select].flatten()

    # Chi-squared 95% confidence interval
    # approximation from Chamber's et al 1983; see Percival and Walden p.256, 1993
    ci = np.empty((np.shape(v)[0], 2))
    ci[:, 0] = 1./(1-2/(9*v) - 1.96*np.sqrt(2/(9*v)))**3
    ci[:, 1] = 1./(1-2/(9*v) + 1.96*np.sqrt(2/(9*v)))**3

    return P, s, ci
