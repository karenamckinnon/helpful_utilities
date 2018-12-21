import numpy as np
from scipy.optimize import fmin
from numpy.polynomial.polynomial import polyval


def QR(x, y, modelOrder, percentiles, xtol=1e-8, maxiter=10000):
    """Calculate the quantile regression coefficients for a given dataset.

    Modified from http://phillipmfeldman.org/Python/quantile_regression_demo.py

    Parameters
    ----------
    x : numpy array
        Predictor
    y : numpy array
        Predictand
    modelOrder : int
        Number of coefficients in model. Model order = 1 means intercept only
    percentiles : numpy array
        Percentiles [0, 1] to calculate trends for
    xtol : float
        Tolerable error when maximizing objective function. Optional.
    maxiter : int
        Maximum number of iterations to do in optimization. Optional.

    Returns
    -------
    beta_hat : numpy array
        Coefficients for each quantile
    """

    def _tilted_abs(rho, x):
        return x * (rho - (x < 0))

    def _model(x, beta):

        return polyval(x, beta)

    def _objective(beta, rho):
        return _tilted_abs(rho, y - _model(x, beta)).sum()

    # Runs QR on data (x, y) using a model of order modelOrder. Model Order 2 = linear.
    # Percentiles should be fractions, i.e. between 0 and 1

    # Define starting point for optimization:
    beta_0 = np.zeros(modelOrder)

    if modelOrder >= 2:
        beta_0[1] = 1.0

    # `beta_hat[i]` will store the parameter estimates for the quantile
    # corresponding to `percentile[i]`:

    beta_hat = []

    for i, percentile in enumerate(percentiles):
        beta_hat.append(fmin(_objective, x0=beta_0, args=(percentile,), xtol=xtol,
                        disp=False, maxiter=maxiter))

    return np.array(beta_hat)
