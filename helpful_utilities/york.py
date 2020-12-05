"""Function for fitting York, 2004, bivariate fit.

Copyright (C) 2019 Mikko Pitkanen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt


def bivariate_fit(xi, yi, dxi, dyi, ri=0.0, b0=1.0, maxIter=1e6):
    """Make a linear bivariate fit to xi, yi data using York et al. (2004).

    This is an implementation of the line fitting algorithm presented in:
    York, D et al., Unified equations for the slope, intercept, and standard
    errors of the best straight line, American Journal of Physics, 2004, 72,
    3, 367-375, doi = 10.1119/1.1632486

    See especially Section III and Table I. The enumerated steps below are
    citations to Section III

    Parameters:
      xi, yi      x and y data points
      dxi, dyi    errors for the data points xi, yi
      ri          correlation coefficient for the weights
      b0          initial guess b
      maxIter     float, maximum allowed number of iterations

    Returns:
      a           y-intercept, y = a + bx
      b           slope
      S           goodness-of-fit estimate
      sigma_a     standard error of a
      sigma_b     standard error of b

    Usage:
    [a, b] = bivariate_fit( xi, yi, dxi, dyi, ri, b0, maxIter)

    """
    # (1) Choose an approximate initial value of b
    b = b0

    # (2) Determine the weights wxi, wyi, for each point.
    wxi = 1.0 / dxi**2.0
    wyi = 1.0 / dyi**2.0

    alphai = (wxi * wyi)**0.5
    b_diff = 999.0

    # tolerance for the fit, when b changes by less than tol for two
    # consecutive iterations, fit is considered found
    tol = 1.0e-8

    # iterate until b changes less than tol
    iIter = 1
    while (abs(b_diff) >= tol) & (iIter <= maxIter):

        b_prev = b

        # (3) Use these weights wxi, wyi to evaluate Wi for each point.
        Wi = (wxi * wyi) / (wxi + b**2.0 * wyi - 2.0*b*ri*alphai)

        # (4) Use the observed points (xi ,yi) and Wi to calculate x_bar and
        # y_bar, from which Ui and Vi , and hence betai can be evaluated for
        # each point
        x_bar = np.sum(Wi * xi) / np.sum(Wi)
        y_bar = np.sum(Wi * yi) / np.sum(Wi)

        Ui = xi - x_bar
        Vi = yi - y_bar

        betai = Wi * (Ui / wyi + b*Vi / wxi - (b*Ui + Vi) * ri / alphai)

        # (5) Use Wi, Ui, Vi, and betai to calculate an improved estimate of b
        b = np.sum(Wi * betai * Vi) / np.sum(Wi * betai * Ui)

        # (6) Use the new b and repeat steps (3), (4), and (5) until successive
        # estimates of b agree within some desired tolerance tol
        b_diff = b - b_prev

        iIter += 1

    # (7) From this final value of b, together with the final x_bar and y_bar,
    # calculate a from
    a = y_bar - b * x_bar

    # Goodness of fit
    S = np.sum(Wi * (yi - b*xi - a)**2.0)

    # (8) For each point (xi, yi), calculate the adjusted values xi_adj
    xi_adj = x_bar + betai

    # (9) Use xi_adj, together with Wi, to calculate xi_adj_bar and thence ui
    xi_adj_bar = np.sum(Wi * xi_adj) / np.sum(Wi)
    ui = xi_adj - xi_adj_bar

    # (10) From Wi , xi_adj_bar and ui, calculate sigma_b, and then sigma_a
    # (the standard uncertainties of the fitted parameters)
    sigma_b = np.sqrt(1.0 / np.sum(Wi * ui**2))
    sigma_a = np.sqrt(1.0 / np.sum(Wi) + xi_adj_bar**2 * sigma_b**2)

    # calculate covariance matrix of b and a (York et al., Section II)
    cov = -xi_adj_bar * sigma_b**2
    # [[var(b), cov], [cov, var(a)]]
    cov_matrix = np.array(
        [[sigma_b**2, cov], [cov, sigma_a**2]])

    if iIter <= maxIter:
        return a, b, S, cov_matrix
    else:
        print("bivariate_fit.py exceeded maximum number of iterations, " +
              "maxIter = {:}".format(maxIter))
        return np.nan, np.nan, np.nan, np.nan


if __name__ == "__main__":
    """Test the Yorks bivariate line fitting by attempting linear fit similarly
    as in Fig. 1 in Cantrell et al, 2008, Technical Note: Review of methods for
    linear least-squares fitting of data and application to atmospheric
    chemistry problems, Atmos. Chem. Phys., doi 10.5194/acp-8-5477-2008

    In Ipython you can use this test by calling:
    run bivariate_fit.py
    """

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    # define test data set points
    x = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
    y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])

    # define weights for the data points
    wx = np.array(
        [1000.0, 1000.0, 500.0, 800.0, 200.0, 80.0, 60.0, 20.0, 1.8, 1.0])
    wy = np.array([1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0, 500.0])

    # plot error bar
    ax1.errorbar(x, y, xerr=1/(wx**0.5), yerr=1/(wy**0.5), fmt='o')

    # remember to convert weights to errors by 1/wx**0.5
    a_bivar, b_bivar, S, cov = bivariate_fit(
        x, y, 1/(wx**0.5), 1/(wy**0.5), b0=0.0)
    label_bivar = 'y={:1.2f}x+{:1.2f}'.format(b_bivar, a_bivar)

    a_ols = np.polyfit(x, y, 1)
    label_ols = 'y={:1.2f}x+{:1.2f}'.format(a_ols[0], a_ols[1])

    xlim = np.array([-0.5, 8.5])
    ylim = np.array([0, 8])

    ax_york = plt.plot(xlim, b_bivar*xlim + a_bivar,  'b-',
                       label='York et al, 2004: ' + label_bivar)

    ax_lsq = plt.plot(xlim, a_ols[0]*xlim + a_ols[1], 'r-',
                      label='Standard OLS, no weigths: ' + label_ols)

    ax1.legend()

    plt.suptitle('Cantrell et al, 2008, Fig. 1, adopted', fontsize=16)

    ax1.set_xlabel('x data')
    ax1.set_ylabel('y data')

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    ax1.grid(b=True)
    plt.show()
