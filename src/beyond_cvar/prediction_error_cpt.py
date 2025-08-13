from math import *
import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.neighbors import KernelDensity
from numpy import median

"------------------------------------------Some beforehand functions---------------------------------------------------"


# Mean estimation methods for real variables and variables in R^d:
def catoni_d(sample, delta):
    n = len(sample[0])
    d = len(sample)

    ll = sample

    variance = np.array([np.var(sample[k]) for k in range(d)])
    variance = np.where(variance > 0, variance, 0.00001)
    s = np.array([sqrt(variance[k] * n / log(2 / delta)) for k in range(d)])
    # s = np.where(s > 0, s, 0.00001)

    theta = np.zeros(d)
    for i in range(5):
        xx = [(ll[k] - theta[k]) / s[k] for k in range(d)]
        xx = [np.where(xx[k] >= 0, np.log(1 + xx[k] + xx[k] * xx[k]), -np.log(1 - xx[k] + xx[k] * xx[k])) for k in
              range(d)]
        theta = [theta[k] + (s[k] / n) * sum(xx[k]) for k in range(d)]

    return np.array(theta)


def catoni_1(X, delta):
    n = len(X)
    var = np.var(X)
    if var == 0:
        var = 10 ** (-5)

    s = sqrt(var * n / log(2 / delta))  # scaling parameter
    theta = 0

    for k in range(3):
        l = (X - theta) / s
        l = np.where(l >= 0, np.log(1 + l + l * l), -np.log(1 - l + l * l))
        theta = theta + (s / n) * sum(l)
    return theta


def MoM_d(sample, delta):
    k = 1 + 3.5 * log(1 / delta)
    n = len(sample[0])
    d = len(sample)
    M = np.zeros(d)
    for s in range(d):
        L = []
        for i in range(1, int(k)):
            L = L + [np.mean(sample[:, (i - 1) * int(n / k):i * int(n / k)][s])]
        M[s] += median(L)
    return M


def MoM_1(X, delta):
    k = 1 + 3.5 * log(1 / delta)
    n = len(X)
    L = []
    for i in range(1, int(k)):
        L = L + [np.mean(X[(i - 1) * int(n / k):i * int(n / k)])]
    return median(L)


# The linear regression loss
def loss(w, X, Y, norm):
    if norm == '1':
        return abs(np.dot(w, X) - Y)
    elif norm == '2':
        return 2 * (np.dot(w, X) - Y) * (np.dot(w, X) - Y)


# The loss gradient
def grad_loss(w, X, Y, norm):
    d = len(X)
    if norm == '1':
        return np.where(np.dot(w, X) - Y > 0, X, -X)
    elif norm == '2':
        return [(np.dot(w, X) - Y) * X[i] for i in range(d)]


# Empirical estimation of the CDF
def emp_cdf(w, X, Y, utility, norm):
    return ECDF(utility(loss(w, X, Y, norm)))


# Kernel estimation of the density function
def kernel_estimate(w, X, Y, bandwidth, utility, norm):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit([utility(loss(w, X, Y, norm))])
    return kde


# New loss = The spectral risk measure
def new_loss(w, X, Y, CDF, weight_func, utility, norm):
    return utility(loss(w, X, Y, norm)) * weight_func(CDF(utility(loss(w, X, Y, norm))))


# Empirical estimation of the spectral risk measure
def risk_estimate(w, X, Y, weight_func, utility, norm):
    n = len(Y)
    X_1 = X[:, 0:int(n / 2)]
    X_2 = X[:, int(n / 2):n]
    Y_1 = Y[0:int(n / 2)]
    Y_2 = Y[int(n / 2):n]

    ecdf = ECDF(utility(loss(w, X_2, Y_2, norm)))
    return np.mean(new_loss(w, X_1, Y_1, ecdf, weight_func, utility, norm))


# Gradient for the stage-wise gradient descent (i.e for a fixed w_s)
def stagewise_grad(w, X, Y, ecdf, kde, weight_func, weight_func_derivative, utility, utility_deriv, norm):
    L = utility(loss(w, X, Y, norm))
    GL = grad_loss(w, X, Y, norm) * utility_deriv(L)
    grad_1 = GL * weight_func(ecdf(L))
    grad_2 = GL * L * np.exp(kde.score_samples([L])) * weight_func_derivative(ecdf(L))
    return grad_1 + grad_2


"---------------------------------------Defining the different gradient descents---------------------------------------"


def emp_grad(w, X, Y, weight_func, utility, norm):
    """
        The empirical gradient (computed by estimating the loss)

        Parameters
        ----------
        w : d-array

        X : d-n array, x-sample

        Y : n array, y-sample

        weight_func : func, weight function

        utility : func, utility function

        norm : string, 1 if the loss is the absolute error, 2 if it is the square error

        Returns
        -------
        grad : d-array, estimate of the gradient at w
        """
    epsilon = 10 ** (-12)
    d = len(X)
    grad = np.zeros(d)
    for i in range(d):
        base = np.zeros(d)
        base[i] += 1
        grad[i] = (risk_estimate(w + epsilon * base, X, Y, weight_func, utility, norm) -
                   risk_estimate(w, X, Y, weight_func, utility, norm)) / epsilon
    return grad


def erm(initial_w, X, Y, eta, max_iterations, weight_func, weight_func_, utility, utility_, norm):
    """
        A gradient descent (using empirical risk minimization)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        X : d-n array, x-sample

        Y : n array, y-sample

        eta : float, step-size for each gradient descent

        max_iterations : int, maximum number of iterations

        weight_func : func,  weight function

        weight_func_ : func, weight function

        utility : func, utility function

        utility_ : func, utility function

        norm : string, 1 if the loss is the absolute error, 2 if it is the square error

        Returns
        -------
        w_history : max_iterations*d-array, list of the candidates found by the algorithm

        """

    d = len(initial_w)
    w0 = initial_w

    w_history = []
    T = max_iterations

    wt = w0
    for t in range(T):
        gradient = emp_grad(wt, X, Y, weight_func, utility, norm) - emp_grad(wt, X, Y, weight_func_, utility_, norm)
        print(gradient)
        wt = [wt[i] - eta * gradient[i] for i in range(d)]

        w_history += [wt]
        print(wt)

    return w_history


def robust_grad_estimation_cat(w, X, Y, delta, weight_func, weight_func_derivative, utility, utility_deriv, ws,
                               bandwidth, norm):
    """
         Robust estimate of the gradient of the SRM (using the Cat estimator)

         Parameters
         ----------
         w : d-array, candidate

         X : d-n array, x sample

         Y : n-array, y sample

         delta : float, 0.002

         weight_func : func, weight function

         weight_func_derivative : func, derivative of the weight function

         utility : func, utility function

         utility_deriv : func, derivative of the utility function

         ws : d-array, defines the stage in the stagewise gradient descent

         bandwidth : float, for the density estimation

         norm : string, loss norm

         Returns
         -------
         float, estimate of the SRM at w
         """
    nn = len(Y)
    X_1 = X[:, 0:int(nn / 2)]
    X_2 = X[:, int(nn / 2):nn]
    Y_1 = Y[0:int(nn / 2)]
    Y_2 = Y[int(nn / 2):nn]

    ecdf = emp_cdf(ws, X_2, Y_2, utility, norm)
    kde = kernel_estimate(ws, X_2, Y_2, bandwidth, utility, norm)
    new_sample = stagewise_grad(w, X_1, Y_1, ecdf, kde, weight_func, weight_func_derivative, utility, utility_deriv,
                                norm)

    return catoni_d(new_sample, delta)


def stagewise_rgd_cat(initial_w, X, Y, delta, eta, max_iterations, max_gradient_descents, bandwidth, weight_func,
                      weight_func_derivative, weight_func_, weight_func_derivative_, utility, utility_deriv,
                      utility_, utility_deriv_, norm):
    """
        A stage-wise robust gradient descent (uses the Cat estimator for the gradient)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        X : d-n array, x-sample

        Y : n array, y-sample

        delta : float

        eta : float, step-size

        max_iterations : int, maximum number of iterations for each gradient descent

        max_gradient_descents : int, maximum number of gradient descents

        bandwidth : float, for the density estimation

        weight_func : func, weight function

        weight_func_derivative : func, derivative of the weight function

        weight_func_ : func, weight function

        weight_func_derivative_ : func, derivative of the weight function

        utility : func, utility function

        utility_deriv : func, derivative of the utility function

        utility_ : func, utility function

        utility_deriv_ : func, derivative of the utility function

        norm : string, loss norm

        Returns
        -------
        w_history : T*S-d array, list of the candidates found by the algorithm
        """

    d = len(initial_w)

    ws = initial_w

    w_history = []

    S = max_gradient_descents
    T = max_iterations

    for s in range(S):
        w_0 = ws

        wt = w_0

        for t in range(T):
            gradient = robust_grad_estimation_cat(wt, X, Y, delta, weight_func, weight_func_derivative, utility,
                                                  utility_deriv, ws, bandwidth, norm) - \
                       robust_grad_estimation_cat(wt, X, Y, delta, weight_func_, weight_func_derivative_,
                                                  utility_, utility_deriv_, ws, bandwidth, norm)
            print(gradient)

            wt = [wt[i] - eta * gradient[i] for i in range(d)]

            w_history += [wt]
            print(wt)

        ws = wt

    return w_history


def robust_grad_estimation_mom(w, X, Y, delta, weight_func, weight_func_derivative, utility, utility_deriv,
                               ws, bandwidth, norm):
    """
         Robust estimate of the gradient of the SRM (using the MoM estimator)

         Parameters
         ----------
         w : d-array, candidate

         X : d-n array, x sample

         Y : n-array, y sample

         delta : float, 0.002

         weight_func : func, weight function

         weight_func_derivative : func, derivative of the weight function

         utility : func, utility function

         utility_deriv : func, derivative of the utility function

         ws : d-array, defines the stage in the stagewise gradient descent

         bandwidth : float, for the density estimation

         norm : string, loss norm

         Returns
         -------
         float, estimate of the SRM at w
         """
    nn = len(Y)
    X_1 = X[:, 0:int(nn / 2)]
    X_2 = X[:, int(nn / 2):nn]
    Y_1 = Y[0:int(nn / 2)]
    Y_2 = Y[int(nn / 2):nn]

    ecdf = emp_cdf(ws, X_2, Y_2, utility, norm)
    kde = kernel_estimate(ws, X_2, Y_2, bandwidth, utility, norm)
    new_sample = stagewise_grad(w, X_1, Y_1, ecdf, kde, weight_func, weight_func_derivative, utility, utility_deriv,
                                norm)

    return MoM_d(new_sample, delta)


def stagewise_rgd_mom(initial_w, X, Y, delta, eta, max_iterations, max_gradient_descents, bandwidth, weight_func,
                      weight_func_derivative, weight_func_, weight_func_derivative_, utility,
                      utility_deriv, utility_, utility_deriv_, norm):
    """
        A stage-wise robust gradient descent (uses the MoM estimator for the gradient)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        X : d-n array, x-sample

        Y : n array, y-sample

        delta : float

        eta : float, step-size

        max_iterations : int, maximum number of iterations for each gradient descent

        max_gradient_descents : int, maximum number of gradient descents

        bandwidth : float, for the density estimation

        weight_func : func, weight function

        weight_func_derivative : func, derivative of the weight function

        weight_func_ : func, weight function

        weight_func_derivative_ : func, derivative of the weight function

        utility : func, utility function

        utility_deriv : func, derivative of the utility function

        utility_ : func, utility function

        utility_deriv_ : func, derivative of the utility function

        norm : string, loss norm

        Returns
        -------
        w_history : T*S-d array, list of the candidates found by the algorithm
        """

    d = len(initial_w)

    ws = initial_w

    w_history = []

    S = max_gradient_descents
    T = max_iterations

    for s in range(S):
        w_0 = ws

        wt = w_0

        for t in range(T):
            gradient = robust_grad_estimation_mom(wt, X, Y, delta, weight_func, weight_func_derivative,
                                                  utility, utility_deriv, ws, bandwidth, norm) - \
                       robust_grad_estimation_mom(wt, X, Y, delta, weight_func_, weight_func_derivative_,
                                                  utility_, utility_deriv_, ws, bandwidth, norm)
            # print(gradient)

            wt = [wt[i] - eta * gradient[i] for i in range(d)]

            w_history += [wt]
            # print(wt)

        ws = wt

    return w_history

