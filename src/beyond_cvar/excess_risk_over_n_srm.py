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

    return theta


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
        return 2*(np.dot(w, X) - Y) * (np.dot(w, X) - Y)


# The loss gradient
def grad_loss(w, X, Y, norm):
    d = len(X)
    if norm == '1':
        return np.where(np.dot(w, X) - Y > 0, X, -X)
    elif norm == '2':
        return [(np.dot(w, X) - Y) * X[i] for i in range(d)]



# Empirical estimation of the CDF
def emp_cdf(w, X, Y, norm):
    return ECDF(loss(w, X, Y, norm))


# Kernel estimation of the density function
def kernel_estimate(w, X, Y, bandwidth, norm):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit([loss(w, X, Y, norm)])
    return kde


# New loss = The spectral risk measure
def new_loss(w, X, Y, CDF, weight_func, norm):
    return loss(w, X, Y, norm) * weight_func(CDF(loss(w, X, Y, norm)))


# Empirical estimation of the spectral risk measure
def risk_estimate(w, X, Y, weight_func, norm):
    n = len(Y)
    X_1 = X[:, 0:int(n / 2)]
    X_2 = X[:, int(n / 2):n]
    Y_1 = Y[0:int(n / 2)]
    Y_2 = Y[int(n / 2):n]

    ecdf = ECDF(loss(w, X_2, Y_2, norm))
    return np.mean(new_loss(w, X_1, Y_1, ecdf, weight_func, norm))


# Gradient for the stage-wise gradient descent (i.e for a fixed w_s)
def stagewise_grad(w, X, Y, ecdf, kde, weight_func, weight_func_derivative, norm):
    L = loss(w, X, Y, norm)
    GL = grad_loss(w, X, Y, norm)
    grad_1 = GL * weight_func(ecdf(L))
    grad_2 = GL * L * np.exp(kde.score_samples([L])) * weight_func_derivative(ecdf(L))
    return grad_1 + grad_2


"---------------------------------------Defining the different gradient descents---------------------------------------"


def emp_grad(w, X, Y, weight_func, norm):
    """
        The empirical gradient (computed by estimating the loss)

        Parameters
        ----------
        w : d-array

        X : d-n array, x-sample

        Y : n array, y-sample

        weight_func : func, the weight function that defines the spectral risk measure

        norm : string, 1 if the loss is the absolute error, 2 if it is the square error

        Returns
        -------
        grad : d-array, estimate of the gradient at w
        """
    epsilon = 10 ** (-10)
    d = len(X)
    grad = np.zeros(d)
    for i in range(d):
        base = np.zeros(d)
        base[i] += 1
        grad[i] = (risk_estimate(w + epsilon * base, X, Y, weight_func, norm) - risk_estimate(w, X, Y,
                                                                                              weight_func,
                                                                                              norm)) / epsilon
    return grad


def erm(initial_w, X, Y, eta, max_iterations, weight_func, norm):
    """
        A gradient descent (using empirical risk minimization)

        Parameters
        ----------
        initial_w : d-array, initial value of w

        X : d-n array, x-sample

        Y : n array, y-sample

        eta : float, step-size for each gradient descent

        max_iterations : int, maximum number of iterations

        weight_func : func, the weight function that defines the spectral risk measure

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
        gradient = emp_grad(wt, X, Y, weight_func, norm)
        print(gradient)
        wt = [wt[i] - eta * gradient[i] for i in range(d)]

        w_history += [wt]
        print(wt)

    return w_history


def robust_grad_estimation_cat(w, X, Y, delta, weight_func, weight_func_derivative, ws, bandwidth, norm):
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

         ws : d-array, defines the stage in the stagewise gradient descent

         bandwidth : float, for the density estimation

         norm : string, loss norm

         Returns
         -------
         float, estimate of the SRM at w
         """
    n = len(Y)
    X_1 = X[:, 0:int(n / 2)]
    X_2 = X[:, int(n / 2):n]
    Y_1 = Y[0:int(n / 2)]
    Y_2 = Y[int(n / 2):n]

    ecdf = emp_cdf(ws, X_2, Y_2, norm)
    kde = kernel_estimate(ws, X_2, Y_2, bandwidth, norm)
    new_sample = stagewise_grad(w, X_1, Y_1, ecdf, kde, weight_func, weight_func_derivative, norm)

    return catoni_d(new_sample, delta)


def stagewise_rgd_cat(initial_w, X, Y, delta, eta, max_iterations, max_gradient_descents, bandwidth, weight_func,
                      weight_func_derivative, norm):
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

        norm : string, loss norm

        Returns
        -------
        w_history : T*S-d array, list of the candidates found by the algorithm
        """

    d = len(initial_w)
    n = len(Y)

    ws = initial_w

    w_history = []

    S = max_gradient_descents
    T = max_iterations

    for s in range(S):
        w_0 = ws

        wt = w_0

        for t in range(T):
            gradient = robust_grad_estimation_cat(wt, X, Y, delta, weight_func, weight_func_derivative, ws, bandwidth,
                                                  norm)
            print(gradient)

            wt = [wt[i] - eta * gradient[i] for i in range(d)]

            w_history += [wt]
            print(wt)

        ws = wt

    return w_history


def robust_grad_estimation_mom(w, X, Y, delta, weight_func, weight_func_derivative, ws, bandwidth, norm):
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

         ws : d-array, defines the stage in the stagewise gradient descent

         bandwidth : float, for the density estimation

         norm : string, loss norm

         Returns
         -------
         float, estimate of the SRM at w
         """
    n = len(Y)
    X_1 = X[:, 0:int(n / 2)]
    X_2 = X[:, int(n / 2):n]
    Y_1 = Y[0:int(n / 2)]
    Y_2 = Y[int(n / 2):n]

    ecdf = emp_cdf(ws, X_2, Y_2, norm)
    kde = kernel_estimate(ws, X_2, Y_2, bandwidth, norm)
    new_sample = stagewise_grad(w, X_1, Y_1, ecdf, kde, weight_func, weight_func_derivative, norm)

    return MoM_d(new_sample, delta)


def stagewise_rgd_mom(initial_w, X, Y, delta, eta, max_iterations, max_gradient_descents, bandwidth, weight_func,
                      weight_func_derivative, norm):
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

        norm : string, loss norm

        Returns
        -------
        w_history : T*S-d array, list of the candidates found by the algorithm
        """

    d = len(initial_w)
    n = len(Y)

    ws = initial_w

    w_history = []

    S = max_gradient_descents
    T = max_iterations

    for s in range(S):
        w_0 = ws

        wt = w_0

        for t in range(T):
            gradient = robust_grad_estimation_mom(wt, X, Y, delta, weight_func, weight_func_derivative, ws, bandwidth,
                                                  norm)
            # print(gradient)

            wt = [wt[i] - eta * gradient[i] for i in range(d)]

            w_history += [wt]
            # print(wt)

        ws = wt

    return w_history


"---------------------------------------Parameters---------------------------------------------------------------------"

"Defining the parameters for the tests"
# Maximum number of trials
max_trials = 10 ** 4
# loss norm
norm = '1'


beta = 0
eta = 0

if norm == '2':
    beta = 0.01 / sqrt(2)  # step size for the stage wise gradient descent
    eta = 0.01 / sqrt(2)  # step size for the gradient descent (loss estimation)
elif norm == '1':
    beta = 0.1 / sqrt(2)
    eta = 0.1 / sqrt(2)


# the weight function
def phi(x):
    return 2 * np.exp(-2 * (1 - x)) / (1 - exp(-2))


# the weight function derivative
def dphi(x):
    return 4 * np.exp(-2 * (1 - x)) / (1 - exp(-2))


delta = 0.002  # parameter for the robust mean estimation procedures

sample_sizes = linspace(10, 1000, 40).astype(int)  # Size of the sample
optimal_w = [2, 3]  # Optimal candidate
initial_w = [4, 5]  # Initial candidate

# X sample
mu = [0, 0]
covariance = [[10, 0], [0, 10]]

# Maximum number of iterations
T = 100

"---------------------------------------Tests----------------------------------------------------------------------"

"Plotting the excess risk over the number of iterations"

# Distribution
distributions = ['pareto', 'normal', 'lognormal']

for norm in ['1', '2']:

    beta = 0
    eta = 0

    if norm == '2':
        beta = 0.01 / sqrt(2)  # step size for the stage wise gradient descent
        eta = 0.01 / sqrt(2)  # step size for the gradient descent (loss estimation)
    elif norm == '1':
        beta = 0.1 / sqrt(2)
        eta = 0.1 / sqrt(2)

    for dist in distributions:
        mean_erm = []
        var_erm = []
        mean_cat = []
        var_cat = []
        mean_mom = []
        var_mom = []
        for n in sample_sizes:
            X = [np.random.multivariate_normal(mu, covariance, n).T for s in range(max_trials)]
            noise = []
            if dist == 'pareto':
                noise = [(np.random.pareto(2.1, n) + 1) * 3.5 for i in range(max_trials)]
            elif dist == 'normal':
                noise = [np.random.normal(0, 2.2, n) for i in range(max_trials)]
            elif dist == 'lognormal':
                noise = [np.random.lognormal(0, 1.75, n) for i in range(max_trials)]
            Y = np.dot(optimal_w, X) + noise

            bandwidth = log(n / 2) * (n / 2) ** (-1 / 3)  # bandwidth for the kernel density estimation

            w_cat = [stagewise_rgd_cat(initial_w, X[i], Y[i], delta, beta, int(T / 10), 10, bandwidth, phi, dphi,
                                    norm)[-1] for i in range(max_trials)]
            w_emp = [erm(initial_w, X[i], Y[i], eta, T, phi, norm)[-1] for i in range(max_trials)]
            w_mom = [stagewise_rgd_mom(initial_w, X[i], Y[i], delta, beta, int(T / 10), 10, bandwidth, phi, dphi,
                                    norm)[-1] for i in range(max_trials)]

            w_cat = np.array(w_cat)
            w_emp = np.array(w_emp)
            w_mom = np.array(w_mom)

            # Creating a large sample to compute the true value of the SRM
            M = 10 ** 5
            X = np.random.multivariate_normal(mu, covariance, M).T
            if dist == 'pareto':
                noise = (np.random.pareto(2.1, M) + 1) * 3.5
            elif dist == 'normal':
                noise = np.random.normal(0, 2.2, M)
            elif dist == 'lognormal':
                noise = np.random.lognormal(0, 1.75, M)
            Y = np.dot(optimal_w, X) + noise
            minimum = risk_estimate(optimal_w, X, Y, phi, norm)

            L_1 = np.array(np.array([abs(
                risk_estimate(w_emp[s], X, Y, phi, norm) - minimum) for s in range(max_trials)]))
            L_2 = np.array(np.array([abs(
                risk_estimate(w_cat[s], X, Y, phi, norm) - minimum) for s in range(max_trials)]))
            L_3 = np.array(np.array([abs(
                risk_estimate(w_mom[s], X, Y, phi, norm) - minimum) for s in range(max_trials)]))

            mean_erm += [np.mean(L_1)]
            var_emp = [np.var(L_1)]
            mean_cat = [np.mean(L_2)]
            var_cat = [np.var(L_2)]
            mean_mom = [np.mean(L_3)]
            var_mom = [np.var(L_3)]

        fig, ax = plt.subplots()
        ax.plot(sample_sizes, mean_cat, 'red', label='Cat-SRGD')
        ax.plot(sample_sizes, mean_mom, 'blue', label='MoM-SRG')
        ax.plot(sample_sizes, mean_erm, 'green', label='LE-GD')
        ax.legend(loc='upper right', shadow=True)
        fig.suptitle('')
        # plt.show()
        plt.savefig(
            '~/Excess risk over n for l' + norm +
            'loss - ' + dist + ' (ave).pdf ')

        fig, ax = plt.subplots()
        ax.plot(sample_sizes, var_cat, 'red', label='Cat-SRGD')
        ax.plot(sample_sizes, var_mom, 'blue', label='MoM-SRG')
        ax.plot(sample_sizes, var_erm, 'green', label='LE-GD')
        ax.legend(loc='upper right', shadow=True)
        fig.suptitle('')
        # plt.show()
        plt.savefig(
            '~/Excess risk over n for l' + norm +
            'loss - ' + dist + ' (var).pdf ')
