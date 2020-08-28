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


"---------------------------------------Parameters---------------------------------------------------------------------"

"Defining the parameters for the tests"

max_trials = 500  # number of trials
# loss norm
norm = '2'

beta = 0
eta = 0

if norm == '2':
    beta = 0.01 / sqrt(2)  # step size for the stage wise gradient descent
    eta = 0.01 / sqrt(2)  # step size for the gradient descent (loss estimation)
elif norm == '1':
    beta = 0.1 / sqrt(2)
    eta = 0.1 / sqrt(2)

c = 0.05
d = 20


# the weight functions and their derivatives
def tilde_omega(x):
    return (2 * pi * (np.cos(2 * pi * (1 - x + c))) + d) / (np.sin(2 * pi * (1 + c)) + d - sin(2 * pi * c))


def tilde_omega_deriv(x):
    return (4 * (pi ** 2) * np.sin(2 * pi * (1 - x + c))) / (np.sin(2 * pi * (1 + c)) + d - sin(2 * pi * c))


def tilde_omega_(x):
    return (2 * pi * (np.cos(2 * pi * (x + c))) + d) / (np.sin(2 * pi * (1 + c)) + d - sin(2 * pi * c))


def tilde_omega_deriv_(x):
    return -4 * (pi ** 2) * (np.sin(2 * pi * (x + c))) / (np.sin(2 * pi * (1 + c)) + d - sin(2 * pi * c))


# the utility functions and their derivatives
def util(x):
    return np.where(x > 0, np.log(1 + x), 0)


def util_deriv(x):
    A = []
    for a in x:
        if a > 0:
            A.append(1 / (1 + a))
        else:
            A.append(0)
    return np.array(A)


def util_(x):
    return np.where(x < 0, -np.log(1 - x) + x, 0)


def util_deriv_(x):
    A = []
    for a in x:
        if a < 0:
            A.append(1 / (1 - a))
        else:
            A.append(0)
    return np.array(A)


delta = 0.002  # parameter for the robust mean estimation procedures

max_iterations = 10  # number of iterations for the stagewise gd
max_gradient_descents = 10  # number of gradient descents
iterations = 100  # number of iterations for the empirical minimization

sample_size = 500  # sample size

optimal_w = [2, 3]  # optimal candidate
initial_w = [4, 5]  # initial candidate

bandwidth = log(sample_size / 2) * (sample_size / 2) ** (-1 / 3)  # bandwidth for the kernel density estimation

# List of the sample sizes for the prediction
min_sample = 100
max_sample = 10000
samples = 10
prediction = linspace(min_sample, max_sample, samples).astype(int)

"---------------------------------------Tests-------------------------------------------------------------------------"

"Plotting the prediction error with respect to the sample size"

# Distribution
distributions = ['lognormal', 'normal', 'pareto']

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
        # X sample
        mu = [0, 0]
        covariance = [[10, 0], [0, 10]]
        X = np.random.multivariate_normal(mu, covariance, sample_size).T

        # Defining the noise
        noise = []
        if dist == 'pareto':
            noise = (np.random.pareto(2.1, sample_size) + 1) * 12
        elif dist == 'normal':
            noise = np.random.normal(0, 45, sample_size)
        elif dist == 'lognormal':
            noise = np.random.lognormal(0, 2, sample_size)
        noise = noise - np.mean(noise)

        # Training the model
        Y = np.dot(optimal_w, X) + noise

        rgd_cat = \
            stagewise_rgd_cat(initial_w, X, Y, delta, beta, max_iterations, max_gradient_descents, bandwidth,
                              tilde_omega, tilde_omega_deriv, tilde_omega_, tilde_omega_deriv_,
                              util, util_deriv, util_, util_deriv_, norm)[-1]
        empirical_minimization = erm(initial_w, X, Y, eta, iterations, tilde_omega, tilde_omega_, util, util_, norm)[-1]
        rgd_mom = \
            stagewise_rgd_mom(initial_w, X, Y, delta, beta, max_iterations, max_gradient_descents, bandwidth,
                              tilde_omega, tilde_omega_deriv, tilde_omega_, tilde_omega_deriv_,
                              util, util_deriv, util_, util_deriv_, norm)[-1]
        print(rgd_cat, rgd_mom, empirical_minimization)

        # Computing the prediction error over many trials with respect to the sample size
        mean_rgd_cat = []
        mean_rgd_mom = []
        mean_erm = []
        var_rgd_cat = []
        var_rgd_mom = []
        var_erm = []
        for n in prediction:
            X = [np.random.multivariate_normal(mu, covariance, n).T for i in range(max_trials)]
            if dist == 'pareto':
                noise = [(np.random.pareto(2.1, n) + 1) * 12 for i in range(max_trials)]
            elif dist == 'normal':
                noise = [np.random.normal(0, 45, n) for i in range(max_trials)]
            elif dist == 'lognormal':
                noise = [np.random.lognormal(0, 2, n) for i in range(max_trials)]
            noise = [noise[i] - np.mean(noise[i]) for i in range(max_trials)]

            Y = np.dot(optimal_w, X) + noise
            if norm == '2':
                mean_rgd_cat += [
                    sum([np.mean((np.dot(rgd_cat, X[i]) - Y[i]) ** 2) for i in range(max_trials)]) / max_trials]
                mean_rgd_mom += [
                    sum([np.mean((np.dot(rgd_mom, X[i]) - Y[i]) ** 2) for i in range(max_trials)]) / max_trials]
                mean_erm += [
                    sum([np.mean((np.dot(empirical_minimization, X[i]) - Y[i]) ** 2) for i in range(max_trials)])
                    / max_trials]

                var_rgd_cat += [
                    np.var(np.array([np.mean((np.dot(rgd_cat, X[i]) - Y[i]) ** 2) for i in range(max_trials)]))]
                var_rgd_mom += [
                    np.var(np.array([np.mean((np.dot(rgd_mom, X[i]) - Y[i]) ** 2) for i in range(max_trials)]))]
                var_erm += [
                    np.var(
                        np.array(
                            [np.mean((np.dot(empirical_minimization, X[i]) - Y[i]) ** 2) for i in range(max_trials)]))]
            if norm == '1':
                mean_rgd_cat += [
                    sum([np.mean(abs(np.dot(rgd_cat, X[i]) - Y[i])) for i in range(max_trials)]) / max_trials]
                mean_rgd_mom += [
                    sum([np.mean(abs(np.dot(rgd_mom, X[i]) - Y[i])) for i in range(max_trials)]) / max_trials]
                mean_erm += [sum([np.mean(abs(np.dot(empirical_minimization, X[i]) - Y[i])) for i in range(max_trials)])
                             / max_trials]

                var_rgd_cat += [
                    np.var(np.array([np.mean(abs(np.dot(rgd_cat, X[i]) - Y[i])) for i in range(max_trials)]))]
                var_rgd_mom += [
                    np.var(np.array([np.mean(abs(np.dot(rgd_mom, X[i]) - Y[i])) for i in range(max_trials)]))]
                var_erm += [
                    np.var(
                        np.array(
                            [np.mean(abs(np.dot(empirical_minimization, X[i]) - Y[i])) for i in range(max_trials)]))]

            print(n)

        fig, ax = plt.subplots()
        if dist == 'normal':
            ax.plot(prediction, mean_rgd_cat, 'red', label='Cat-SRGD')
            ax.plot(prediction, mean_rgd_mom, 'blue', label='MoM-SRGD')
            ax.plot(prediction, mean_erm, 'green', label='LE-GD')
            ax.legend(loc='upper right', shadow=True)
            fig.suptitle('Prediction error (ave)')
        else:
            ax.plot(prediction, mean_rgd_cat, 'red')
            ax.plot(prediction, mean_rgd_mom, 'blue')
            ax.plot(prediction, mean_erm, 'green')
            # ax.legend(loc='upper right', shadow=True)
        # plt.show()
        plt.savefig('/Users/omar/Desktop/Japan/CPT value/L' + norm + '_CPT_prediction_' + dist + '(ave).pdf', )

        fig, ax = plt.subplots()
        if dist == 'normal':
            fig.suptitle('Prediction error (var)')
            ax.plot(prediction, var_rgd_cat, 'red', label='Cat-SRGD')
            ax.plot(prediction, var_rgd_mom, 'blue', label='MoM-SRGD')
            ax.plot(prediction, var_erm, 'green', label='LE-GD')
            ax.legend(loc='upper right', shadow=True)
        else:
            ax.plot(prediction, var_rgd_cat, 'red')
            ax.plot(prediction, var_rgd_mom, 'blue')
            ax.plot(prediction, var_erm, 'green')
        # plt.show()
        plt.savefig('/Users/omar/Desktop/Japan/CPT value/L' + norm + '_CPT_prediction_' + dist + '(var).pdf', )
