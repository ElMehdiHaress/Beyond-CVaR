from math import *
import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from beyond_cvar.prediction_error_cpt import stagewise_rgd_cat, erm, stagewise_rgd_mom
from statsmodels.distributions.empirical_distribution import ECDF
try:
    from sklearn.neighbors import KernelDensity  # optional at import-time
except Exception:  # ImportError in older envs or version mismatches
    KernelDensity = None
from numpy import median


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
        # plt.savefig('/Users/omar/Desktop/Japan/CPT value/L' + norm + '_CPT_prediction_' + dist + '(ave).pdf', )

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
        # plt.savefig()
