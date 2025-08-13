from math import *
import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from beyond_cvar.excess_risk_over_n_srm import stagewise_rgd_cat, erm, stagewise_rgd_mom
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.neighbors import KernelDensity
from numpy import median

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
        # plt.savefig()

        fig, ax = plt.subplots()
        ax.plot(sample_sizes, var_cat, 'red', label='Cat-SRGD')
        ax.plot(sample_sizes, var_mom, 'blue', label='MoM-SRG')
        ax.plot(sample_sizes, var_erm, 'green', label='LE-GD')
        ax.legend(loc='upper right', shadow=True)
        fig.suptitle('')
        # plt.show()
        # plt.savefig()
