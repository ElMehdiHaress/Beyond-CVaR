
from math import *
import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace
from beyond_cvar.excess_risk_over_t_srm import stagewise_rgd_cat, erm, stagewise_rgd_mom
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.neighbors import KernelDensity
from numpy import median


"---------------------------------------Parameters---------------------------------------------------------------------"

"Defining the parameters for the tests"

# Maximum number of trials
max_trials = 10 ** 4
# loss norm

# the weight function
def phi(x):
    return 2 * np.exp(-2 * (1 - x)) / (1 - exp(-2))


# the weight function derivative
def dphi(x):
    return 4 * np.exp(-2 * (1 - x)) / (1 - exp(-2))


delta = 0.002  # parameter for the robust mean estimation procedures

sample_size = 500  # Size of the sample
optimal_w = [2, 3]  # Optimal candidate
initial_w = [4, 5]  # Initial candidate

bandwidth = log(sample_size / 2) * (sample_size / 2) ** (-1 / 3)  # bandwidht for the kernel density estimation

# X sample
mu = [0, 0]
covariance = [[10, 0], [0, 10]]

# Maximum number of iterations
T = 500

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
        # Filling W with the three candidates given by the three algorithms as a function of the number of iterations
        W = []
        initial_w_cat = initial_w
        initial_w_mom = initial_w
        initial_w_emp = initial_w
        t = 0

        N = sample_size

        X = [np.random.multivariate_normal(mu, covariance, N).T for s in range(max_trials)]
        noise = []
        if dist == 'pareto':
            noise = [(np.random.pareto(2.1, N) + 1) * 3.5 for i in range(max_trials)]
        elif dist == 'normal':
            noise = [np.random.normal(0, 2.2, N) for i in range(max_trials)]
        elif dist == 'lognormal':
            noise = [np.random.lognormal(0, 1.75, N) for i in range(max_trials)]
        Y = np.dot(optimal_w, X) + noise

        w_cat = [stagewise_rgd_cat(initial_w, X[i], Y[i], delta, beta, int(T / 10), 10, bandwidth, phi, dphi,
                               norm) for i in range(max_trials)]
        w_emp = [erm(initial_w, X[i], Y[i], eta, T, phi, norm) for i in range(max_trials)]
        w_mom = [stagewise_rgd_mom(initial_w, X[i], Y[i], delta, beta, int(T / 10), 10, bandwidth, phi, dphi,
                               norm) for i in range(max_trials)]

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

        L_1 = np.array([[np.array([abs(
            risk_estimate(w_emp[s][t], X, Y, phi, norm) - minimum) for s in range(max_trials)])] for t in
            range(T)])
        L_2 = np.array([[np.array([abs(
            risk_estimate(w_cat[s][t], X, Y, phi, norm) - minimum) for s in range(max_trials)])] for t in
            range(T)])
        L_3 = np.array([[np.array([abs(
            risk_estimate(w_mom[s][t], X, Y, phi, norm) - minimum) for s in range(max_trials)])] for t in
            range(T)])

        mean_erm = [np.mean(L_1[i]) for i in range(T)]
        var_emp = [np.var(L_1[i]) for i in range(T)]
        mean_cat = [np.mean(L_2[i]) for i in range(T)]
        var_cat = [np.var(L_2[i]) for i in range(T)]
        mean_mom = [np.mean(L_3[i]) for i in range(T)]
        var_mom = [np.var(L_3[i]) for i in range(T)]

        fig, ax = plt.subplots()
        ax.plot([i for i in range(50, T)], mean_cat[50:T], 'red', label='Cat-SRGD')
        ax.plot([i for i in range(50, T)], mean_mom[50:T], 'blue', label='MoM-SRG')
        ax.plot([i for i in range(50, T)], mean_erm[50:T], 'green', label='LE-GD')
        ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        fig.suptitle('')
        # plt.show()
        # plt.savefig()

        fig, ax = plt.subplots()
        ax.plot([i for i in range(50, T)], var_cat[50:T], 'red', label='Cat-SRGD')
        ax.plot([i for i in range(50, T)], var_mom[50:T], 'blue', label='MoM-SRG')
        ax.plot([i for i in range(50, T)], var_emp[50:T], 'green', label='LE-GD')
        ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        fig.suptitle('')
        # plt.show()
        # plt.savefig()
