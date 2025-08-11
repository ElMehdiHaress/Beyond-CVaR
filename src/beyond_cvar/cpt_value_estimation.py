import numpy as np
from math import *
from matplotlib import pyplot as plt
from numpy import linspace
from statsmodels.distributions.empirical_distribution import ECDF
from numpy import median

"------------------------------------Some beforehand function-----------------------------------------"


# empirical mean :
def empirical(X):
    return np.mean(X)


# Catoni estimator :
def catoni(X):
    n = len(X)
    delta = 0.002
    var = np.var(X)
    if var == 0:
        var = 10 ** (-5)

    s = sqrt(var * n / log(2 / delta))  # scaling parameter
    theta = 0

    for k in range(4):
        l = (X - theta) / s
        l = np.where(l >= 0, np.log(1 + l + l * l), -np.log(1 - l + l * l))
        theta = theta + (s / n) * sum(l)
    return theta


# Median of means :
def median_means(X):
    delta = 0.002
    k = 1 + 3.5 * log(1 / delta)
    n = len(X)
    L = []
    for i in range(1, int(k)):
        L = L + [np.mean(X[(i - 1) * int(n / k):i * int(n / k)])]
    return median(L)


# Trimmed mean :
def trimmed_mean(X):
    delta = 0.002
    n = len(X)
    X1 = X[0:int(n / 2)]
    X2 = X[int(n / 2):n + 1]
    epsilon = 12 * log(4 / delta) / (n / 2)

    X2.sort()
    alpha = X2[int(epsilon * (n / 2))]
    beta = X2[int((1 - epsilon) * (n / 2))]

    X1 = np.where(X1 <= beta, X1, beta)
    X1 = np.where(X1 >= alpha, X1, alpha)
    return np.mean(X1)


# Random truncation_with empirical u
def random_trunc(X):
    u = np.var(X) + np.mean(X) ** 2
    delta = 0.002
    n = len(X)
    B = [sqrt(u * i / log(1 / delta)) for i in range(n)]
    X = np.where(X <= B, X, 0)
    return np.mean(X)


# Defining the utility functions:
def utility(x):
    A = []
    for a in x:
        if a > 0:
            A.append(a ** 0.88)
        else:
            A.append(0)
    return np.array(A)


def utility_(x):
    A = []
    for a in x:
        if a < 0:
            A.append(-2.25 * ((-a) ** 0.88))
        else:
            A.append(0)
    return np.array(A)


# Defining the weight functions: - Only 'weight_' is used but both of them can be used -
a = 2.1 * pi / 4
b = - pi / 4
c = 0.05
d = 20


def weight(p):
    return (np.tan(a * p + b) - tan(b)) / (tan(a + b) - tan(b))


def weight_(p):
    return (np.sin(2 * pi * (p + c)) + d * p - sin(2 * pi * c)) / (sin(2 * pi * (1 + c)) + d - sin(2 * pi * c))


# Defining the derivatives of the weight functions since they are needed for the estimation:
def derivative_weight(p):
    return (a / (tan(a + b) - tan(b))) * (1 + np.tan(a * p + b) * np.tan(a * p + b))


def derivative_weight_(p):
    return (2 * pi * (np.cos(2 * pi * (p + c))) + d) / (np.sin(2 * pi * (1 + c)) + d - sin(2 * pi * c))


# Estimating the cumulative distribution function empirically from a sample 'sample' and evaluating it on another
# sample 'X':
def ecdf(X, Y):
    return np.array([np.mean(np.where(X < y, 1, 0)) for y in Y])


# The CPT value estimator is calculated from 2 samples by using one to estimate the CDF and using the other for a
# robust mean estimation:
def cpt_estimate(sample1, sample2, method):
    estimator = diction[method]
    CDF = ECDF(utility(sample2))
    CDF_ = ECDF(utility_(sample2))
    return estimator(utility(sample1) * derivative_weight_(1 - CDF(utility(sample1)))) - \
           estimator(utility_(sample1) * derivative_weight_(1 - CDF_(utility_(sample1))))


# The parshant estimator is computed by estimating the quantiles:
def empirical_parshant(sample1, sample2):
    sample = np.array(list(sample1) + list(sample2))
    n = len(sample)
    L1 = np.array([(n + 1 - i) / n for i in range(1, n)])
    L2 = np.array([(n - i) / n for i in range(1, n)])
    L1_ = np.array([i / n for i in range(1, n)])
    L2_ = np.array([(i - 1) / n for i in range(1, n)])
    W = weight_(L1) - weight_(L2)
    W_ = weight_(L1_) - weight_(L2_)
    # W = L1 - L2
    # W_ = L1_ - L2_
    sample.sort()
    U = utility(sample)
    U_ = utility_(sample)
    return sum(U[0:n - 1] * W) - sum(U_[0:n - 1] * W_)


# Creating a dictionary of the estimators :
diction = dict(catoni=catoni, empirical=empirical, median_means=median_means, trimmed_mean=trimmed_mean,
               random_trunc=random_trunc)
# Creating a dictionary of the distributions:
dic = dict(normal=np.random.normal, lognormal=np.random.lognormal, pareto=np.random.pareto)


# Defining a function that gives result for one setting (distribution and estimation method fixed):
def cpt_meanvariance(dist_param, samples_size, trials, method, dist):
    """
        this function returns an array containing the evolution of the mean&variance with respect to the sample size
        -when both the distribution and the estimation method is fixed- over many trials

        Parameters
        ----------
        dist_param : 1d-array
            containing the different parameters of the distribution

        samples_size : 1d-array
            containing the different sample sizes we want to consider

        trials : float
           number of trials

        method : string
            the estimation method

        dist : string
            the distribution, here we consider only normal, lognormal and pareto distributions

        Returns
        -------
        (number of samples)-2 array
        """

    # Calculating the true CPT_value
    if dist == 'pareto':
        large_sample1 = (dic[dist](dist_param[1], 100000) + 1) * dist_param[0]
        large_sample2 = (dic[dist](dist_param[1], 100000) + 1) * dist_param[0]
    else:
        large_sample1 = dic[dist](dist_param[0], dist_param[1], 100000)
        large_sample2 = dic[dist](dist_param[0], dist_param[1], 100000)

    True_S = cpt_estimate(large_sample1, large_sample2, 'empirical')

    CPT = []

    # For the parshant method, the estimation is different
    if method == 'prashanth':
        for sample_size in samples_size:
            if dist == 'pareto':
                cpt = [abs(empirical_parshant((dic[dist](dist_param[1], sample_size) + 1) * dist_param[0],
                                              (dic[dist](dist_param[1], sample_size) + 1) * dist_param[0]) -
                           True_S) for i in range(trials)]
            else:
                cpt = [abs(empirical_parshant(dic[dist](dist_param[0], dist_param[1], sample_size),
                                              dic[dist](dist_param[0], dist_param[1], sample_size)) - True_S)
                       for i in range(trials)]
            CPT.append([np.mean(cpt), np.var(cpt)])
            print(sample_size, method)
        return CPT

    # For the other methods, we use the function cpt_estimate defined above
    for sample_size in samples_size:
        if dist == 'pareto':
            cpt = [abs(cpt_estimate((dic[dist](dist_param[1], sample_size) + 1) * dist_param[0],
                                    (dic[dist](dist_param[1], sample_size) + 1) * dist_param[0],
                                    method) - True_S) for i in range(trials)]
        else:
            cpt = [abs(cpt_estimate(dic[dist](dist_param[0], dist_param[1], sample_size),
                                    dic[dist](dist_param[0], dist_param[1], sample_size),
                                    method) - True_S) for i in range(trials)]

        CPT.append([np.mean(cpt), np.var(cpt)])

        print(sample_size, method)

    return CPT


# Setting the sample sizes we are interested in:
min_sample_size = 100
max_sample_size = 10000
numberofsamples = 10
samples_size = linspace(min_sample_size, max_sample_size, numberofsamples).astype(int)
trials = 500


# Setting the distribution parameters we are interested in. In each setting, the distributions have the same variance:
def dist_parameters(setting, distribution):
    if distribution == 'normal' and setting == '1':
        return [0, 45]
    elif distribution == 'lognormal' and setting == '1':
        return [0, 2]
    elif distribution == 'pareto' and setting == '1':
        return [12, 2.1]  # [m, a]


DIST = ['normal', 'lognormal', 'pareto']
METHODS = ['empirical', 'catoni', 'median_means', 'random_trunc', 'prashanth']

# For each setting and each distribution, we plot the evolution of the mean&variance with respect to the sample size
# over many trials and for all the estimation methods
i = 1
for dist in DIST:
    dist_param = dist_parameters(str(i), dist)
    S = []
    for method in METHODS:
        print(method)
        S += [cpt_meanvariance(dist_param, samples_size, trials, method, dist)]

    print(dist)

    fig, ax = plt.subplots()
    if dist == 'normal':
        for p in range(len(S)):
            ax.plot(samples_size, np.array(S[p])[:, 0], label=METHODS[p])
        ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        fig.suptitle('Deviation (Ave)', fontsize=16)
    else:
        for p in range(len(S)):
            ax.plot(samples_size, np.array(S[p])[:, 0])
    # fig.suptitle('Deviation' + dist + str(i), fontsize=16)
    plt.show()
    plt.savefig('/Users/omar/Desktop/Japan/CPT value/Deviation' + dist + str(i) + '.pdf', )

    fig, ax = plt.subplots()
    for p in range(len(S)):
        ax.plot(samples_size, np.array(S[p])[:, 1])
    if dist == 'normal':
        # ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        fig.suptitle('Deviations (Var)', fontsize=16)
    plt.show()
    plt.savefig('/Users/omar/Desktop/Japan/CPT value/Variance' + dist + str(i) + '.pdf', )
