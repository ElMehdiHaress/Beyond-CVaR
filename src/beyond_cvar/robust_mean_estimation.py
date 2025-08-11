import numpy as np
from math import *
from numpy import median


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

    for k in range(3):
        l = (X - theta) / s
        l = np.where(l >= 0, np.log(1 + l + l * l), -np.log(1 - l + l * l))
        theta = theta + (s / n) * sum(l)
    return theta


# Median of means :
def median_means(X):
    delta = 0.002
    k = 1 + 3.5*log(1/delta)
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
