# Small example written in casadi symbolic solver

import numpy as np
from casadi import *
import matplotlib.pyplot as plt

X_full = np.linspace(-3,3,100)
Y_full = np.sin(X_full) + np.random.randn(len(X_full)) * 0.01

index = np.arange(len(X_full))
index = np.random.choice(index, 3)

X = X_full[index]
Y = Y_full[index]

def kernel(x, x_, l=1, alpha=1):
    return alpha * np.exp( - (x - x_) ** 2 / (l**2))

def gram(X, kernel):
    K = np.zeros([X.shape[0], X.shape[0]])
    for i in range(X.shape[0]):
        K[i, :] = kernel(X[i], X)
    return K

def mean(x, X, Y, kernel, noise=0.):
    kappa = kernel(x, X).reshape(-1,1)
    K = gram(X, kernel)
    return kappa.T @ np.linalg.inv(K + np.eye(K.shape[0])*noise) @ Y

def var(x, X, kernel, noise):
    k = kernel(x, x)
    kappa = kernel(x, X).reshape(-1,1)
    K = gram(X, kernel)
    return k - kappa.T @ np.linalg.inv(K + np.eye(K.shape[0])*noise) @ kappa

def batch_var(x, X, kernl, noise=0.):
    l = len(x)
    res = np.zeros(l)
    for i in range(l):
        res[i] = var(x[i], X, kernel, noise)
    return res

def batch_mean(x, X, Y, kernel, noise=0.):
    l = len(x)
    res = np.zeros(l)
    for i in range(l):
        res[i] = mean(x[i], X, Y, kernel, noise)
    return res

def calc_max_entropy(X, kernel, noise=0.01):
    x = SX.sym('x', 1)
    x0 = np.random.uniform(-0,0)
    k = kernel(x, x)
    kappa = kernel(x, X)
    K = gram(X, kernel)
    variance = k - kappa.T @ inv(K + np.eye(K.shape[0])*noise) @ kappa
    entropy = 0.5 * log(2 * pi * exp(1) * variance)

    nlp = {'x':x, 'f':-entropy}
    opts = {'ipopt.print_level':0, 'print_time':0}
    S = nlpsol('S', 'ipopt', nlp, opts)
    res = S(x0=x0, lbx=-3, ubx=3)
    print("Initial: ", x0)
    x_opt = res['x']
    return x_opt

for _ in range(50):
    plt.figure(figsize=[15,10])
    t = np.linspace(-4,4,100)
    mu = batch_mean(t, X, Y, kernel, noise=.1)
    variance = batch_var(t, X, kernel, noise=.1)
    plt.plot(t, mu, label="mean prediction")
    plt.plot(t, np.sin(t), '-c', linewidth=2.5, label="Ground trueth")
    plt.plot(t, mu + variance, '-.r')
    plt.plot(t, mu - variance, '-.r', label="1 standard variance")
    plt.xlabel("input value")
    plt.ylabel("prediction")

    next_point = calc_max_entropy(X, kernel)
    for i, x in enumerate(X):
        if i == 0:
            y = Y[i]
            plt.scatter(x, y, c='b',label="data points")
            plt.scatter(next_point.full(), 0, c='r', marker='X', s=200, label="next sampling spot")
        else:
            y = Y[i]
            plt.scatter(x, y, c='b')
            plt.scatter(next_point.full(), 0, c='r', marker='X', s=200)

    # update dataset
    X = np.append(X, next_point)
    Y = np.append(Y, np.sin(next_point))
    plt.legend()
    plt.show()