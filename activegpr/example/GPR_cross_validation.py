from algpr.gpr import GaussianProcessRegressor
from algpr.kernels import RBF
from algpr.ppgpr import PPGPR
from algpr.vigpr import VIGPR
import numpy as np
import matplotlib.pyplot as plt

N = 200
X_g = np.linspace(-1.9, 1.9, N)
X = np.linspace(-2, 2, N) + np.concatenate([np.random.randn(50) * 0.1, np.random.randn(50) * 0.2, np.random.randn(50) * 0.3, np.random.randn(50) * 0.1])
Y = (np.sin(X) + np.random.randn(N) * 1e-1) *  (np.cos(2*X) + np.random.randn(N) * 1e-1)

gpr = GaussianProcessRegressor(kernel=RBF(l=np.array([0.9])))
gpr.fit(X, Y)

pred = gpr.predict(X_g)


rs_gpr = GaussianProcessRegressor(kernel=RBF(l=np.array([0.9])))
rs_index = np.random.choice(np.arange(N), 40)
rs_gpr.fit(X[rs_index], Y[rs_index])


pred = gpr.predict(X_g)
rs_pred = rs_gpr.predict(X_g)


plt.figure()
plt.plot(X, Y, '.')
plt.plot(X_g, pred, '-r')
plt.plot(X_g, rs_pred, '-b')
plt.show()