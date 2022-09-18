from algpr.gpoegpr import GpoeGPR
from algpr.ppgpr import PPGPR
from algpr.gpr import GaussianProcessRegressor
from algpr.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-5, 5, 1000) 
Y = 2*np.sin(X)

# set the random seed and partial dataset
np.random.seed(0)
index = np.random.randint(0, len(X), 100)
Xtrain, Ytrain = X[index].reshape(-1,1), Y[index].reshape(-1, 1)

# full gpr
gpr = GaussianProcessRegressor(kernel=RBF(l=np.array([1.])))
gpr.fit(Xtrain, Ytrain)
gpr_pred, gpr_std = gpr.predict(X, return_var=True)
# gpoe gpr
gpr_ = GpoeGPR(kernel=RBF(l=np.array([1.])), max_subset_num=10)
gpr_.fit(Xtrain, Ytrain, style='local') # 'local' will run kmean dataset split, 'rs' random subset
gpr_pred_, gpr_std_ = gpr_.predict(X)
# pp gpr
gpr__ = PPGPR(kernel=RBF(l=np.array([1.])))
gpr__.fit(Xtrain, Ytrain, m=7)
gpr_pred__, gpr_std__ = gpr__.predict(X, return_var=True)

plt.plot(X, Y, '-r', linewidth=4)
plt.plot(Xtrain, Ytrain, 'rx')
plt.plot(X, gpr_pred, '-b', label='full gpr')
plt.plot(X, gpr_pred+gpr_std, '-.b', linewidth=1)
plt.plot(X, gpr_pred-gpr_std, '-.b', linewidth=1)

plt.plot(X, gpr_pred_, '-c', label='gpoe gpr')
plt.plot(X, gpr_pred_+gpr_std_, '-.c', linewidth=1)
plt.plot(X, gpr_pred_-gpr_std_, '-.c', linewidth=1)

plt.plot(X, gpr_pred__, '-r', label='pp gpr')
plt.plot(X, gpr_pred__+gpr_std__, '-.r', linewidth=1)
plt.plot(X, gpr_pred__-gpr_std__, '-.r', linewidth=1)
plt.legend()
plt.show()