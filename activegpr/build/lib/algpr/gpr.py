from unittest.mock import call
import numpy as np
from pyrsistent import l
import scipy
import copy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from algpr.entropy_maximizor import MaxEntropy, MaxEntropyBFGS
from algpr.hyperparameter import HyperparameterRBF
from aula.solver import SolverAugmentedLagrangian
from scipy.optimize import minimize

class GaussianProcessRegressor:
    ''' The implementation follows "Rasmussen, Gaussian processes for machine learning" Algorithm 2.1'''
    def __init__(self, kernel, noise_level=0., which_solver='LBFGSB', share_kernel=True):
        '''
            kernel:
                    the kernel function
            noise_level:
                    the preset noise variance in the dataset
        '''
        self.kernel = kernel
        self.noise_level = noise_level
        self.which_solver = which_solver # or "AULA"
        self.share_kernel = share_kernel # If false, the GPR will have Y_t kernel hyperparameters

    def fit(self, X, Y, call_hyper_opt=False):
        '''
        fit the data to get necessary parameters to predict
        '''
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        n = X.shape[0]
        d = X.shape[1]
        y_t = Y.shape[1]
        self.y_t = y_t
        self.X = X
        self.Y = Y
        ###
        if call_hyper_opt:
            if self.share_kernel:
                length_scale = self.__hyper_parameter_optimize(self.X, self.Y)
                self.kernel.l = length_scale
            else:
                length_scale = []
                for y_i in range(y_t):
                    length_scale_i = self.__hyper_parameter_optimize(self.X, self.Y[:,y_i:y_i+1])
                    length_scale.append(length_scale_i)
                self.kernel_l = [copy.deepcopy(self.kernel) for _ in range(y_t)]
                for y_i in range(y_t):
                    self.kernel_l[y_i].length_scale = length_scale[y_i]
        else:
            if not self.share_kernel:
                self.kernel_l = [copy.deepcopy(self.kernel) for _ in range(y_t)]
        ###
        if self.share_kernel:
            K = self.kernel(X, X) # K has shape (n,n) number of data
            try:
                temp = K + np.eye(n) * self.noise_level ** 2
                c, low = cho_factor(temp) # (n,n)
            except:
                print("The cho_factor meet singular matrix, now add damping...")
                temp += np.eye(temp.shape[0]) * 1e-9
                c, low = cho_factor(temp)
            
            self.alpha = cho_solve((c, low), Y) # (n,n_y) yd-output space dim, important -> cho_solve
            self.L = np.tril(c) if low else np.triu(c).T

            # print the margianl likelyhood
            evidence = - 0.5 * np.einsum("ik,ik->k", Y, self.alpha) - np.log(np.diag(self.L)).sum() - n / 2 * np.log(2*np.pi)
            evidence = evidence.sum(axis=-1)
            print("The evidence is: ", evidence)
            return evidence
        else:
            self.alpha = np.zeros([n, y_t])
            self.L = np.zeros([n,n,y_t])
            for y_i in range(y_t):
                K = self.kernel_l[y_i](X, X) # K has shape (n,n) number of data
                try:
                    temp = K + np.eye(n) * self.noise_level ** 2
                    c, low = cho_factor(temp) # (n,n)
                except:
                    print("The cho_factor meet singular matrix, now add damping...")
                    temp += np.eye(temp.shape[0]) * 1e-9
                    c, low = cho_factor(temp)
                
                self.alpha[:,y_i:y_i+1] = cho_solve((c, low), Y[:,y_i:y_i+1]) # (n,n_y) yd-output space dim, important -> cho_solve
                self.L[:,:,y_i] = np.tril(c) if low else np.triu(c).T
            
    def predict(self, x, return_var=False, return_prior_std=False):
        '''
        predict
        return_prior_std:
                    return_prior_std can only happend when the return_var is true
                    this is designed for the GpoeGPR
        '''
        if len(x.shape) == 1:
            # in case the data is 1-d
            x = x.reshape(-1,1)
        if self.share_kernel:
            k = self.kernel(x, self.X) # (n_x, n)
            mean = k @ self.alpha
            if return_var:
                v = solve_triangular(self.L, k.T, lower=True, check_finite=False) # very critical, this solve triangle system
                prior_std = self.kernel.diag(x)
                var = prior_std - np.einsum("ij,ji->i", v.T, v)
                if return_prior_std:
                    return mean, var.reshape(-1,1), prior_std.reshape(-1,1)
                else:
                    return mean, var.reshape(-1,1)
            else:
                return mean
            
        else:
            mean = np.zeros([len(x), self.y_t])
            if return_var:
                var = np.zeros([len(x), self.y_t])
            for y_i in range(self.y_t):
                k_i = self.kernel_l[y_i](x, self.X) # (n_x, n)
                mean_i = k_i @ self.alpha[:,y_i:y_i+1]
                mean[:,y_i:y_i+1] = mean_i
                if return_var:
                    v_i = solve_triangular(self.L[:,:,y_i], k_i.T, lower=True, check_finite=False) # very critical, this solve triangle system
                    prior_std = self.kernel_l[y_i].diag(x)
                    var_i = prior_std - np.einsum("ij,ji->i", v_i.T, v_i)
                    var[:,y_i] = var_i
                    
            if return_var:
                return mean, var
            else:
                return mean
                
    def max_entropy_x(self, lbx, ubx, x_init=None):
        ''' Find the maximum entropy point x
            Now it's only able to use RBF kernel
            Note:
                    The LBFGSB methods is more efficient to treat box bounds than aula
        '''
        if self.which_solver == "AULA":
            # define the optimization problem
            problem = MaxEntropy(self.X, self.kernel, self.noise_level, lbx, ubx, x_init=x_init)
            # initialize the augmented lagrangian solver
            solver = SolverAugmentedLagrangian()
            # solve the problem
            solver.setProblem(problem)
            x = solver.solve()
        elif self.which_solver == "LBFGSB":
            if type(lbx) != np.ndarray or type(ubx) != np.ndarray:
                lbx = np.array(lbx)
                ubx = np.array(ubx)
            problem = MaxEntropyBFGS(self.X, self.kernel, self.noise_level)
            if x_init is not None:
                x0 = x_init
            else:
                x0 = np.random.uniform(lbx, ubx)
            bounds = tuple(zip(lbx, ubx))
            kernel, X, M = self.kernel, self.X, problem.M
            res = minimize(problem.obj, x0, args=(kernel, X, M), jac=problem.jacobian,
                            bounds=bounds, method='L-BFGS-B')
            x = res.x.reshape(-1,1)
        return x

    def __hyper_parameter_optimize(self, X, Y, partial_dataset=2000):
        ''' The necessary helper function to optimize the hyperparameters of the covariance function
            partial_dataset:
                        When True and the dataset larger than 2000, 
                        the parameter optimization will only use 2000 random sampled data points
        '''
        problem = HyperparameterRBF()
        x0 = self.kernel.l
        if len(self.X) > partial_dataset:
            index = np.arange(0,len(self.X))
            np.random.shuffle(index)
            index = index[:partial_dataset]
            X, Y, noise = X[index], Y[index], self.noise_level
        else:
            noise = self.noise_level
        anisotropic = self.kernel.anisotropic
        try:
            res = minimize(problem.obj, x0, args=(X, Y, noise, anisotropic), jac=True, method='L-BFGS-B')
            length_scale = res.x
            print(res)
        except:
            print("The hyperparameter opt failed... pullback...")
            length_scale = x0
        return length_scale