from aula.solver import SolverAugmentedLagrangian
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class MaxEntropy(MathematicalProgram):
    """ Maximize the entropy of gaussian process
    """

    def __init__(self, X, kernel, noise, lbx, ubx, calc_hessian=True, x_init=None):
        """
        X:          data matrix
        kernel:     kernel object
        noise:      noise level
        """
        # in case you want to initialize some class members or so...
        self.X = X
        self.dim_x = X.shape[1]
        self.kernel = kernel
        self.noise_level = noise
        self.bound_num = len(lbx)
        self.M = np.linalg.inv(self.kernel(X, X) + self.noise_level * np.eye(X.shape[0])) # inv(K + n*I)
        if type(lbx) != np.ndarray or type(ubx) != np.ndarray:
            self.lbx = np.array(lbx)
            self.ubx = np.array(ubx)
        else:
            self.lbx, self.ubx = lbx, ubx
        self.counter_evaluate = 0
        self.calc_hessian = calc_hessian
        self.x_init = x_init
        if self.calc_hessian:
            self.H = 0 # hessian initialization

    def evaluate(self, x):
        """ The problem we want to solve is:
                            1/2 * log (2*pi*e*var(x))
            input:
                    x: the current evaluated point
            output:
                    phi: the feature vector contains objective function value
                         and constraints
                    J:
                         Jaconbian matrix of phi w.r.t x
        """
        self.counter_evaluate += 1
        x = x.reshape(-1,1)
        first_term = self.kernel.kernel_grad(x.T, x.T)
        X_grad = self.kernel.kernel_grad(x.T, self.X)
        k = self.kernel(x.T, self.X) # k is a column vector
        second_term = 2 * X_grad @ self.M @ k.T
        sigma_grad = first_term - second_term

        sigma = self.kernel(x.T,x.T) - k @ self.M @ k.T
        obj = - 0.5 * np.log(2 * np.pi * np.e * sigma)
        obj_grad = - 1 / (2 * sigma) * sigma_grad

        phi_lbx = self.lbx - x.reshape(-1)
        phi_ubx = x.reshape(-1) - self.ubx

        lbx_grad = np.eye(self.dim_x) * -1
        ubx_grad = np.eye(self.dim_x)
        phi = np.concatenate([np.squeeze(obj, axis=1), phi_lbx, phi_ubx])

        # evaluation the approximated hessian
        if self.calc_hessian:
            sigma_grad_grad_first_term = self.kernel.kernel_hessian(self.getDimension())
            sigma_grad_grad_second_term = 2 * X_grad @ self.M @ X_grad.T
            sigma_grad_grad = sigma_grad_grad_first_term - sigma_grad_grad_second_term
            self.H = (sigma * sigma_grad_grad - sigma_grad @ sigma_grad.T) / (sigma ** 2)
        return phi, np.vstack([obj_grad.T, lbx_grad.T, ubx_grad.T])

    def getDimension(self):
        """ get the problem dimension
        """
        # return the input dimensionality of the problem (size of x)
        return self.dim_x

    def getInitializationSample(self):
        """ get the start point of x
        """
        if self.x_init is not None:
            return self.x_init
        else:
            return np.random.uniform(self.lbx, self.ubx)
    
    def getFHessian(self, x):
        """ return the hessian matrix
        """
        if self.calc_hessian:
            return self.H
        else:
            return 0

    def getFeatureTypes(self):
        """ get type of feature in phi
            those type are defined in OT
        """
        return [OT.f] + [OT.ineq] * self.dim_x * 2


class MaxEntropyBFGS:
    def __init__(self, X, kernel, noise):
        """ The special problem class designed to fit the scipy-L-BFGS-B optimization module
        X:          data matrix
        kernel:     kernel object
        noise:      noise level
        """
        temp = kernel(X, X) + noise * np.eye(X.shape[0])
        c, low = cho_factor(temp) # use chelosky to solve inverse
        self.M = cho_solve((c, low), np.eye(temp.shape[0]))

    @staticmethod
    def jacobian(x, kernel, X, M):
        x = x.reshape(-1,1)
        first_term = kernel.kernel_grad(x.T, x.T)
        X_grad = kernel.kernel_grad(x.T, X)
        k = kernel(x.T, X) # k is a column vector
        second_term = 2 * X_grad @ M @ k.T
        sigma_grad = first_term - second_term

        sigma = kernel(x.T,x.T) - k @ M @ k.T
        obj = - 0.5 * np.log(2 * np.pi * np.e * sigma)
        obj_grad = - 1 / (2 * sigma) * sigma_grad
        return obj_grad

    @staticmethod
    def obj(x, kernel, X, M):
        x = x.reshape(-1,1)
        k = kernel(x.T, X) # k is a column vector
        sigma = kernel(x.T,x.T) - k @ M @ k.T
        obj = - 0.5 * np.log(2 * np.pi * np.e * sigma)
        return obj