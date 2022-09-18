import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from algpr.kernels import RBF

class HyperparameterRBF:
    ''' the definition of the gpr hyperparameters optimization problem
        For the L-BFGS
    '''
    def __init__(self):
        """ The special problem class designed to fit the scipy-L-BFGS-B optimization module
            The variables appear below
        X:          data matrix
        Y:          target matrix
        noise:      noise level
        """

    @staticmethod
    def obj(theta, X, Y, noise, anisotropic):
        ''' evidence = - 0.5 * y.T @ M @ y - 0.5 * log(det(M)) - n/2 * log(2*pi)
        '''
        kernel = RBF(l=theta, anisotropic=anisotropic)
        n = X.shape[0]
        d = X.shape[1]
        K = kernel(X, X) # K has shape (n,n) number of data
        try:
            temp = K + np.eye(n) * noise ** 2
            c, low = cho_factor(temp)
        except:
            temp += np.eye(n) * 1e-10
            c, low = cho_factor(temp)
        alpha = cho_solve((c, low), Y) # (n, n_y)
        L = np.tril(c) if low else np.triu(c).T

        evidence = - 0.5 * np.einsum("ik,ik->k", Y, alpha) - np.log(np.diag(L)).sum() - n / 2 * np.log(2*np.pi)
        evidence = evidence.sum(axis=-1)
        ################
        inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
        K_inv = cho_solve((L, True), np.eye(K.shape[0]), check_finite=False)
        inner_term -= K_inv[..., np.newaxis] # (n,n,y_out)
        K_partial_theta = kernel.kernel_grad_theta(X, K) # gradient tensor (t, n, n), t-theta number
        log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijk,jil->lk", inner_term, K_partial_theta
            )
            # the log likehood gradient is the sum-up across the outputs
        obj_grad = log_likelihood_gradient_dims.sum(axis=-1)
        print("Current evidence: ", evidence)
        return -evidence, -obj_grad
    
class HyperparameterPPGPR:
    ''' TODO
    '''
    pass