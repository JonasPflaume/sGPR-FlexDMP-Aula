import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

class kernel:
    def __init__(self):
        ''' kernel base class
        '''
    
    def _definition(self, X, Y):
        ''' define the kernel function
            X, Y:
                    parameters have shape (n, N), n-point number, N-feature size
        '''
        raise NotImplementedError

    def __call__(self, X, Y):
        ''' X and Y should be the shape (1, N), N - feature size
        '''
        return self._definition(X, Y)

    def diag(self, X):
        ''' Only calculate the diagonal value
        '''
        n = X.shape[0]
        diag_terms = []
        for i in range(n):
            diag_terms.append(self._definition(X[i:i+1], X[i:i+1]))
        return np.array(diag_terms).reshape(n,)

    def kernel_grad(self, X, Y):
        ''' only desined for calculate the gradient of X == (1, N), Y == (1, N) or (n, N)
        '''
        raise NotImplementedError

class RBF(kernel):
    ''' Radial basis function kernel
        when anisotropic variant:
            exp(-(x - x').T * M * (x - x')), where M = diag(1/l**2)
    '''
    def __init__(self, l, anisotropic=False):
        super().__init__()
        self.anisotropic = anisotropic
        if anisotropic:
            assert len(l) > 1, "The anisotropic kernel need a vector length scale"
        else:
            assert len(l) == 1, "The l for insotropic should be 1 length"
        self.l = np.array(l)

    def _definition(self, X, Y):
        if self.anisotropic:
            distance = cdist(X / self.l, Y / self.l) # equal to np.sqrt((a-b).T @ w @ (a-b))
            return np.exp( - distance ** 2 )
        else:
        # calculate euclidian distance from X to Y
            distance = distance_matrix(X, Y)
            return np.exp( - distance ** 2 / (self.l ** 2) )

    def kernel_grad(self, X, Y):
        ''' get the gradient of kernel w.r.t x
        '''
        if self.anisotropic:
            derivative = self._definition(X, Y).T * (2 * (Y - X) @ np.diag(1/(self.l**2)))
        else:
            assert X.shape[0] == 1
            derivative = self._definition(X, Y).T * (2 * (Y - X) / (self.l ** 2)) # (n, 1) * (n, )
        return derivative.T

    def kernel_hessian(self, x_dim):
        ''' Only for the x==x scalar situation 
            Input:
                    x_dim: the dimension of the input variable
        '''
        if self.anisotropic:
            raise NotImplementedError("No hessian for the anisotropic version, BFGS is already well performing...")
        else:
            return -np.eye(x_dim) * (2/self.l**2)

    def kernel_grad_theta(self, X, K):
        ''' get the grad w.r.t hyperparameter theta
        '''
        if self.anisotropic:
            GRAD = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    self.l ** 2 
                    # found this comment in the sklearn source code that theta = log(l), so here is a **2 instead of **3
                    # $θ_i = \ln(λ_i)$, in Rasmussen.2006 (5.9)
                    # original gradient should multiply a λ_i due to chain rule
                )
            GRAD *= K[..., np.newaxis]
            return GRAD
        else: ### TODO the grandient for single length scale still have some problem ###
            dists = distance_matrix(X, X)
            GRAD = (K * dists)[:, :, np.newaxis]
            return GRAD

    @property
    def length_scale(self):
        return self.l

    @length_scale.setter
    def length_scale(self, new_l):
        if self.anisotropic and len(new_l) != len(self.l):
            raise ValueError("You should pass a new length scale the same as init setting.")
        self.l = new_l


if __name__ == "__main__":
    k = RBF(l=np.array([1.]))
    A = np.random.randn(10,2)
    B = np.random.randn(5,2)
    print(np.allclose(k(A, B), k(B, A).T))
    # RBF kernel is symmetric