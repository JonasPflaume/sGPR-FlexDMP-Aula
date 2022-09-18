import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from algpr.kernels import RBF

class FullGPMaxMulinfoBFGSF:
    def __init__(self):
        ''' the class to contain the obj and gradient function
        '''
    
    @staticmethod
    def obj(x, X, kernel, P, noise_level):
        ''' input:
                    x           - decision variable   shape:  (d,)
                    X           - current data matrix shape:  (n, d)
                    kernel      - kernel
                    noise_level - literal meaning
            return:
                    value       - objective function value
                    grad        - the grad calculated by pytorch
        '''
        if type(kernel) == RBF:
            # input should be (n, d) shape
            def torch_RBF(X_, Y_, l_): # aim to avoid calculating the gradient of mulinfo
                distance = torch.cdist(X_ / l_, Y_ / l_, p=P)
                return torch.exp( - distance ** P )
        
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        ### data selection
        if len(X) > 20:
            x = x.reshape(1,-1) # (1,d)
            distance_l = kernel(x, X)
            distance_l = distance_l.squeeze()
            if len(X) < 100:
                data_ratio = 1.
            elif len(X) > 100 and len(X) < 250:
                data_ratio = 0.8
            elif len(X) > 250 and len(X) < 500:
                data_ratio = 0.6
            elif len(X) > 500 and len(X) < 1000:
                data_ratio = 0.4
            elif len(X) > 1000 and len(X) < 2000:
                data_ratio = 0.2
            elif len(X) > 2000 and len(X) < 4000:
                data_ratio = 0.1
            elif len(X) > 4000 and len(X) < 8000:
                data_ratio = 0.05
            else:
                data_ratio = 0.025
            index_ne = int(len(X)*data_ratio)
            ranked_index = np.argsort(distance_l)[-index_ne:]
            X_sub = X[ranked_index]
        else:
            x = x.reshape(1,-1) # (1,d)
            X_sub = X
        ###
        
        x = torch.from_numpy(x).double().to(device=device)
        x.requires_grad = True
        X_sub = torch.from_numpy(X_sub).double().to(device=device)
        if not X_sub.is_contiguous():
            X_sub = X_sub.contiguous()
        l_ = torch.from_numpy(kernel.length_scale).double().to(device)
        kxx = torch_RBF(x, x, l_) # scalar
        kxX = torch_RBF(x, X_sub, l_) # (1, n)
        X_bar = torch.cat([X_sub, x], dim=0)
        
        kxX_bar = torch_RBF(x, X_bar, l_) # (1, n+1)
        
        K = torch_RBF(X_sub, X_sub, l_)
        K_bar = torch_RBF(X_bar, X_bar, l_)
        K_inv_temp = torch.cholesky(K + 1e-8*torch.eye(K.shape[0]).double().to(device))
        K_bar_inv_temp = torch.cholesky(K_bar + 1e-8*torch.eye(K_bar.shape[0]).double().to(device))
        K_inv = torch.cholesky_inverse(K_inv_temp)
        K_bar_inv = torch.cholesky_inverse(K_bar_inv_temp)
        
        num = kxx - kxX @ K_inv @ kxX.T
        den = kxx - kxX_bar @ K_bar_inv @ kxX_bar.T
        # if den < 1e-8: # avoid den being to small
        #     den += 1e-8
        value_torch = num / den
        value_torch.backward()
        grad = x.grad.cpu().detach().numpy().squeeze()
        value = value_torch.detach().cpu().numpy().squeeze()
        
        ### trust region gradient clip
        x = x.detach().cpu().numpy()
        return -value, -grad
    
    
class SparseMulinfoBFGSF:
    def __init__(self):
        ''' the class to contain the obj and gradient function
        '''
    
    @staticmethod
    def obj(x, Xm, X, kernel, noise_level):
        ''' input:
                    x           - decision variable   shape:  (d,)
                    X           - current data matrix shape:  (n, d)
                    kernel      - kernel
                    noise_level - literal meaning
            return:
                    value       - objective function value
                    grad        - the grad calculated by pytorch
        '''
        if type(kernel) == RBF:
            # input should be (n, d) shape
            def torch_RBF(X_, Y_, l_): # aim to avoid calculating the gradient of mulinfo
                distance = torch.cdist(X_ / l_, Y_ / l_)
                return torch.exp( - distance ** 2 )
        
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        x = torch.from_numpy(x).double().to(device=device)
        x = x.reshape(1,-1) # (1,d)
        x.requires_grad = True
        X = torch.from_numpy(X).double().to(device=device)
        Xm = torch.from_numpy(Xm).double().to(device=device)
        if not X.is_contiguous():
            X = X.contiguous()
        if not Xm.is_contiguous():
            Xm = Xm.contiguous()
        
        # previous
        l_ = torch.from_numpy(kernel.length_scale).double().to(device)
        kxx = torch_RBF(x, x, l_) # scalar
        kxm = torch_RBF(x, Xm, l_) # (1, n)
        Kmn = torch_RBF(Xm, X, l_)
        Kmm = torch_RBF(Xm, Xm, l_)
        K_sparse = (noise_level*Kmm + Kmn @ Kmn.T)
        
        # current
        X_ = torch.cat([X, x], dim=0)
        Xm_ = torch.cat([Xm, x], dim=0)
        kxm_ = torch_RBF(x, Xm_, l_) # (1, n)
        Kmn_ = torch_RBF(Xm_, X_, l_)
        Kmm_ = torch_RBF(Xm_, Xm_, l_)
        K_sparse_ = (noise_level*Kmm_ + Kmn_ @ Kmn_.T)
        
        # inverse
        Kmm_inv_temp = torch.cholesky(Kmm + 1e-8*torch.eye(Kmm.shape[0]).double().to(device=device))
        Kmm_inv = torch.cholesky_inverse(Kmm_inv_temp)
        Kmm_inv_temp_ = torch.cholesky(Kmm_ + 1e-8*torch.eye(Kmm_.shape[0]).double().to(device=device))
        Kmm_inv_ = torch.cholesky_inverse(Kmm_inv_temp_)
        
        K_sparse_inv_temp = torch.cholesky(K_sparse + 1e-8*torch.eye(K_sparse.shape[0]).double().to(device=device))
        K_sparse_inv = torch.cholesky_inverse(K_sparse_inv_temp)
        K_sparse_inv_temp_ = torch.cholesky(K_sparse_ + 1e-8*torch.eye(K_sparse_.shape[0]).double().to(device=device))
        K_sparse_inv_ = torch.cholesky_inverse(K_sparse_inv_temp_)
        
        num = kxx - kxm @ Kmm_inv @ kxm.T + noise_level * kxm @ K_sparse_inv @ kxm.T
        den = kxx - kxm_ @ Kmm_inv_ @ kxm_.T + noise_level * kxm_ @ K_sparse_inv_ @ kxm_.T
        # if den < 1e-8: # avoid den being to small
        #     den += 1e-8
        value_torch = num / den
        value_torch.backward()
        grad = x.grad.cpu().detach().numpy().squeeze()
        value = value_torch.detach().cpu().numpy().squeeze()
        print(value)
        return -value, -grad