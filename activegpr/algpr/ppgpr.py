import numpy as np
import scipy
import copy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from torch import solve
from itertools import product
from algpr.kernels import RBF
from algpr.mulinfo_maximizor import FullGPMaxMulinfoBFGSF, SparseMulinfoBFGSF

class PPGPR:
    ''' The implementation follows "Rasmussen 2006 Chapter 8" 
        Projected process sparse Gaussian Processes
        Note1:
                Didn't implement the hyperparameter optimization and active learning functions
                It will be realized in VIGPR, which has a better optimization objective than PPGPR
        Note2:
                Sparse optimal experiment design is not ideal! Use full.
    '''
    def __init__(self, kernel, noise_level=0.):
        '''
            kernel:
                    the kernel function
            noise_level:
                    the preset noise variance in the dataset
        '''
        self.kernel = kernel
        self.noise_level = noise_level
        self.which_solver = 'LBFGSB'

    def fit(self, X, Y, m, inducing_style='random', call_hyper_opt=False):
        '''
        fit the data to get necessary parameters to predict
            m:      The number of subset
        '''
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        n = X.shape[0]
        d = X.shape[1] # feature dimension
        self.Xm = self.__induced_variable_choose(X, m, inducing_style) # only need to store m sub points
        self.Y = Y
        ###
        if call_hyper_opt:
            length_scale = self.__hyper_parameter_optimize()
            self.kernel.l = length_scale
        ###
        Kmm = self.kernel(self.Xm, self.Xm) # K has shape (m,m)
        Kmn = self.kernel(self.Xm, X)
        if type(self.kernel) == RBF:
            Knm = Kmn.T
        else:
            Knm = self.kernel(X, self.Xm)
        try:
            temp = self.noise_level ** 2 * Kmm + Kmn @ Knm
            c, low = cho_factor(temp) # (n,n)
        except:
            print("The cho_factor meet singular matrix, now add damping...")
            temp += np.eye(m) * 1e-9
            c, low = cho_factor(temp)
        self.alpha = cho_solve((c, low), Kmn @ Y)
        self.L = np.tril(c) if low else np.triu(c).T

        try:
            cmm, lowmm = cho_factor(Kmm)
        except:
            print("The cho_factor meet singular matrix, now add damping...")
            Kmm += np.eye(m) * 1e-9
            cmm, lowmm = cho_factor(Kmm)
        self.Lmm = np.tril(cmm) if lowmm else np.triu(cmm).T

    def __induced_variable_choose(self, X, m, style):
        ''' choose the induced subset of X
        '''
        if style == 'random':
            n = len(X)
            assert n >= m, "please choose a smaller m !"
            index = np.arange(n)
            index_m = np.random.choice(index, m, replace=False)

            return X[index_m]
        
        elif style == 'box2box':
            assert X.shape[1] == 13, "This is NOT box2box case!" 
            l = int( np.ceil( np.sqrt( m / 9 )) )
            box1 = np.unique(X[:,:6], axis=0)
            box2 = np.unique(X[:,6:12], axis=0)
            sorted_box1_index_mi = np.array([], dtype=int)
            sorted_box2_index_mi = np.array([], dtype=int)
            kernel = RBF(l=[2]*6, anisotropic=True)
            for _ in range(len(box1)-1): # arange the box
                max_index_box1_mi, _ = PPGPR.max_mulinfo_x_discrete(box1, sorted_box1_index_mi, kernel=kernel)
                max_index_box2_mi, _ = PPGPR.max_mulinfo_x_discrete(box2, sorted_box2_index_mi, kernel=kernel)
                sorted_box1_index_mi = np.concatenate([sorted_box1_index_mi, np.array([max_index_box1_mi])])
                sorted_box2_index_mi = np.concatenate([sorted_box2_index_mi, np.array([max_index_box2_mi])])
            box1_l_mi = box1[sorted_box1_index_mi[:l]]
            box2_l_mi = box2[sorted_box2_index_mi[:l]]
            box_mix_mi = np.array( list(product(box1_l_mi, box2_l_mi))).reshape(-1,12)
            index_temp_mi = []
            for x_mix in box_mix_mi:
                index_i = np.all(np.isclose(X[:,:-1], x_mix), axis=1)
                index_temp_mi += np.where(index_i)[0].tolist()
            index_temp_mi = np.array( index_temp_mi)
            index_temp_mi = np.random.choice( index_temp_mi, m, replace=False)
            Xp_mi = X[index_temp_mi]
            return Xp_mi

        elif style == 'entropy':
            d = X.shape[1]
            kernel = RBF(l=[float(d)]*d, anisotropic=True)
            x_index_list = np.random.choice(np.arange(len(X)), 1)
            for i in range(m):
                x_index, _ = PPGPR.max_entropy_x_discrete(X, x_index_list, kernel)
                x_index_list = np.concatenate([x_index_list, np.array([x_index])])
                
            return X[x_index_list]

    def predict(self, x, return_var=False, return_type='std', return_prior_std=False):
        '''
        predict
        return_prior_std:
                    return_prior_std can only happend when the return_var is true
                    this is designed for the GpoeGPR
        '''
        if len(x.shape) == 1:
            # in case the data is 1-d
            x = x.reshape(-1,1)
        km = self.kernel(x, self.Xm) # (n_x, m)
        mean = km @ self.alpha
        if return_var:
            # second term in book
            v2 = solve_triangular(self.L, km.T, lower=True, check_finite=False) # very critical, this solve triangle operation
            # first term in book
            v1 = solve_triangular(self.Lmm, km.T, lower=True, check_finite=False)
            if return_type == 'cov':
                n_x = x.shape[0]
                prior_cov = self.kernel(x,x)
                cov =  prior_cov - v1.T @ v1 + self.noise_level**2 * v2.T @ v2
                if return_prior_std:
                    return mean, cov, prior_cov
                else:
                    return mean, cov
            elif return_type == 'std':
                prior_std = self.kernel.diag(x) # the diagonal of covariance of kernel(x,x)
                var = prior_std - np.einsum("ij,ji->i", v1.T, v1) + self.noise_level**2 * np.einsum("ij,ji->i", v2.T, v2)
                if return_prior_std:
                    return mean, var.reshape(-1,1), prior_std.reshape(-1,1)
                else:
                    return mean, var.reshape(-1,1)
        else:
            return mean
        
    def save_model(self, name):
        ''' save the model(state dict) to current cwd
        '''
        state_dict = {
            "Xm": self.Xm.copy(),
            "alpha": self.alpha.copy(),
            "L": self.L.copy(),
            "Lmm": self.Lmm.copy()
        }
        np.save(r"{}.npy".format(name), state_dict)
        
    def load_model(self, addr):
        ''' save the model(state dict) from current cwd
            addr - the path of .npy file
                Please give a absolute path !!!
        '''
        np_load_old = np.load
        np.load.__defaults__=(None, True, True, 'ASCII')
        
        state_dict = np.load(addr)
        self.Xm = state_dict[()]["Xm"]
        self.alpha = state_dict[()]["alpha"]
        self.L = state_dict[()]["L"]
        self.Lmm = state_dict[()]["Lmm"]
        np.load.__defaults__=(None, False, True, 'ASCII')
        
    def max_entropy_x(self, lbx, ubx, x_init=None):
        ''' Find the maximum entropy point x
            Now it's only able to use RBF kernel
            Note1:
                    The LBFGSB methods is more efficient to treat box bounds than aula
            Note2:
                    This can be abandoned in the development of future, the mutual information is superior than 
                    entropy-based method
        '''

    def __hyper_parameter_optimize(self, partial_dataset=2000):
        ''' The necessary helper function to optimize the hyperparameters of the covariance function
            partial_dataset:
                        When True and the dataset larger than 2000, 
                        the parameter optimization will only use 2000 random sampled data points
            
            Note:
                    This function is relevant to the target, therefore, computationally expensive
        '''
        
    @staticmethod
    def max_mulinfo_x(lbx, ubx, Xcurr, kernel, noise_level, P, solve_type='full'):
        ''' Objective function and gradient was calculated according to full GPR!
        
            Greedy approximation by maximizing the mutual information
            Implemented based on "Andreas Krause 2008"
            Based on Xinit, choose the next x
        '''
        if type(lbx) != np.ndarray or type(ubx) != np.ndarray:
                lbx = np.array(lbx)
                ubx = np.array(ubx)
        if solve_type=='sparse':
            problem = SparseMulinfoBFGSF()
            max_m = 7000
        elif solve_type=='full':
            problem = FullGPMaxMulinfoBFGSF()
        
        x0 = np.random.uniform(lbx, ubx) * 0.1
        bounds = tuple(zip(lbx, ubx))
        
        X = Xcurr
        if solve_type=='sparse':
            if len(X) < max_m:
                index = np.arange(len(X))
                if len(X)//2 == 0:
                    index_selected = np.random.choice(index, 1)
                else:
                    index_selected = np.random.choice(index, len(X)//2)
                Xm = X[index_selected]
            else:
                index = np.arange(len(X))
                index_selected = np.random.choice(index, max_m)
                Xm = X[index_selected]
            
            res = minimize(problem.obj, x0, args=(Xm, X, kernel, P, noise_level), jac=True,
                            bounds=bounds, method='L-BFGS-B')
        elif solve_type == 'full':
            res = minimize(problem.obj, x0, args=(X, kernel, P, noise_level), jac=True,
                                bounds=bounds, method='L-BFGS-B')
        x = res.x.reshape(-1,1)
        return x
        
    @staticmethod
    def max_mulinfo_X(lbx, ubx, Xinit, kernel, P, noise_level, N, solve_type):
        ''' Greedy approximation by maximizing the mutual information
            Implemented based on "Andreas Krause 2008"
            Input:
                    lbx, ubx:       lower and upper bound of x
                    Xinit:          initial set of data points
                    N:              the N number of datapoints, should be choosen by the mulinfo criterion
            Output:
                    XU:             Union of initial and chosen points
            Given a initial set of X samples, this static function should return the whole set of
            N data-points, Xinit union X_(V/Xinit).
        '''
        X_cont = Xinit.copy()
        import tqdm
        for i in tqdm.tqdm(range(N)):
            x_next = PPGPR.max_mulinfo_x(lbx, ubx, Xinit, kernel, noise_level, P, solve_type=solve_type)
            x_next = x_next.reshape(1,-1)
            X_cont = np.concatenate([X_cont, x_next], axis=0)
            Xinit = X_cont.copy()
        return X_cont
    
    @staticmethod
    def max_mulinfo_x_discrete(Xpool, Xcurr_index, kernel):
        ''' Objective function and gradient was calculated according to full GPR!
        
            Greedy approximation by maximizing the mutual information
            Implemented based on "Andreas Krause 2008"
            Based on Xcurr, choose the next x from discrete Xpool.
        '''
        sigma_list = []
        complement_index = np.setdiff1d( np.arange(len(Xpool)), Xcurr_index )
        if Xcurr_index.size == 0:
            Xcurr = 0
        else:
            Xcurr = Xpool[Xcurr_index]

        for index in complement_index:
            x = Xpool[index].reshape(1,-1)
            kxx = kernel(x, x)
            if Xcurr_index.size == 0:
                kxX = 0.
                K = 0.
            else:
                kxX = kernel(x, Xcurr)
                K = kernel(Xcurr, Xcurr)
            
            Xbar_index = np.setdiff1d( complement_index, index)
            X_bar = Xpool[Xbar_index]
            kxX_bar = kernel(x, X_bar)
            K_bar = kernel(X_bar, X_bar)
            
            if Xcurr_index.size != 0:
                try:
                    c, low = cho_factor(K) # (n,n)
                except:
                    # print("The cho_factor meet singular matrix, now add damping...")
                    K += np.eye(K.shape[0]) * 1e-9
                    c, low = cho_factor(K)
                tempK = cho_solve((c, low), kxX.T)
            
            try:
                c, low = cho_factor(K_bar) # (n,n)
            except:
                # print("The cho_factor meet singular matrix, now add damping...")
                K_bar += np.eye(K_bar.shape[0]) * 1e-9
                c, low = cho_factor(K_bar)
            tempK_bar = cho_solve((c, low), kxX_bar.T)
            
            if Xcurr_index.size == 0:
                num = kxx
            else:
                num = kxx - kxX @ tempK
            den = kxx - kxX_bar @ tempK_bar
            if den <= 1e-8:
                den += 1e-8
            value = num / den
            sigma_list.append(value)
            
        sigma_list = np.array(sigma_list).squeeze()
        
        sigma_max_index = np.argmax( sigma_list )
        x_max_mulinfo_index = complement_index[sigma_max_index]
        
        return x_max_mulinfo_index, sigma_list
    
    @staticmethod
    def max_entropy_x_discrete(Xpool, Xcurr_index, kernel):
        ''' Objective function and gradient was calculated according to full GPR!
        
            Greedy approximation by maximizing the mutual information
            Implemented based on "Andreas Krause 2008"
            Based on Xcurr, choose the next x from discrete Xpool.
        '''
        complement_index = np.setdiff1d( np.arange(len(Xpool)), Xcurr_index )
        Xcurr = Xpool[Xcurr_index]
        K = kernel(Xcurr, Xcurr)
        try:
            c, low = cho_factor(K) # (n,n)
        except:
            # print("The cho_factor meet singular matrix, now add damping...")
            K += np.eye(K.shape[0]) * 1e-9
            c, low = cho_factor(K)
        
        L = np.tril(c) if low else np.triu(c).T
        k = kernel(Xpool[complement_index], Xcurr)
        v = solve_triangular(L, k.T, lower=True, check_finite=False)
        prior_std = kernel.diag(Xpool[complement_index])
        var = prior_std - np.einsum("ij,ji->i", v.T, v)
        entropy = 1/2 * np.log( np.pi * np.e * var)
        sigma_list = entropy.squeeze()
        sigma_max_index = np.argmax( sigma_list )
        x_max_entropy_index = complement_index[sigma_max_index]
        
        return x_max_entropy_index, sigma_list
            
if __name__ == "__main__":
    Xpool = np.load("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/box_oppo/box1_pool.npy")
    # Xpool = Xpool[np.random.choice(np.arange(len(Xpool)), 20)]
    print(Xpool.shape) # ceil (7,7,5)
    
    Xcurr_index = np.array([], dtype=int)

    kernel = RBF(l=[1.4]*6, anisotropic=True)
    for i in range(100):
        print("Start %ith ..." % i)
        x_next_index, sigma_list = PPGPR.max_mulinfo_x_discrete(Xpool, Xcurr_index, kernel)
        Xcurr_index = np.concatenate([Xcurr_index, np.array([x_next_index])])
    print(Xcurr_index)
    
    np.save("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/box_oppo/box1_picked_index.npy", Xcurr_index)
    
    Xpool = np.load("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/box_oppo/box2_pool.npy")
    # Xpool = Xpool[np.random.choice(np.arange(len(Xpool)), 20)]
    print(Xpool.shape) # ceil (7,7,5)
    
    Xcurr_index = np.array([], dtype=int)

    kernel = RBF(l=[1.4]*6, anisotropic=True)
    for i in range(100):
        print("Start %ith ..." % i)
        x_next_index, sigma_list = PPGPR.max_mulinfo_x_discrete(Xpool, Xcurr_index, kernel)
        Xcurr_index = np.concatenate([Xcurr_index, np.array([x_next_index])])
    print(Xcurr_index)
    
    np.save("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/box_oppo/box2_picked_index.npy", Xcurr_index)