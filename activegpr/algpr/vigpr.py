import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from algpr.kernels import RBF
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class VIGPR:
    ''' Variational sparse GPR follow the (Titsias 2009)

        Note1: too expensive to use for large dataset!!!
        Note2: Multiprocessing can only drag down the elbo computation

        # update1: optmized the code to improve the speed, now, it's ok to use under m <= 1000
        # update2: turn to use random subset to run the EM steps...
        # update3: reimplemented the cuda acceleration of elbo computation
        # update4: use random subset of Xm
    '''
    def __init__(self, kernel, noise_level=0.001):
        '''
            kernel:
                    the kernel function
            noise_level:
                    the preset noise variance in the dataset
        '''
        self.kernel = kernel
        self.noise_level = noise_level

    def fit(self, X, Y, m, call_hyper_opt=False, sub_L=50, init_data=3, max_dataset=500, cuda=False, Xm_subset_num=200):
        '''
        fit the data to get necessary parameters to predict
            sub_L:  The number of subset, from which we greedily choose one point
            m:      The number of subset
            init_data:
                    The initial data points in the incuding variable
            max_dataset:
                    Size of random subset of training data
            cuda:   Usually for large m, cuda will be faster than cpu. However, this VIGPR isn't a full cuda implementation
                    the data type conversion will cost a lot of time, but in larger m, this disadvantage will be neglected due to
                    the expensive matrix operation costs.
            call_hyper_opt:
                    when True, the fit will call __M_step, otherwise, it will use given hyperparameters
        '''
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        self.X, self.Y = X.copy(), Y.copy()
        n = X.shape[0] # training number
        d = X.shape[1] # feature dimension
        self.__reset_data_matrix()
        for _ in range(init_data):
            temp_indx = np.random.randint(0, n)
            self.Xm_index.append(temp_indx)
        # choose the Xm
        print("Start searching the inducing variables...")
        for i in tqdm(range(m)):
            self.Xm_index.append(self.__E_step(sub_L, max_dataset, cuda=cuda, Xm_subset_num=Xm_subset_num)) # update the list
            if call_hyper_opt:
                self.__M_step() # update the hyperparameters

        # start fit the predicting data
        self.Xm = self.X[self.Xm_index]
        Kmm = self.kernel(self.Xm, self.Xm) # K has shape (m,m)
        Kmn = self.kernel(self.Xm, self.X)
        if type(self.kernel) == RBF:
            Knm = Kmn.T
        else:
            Knm = self.kernel(X, self.Xm)
        
        try:
            temp = self.noise_level ** 2 * Kmm + Kmn @ Knm
            c, low = cho_factor(temp)
        except:
            print("The cho_factor meet singular matrix, now add damping...")
            temp += np.eye(len(self.Xm_index)) * 1e-9
            c, low = cho_factor(temp)
        self.alpha = cho_solve((c, low), Kmn @ self.Y)
        self.L = np.tril(c) if low else np.triu(c).T

        try:
            cmm, lowmm = cho_factor(Kmm)
        except:
            print("The cho_factor meet singular matrix, now add damping...")
            Kmm += np.eye(len(self.Xm_index)) * 1e-9
            cmm, lowmm = cho_factor(Kmm)
        self.Lmm = np.tril(cmm) if lowmm else np.triu(cmm).T
    
    def __reset_data_matrix(self):
        ''' reset those data matrix
        '''
        # the list save the Xm index
        self.Xm_index = []
        # process variable, in PPGPR it's no use to save them
        self.Kmn, self.Knm, self.Knn, self.Kmm = None, None, None, None
        # prediction necessary data matrix, important!
        self.alpha, self.L, self.Lmm = None, None, None

    def elbo(self, index_list:np.array, subset_list:np.array, Xm_subset_num=200):
        ''' calculate the variational lower bound
            index_list:     the points need to be tested, numpy 1d array
            subset_list:    the index of subset of full X
            return numpy 1d array elbo value
        '''
        if len(self.Xm_index) > Xm_subset_num:
            Xm_index = np.random.choice(self.Xm_index, Xm_subset_num, replace=False)
        else:
            Xm_index = self.Xm_index
        curr_Xm = self.X[Xm_index]
        X_chosen = self.X[index_list] # sub_L x d
        # sub_L x (Xm.len + 1) x d tensor, in the subsequent operation, we will use tensor product
        Xm_temp = np.array([np.concatenate([curr_Xm, X_chosen[i:i+1]]) for i in range(len(index_list))])
        # sub_L x X.len x (Xm.len + 1)
        X_subset, Y_subset = self.X[subset_list], self.Y[subset_list]
        Knm = np.array([self.kernel(X_subset, Xm_temp[i]) for i in range(len(index_list))])
        # sub_L x (Xm.len + 1) x (Xm.len + 1)
        Kmm_inv_Knm = []
        Kmm_L = []
        Kmm = []
        # sub_l x n x 1, important to only get the diagonal, expensive to get the full matrix
        Knn_diag_sum = self.kernel.diag(X_subset).sum()
        for i in range(len(index_list)):
            Kmm_i = self.kernel(Xm_temp[i], Xm_temp[i])
            Kmm.append(Kmm_i)
            try:
                c, low = cho_factor(Kmm_i)
                L_i = np.tril(c) if low else np.triu(c).T
                Kmm_L.append(L_i)
            except:
                c, low = cho_factor(Kmm_i + np.eye(Kmm_i.shape[0])*1e-9)
                L_i = np.tril(c) if low else np.triu(c).T
                Kmm_L.append(L_i)
            Kmm_inv_Knm_i = cho_solve((c, low), Knm[i].T)
            Kmm_inv_Knm.append(Kmm_inv_Knm_i)
        Kmm_inv_Knm = np.array(Kmm_inv_Knm)
        # sub_L x n x n
        # Q_tilda = []
        # for i in range(len(index_list)):
        #     Q_tilda.append(Knm[i] @ Kmm_inv[i] @ Knm[i].T)
        # Q_tilda = np.array(Q_tilda)
        # sub_L x n x n
        # K_tilda = Knn_diag - Q_tilda
        # start compute the elbo list
        elbo_list = []
        n, m = X_subset.shape[0], len(index_list)
        for i in range(len(index_list)):
            # ############################ inefficient implementation
            # try:
            #     c, low = cho_factor(Q_tilda[i] + self.noise_level**2*np.eye(Q_tilda[i].shape[0]))
            # except:
            #     print("Adding damping....")
            #     c, low = cho_factor(Q_tilda[i] + self.noise_level**2*np.eye(Q_tilda[i].shape[0]) + np.eye(Q_tilda[i].shape[0])*1e-10)
            # alpha_i = cho_solve((c, low), self.Y) # n, n_y
            # L_i = np.tril(c) if low else np.triu(c).T
            # elbo_i = -0.5 * np.einsum("ik,ik->k", self.Y, alpha_i) -np.log(np.diag(L_i)).sum() - self.X.shape[0] / 2 * np.log(2*np.pi) \
            #                 - 1/(2*self.noise_level**2) * (Knn_diag[i] - np.diag(Q_tilda[i]).reshape(-1,1)).sum()
            # elbo_list_1.append(elbo_i)
            # ############################
            F0 = -n/2 * np.log(2*np.pi) - (n-m)/2 * np.log(self.noise_level**2) - 1/(2*self.noise_level**2) * np.einsum("ij,ij->j", Y_subset, Y_subset) # (n_y,)
            F1 = np.log(np.diag(Kmm_L[i])).sum()

            temp_1 = self.noise_level**2*Kmm[i] + Knm[i].T @ Knm[i]

            try:
                temp_c, temp_low = cho_factor(temp_1)
                temp_L_i = np.tril(temp_c) if temp_low else np.triu(temp_c).T
            except:
                temp_c, temp_low = cho_factor(temp_1 + np.eye(temp_1.shape[0]) * 1e-9)
                temp_L_i = np.tril(temp_c) if temp_low else np.triu(temp_c).T
            F2 = - np.log(np.diag(temp_L_i)).sum()
            
            temp_2 = (Y_subset.T @ Knm[i]) @ cho_solve((temp_c, temp_low), (Knm[i].T @ Y_subset)) # temp_1 (m,m), temp2 (n,n)
            F3 = 1/(2*self.noise_level**2) * np.diag(temp_2) #(n_y,)

            F4 = -1/(2*self.noise_level**2) * Knn_diag_sum

            F5 = 1/(2*self.noise_level**2) * np.einsum("ij,ji->", Kmm_inv_Knm[i], Knm[i])

            elbo_i = (F0 + F1 + F2 + F3 + F4 + F5).sum()
            elbo_list.append(elbo_i)
        elbo_list = np.array(elbo_list).reshape(len(index_list),)

        return elbo_list

    def elbo_cuda(self, index_list:np.array, subset_list:np.array, Xm_subset_num=200):
        ''' calculate the variational lower bound utilizing cuda acceleration
            index_list:     the points need to be tested, numpy 1d array
            subset_list:    the index of subset of full X
            return numpy 1d array elbo value
        '''
        if len(self.Xm_index) > Xm_subset_num:
            Xm_index = np.random.choice(self.Xm_index, Xm_subset_num, replace=False)
        else:
            Xm_index = self.Xm_index
        curr_Xm = self.X[Xm_index]
        X_chosen = self.X[index_list] # sub_L x d
        # sub_L x (Xm.len + 1) x d tensor, in the subsequent operation, we will use tensor product
        Xm_temp = np.array([np.concatenate([curr_Xm, X_chosen[i:i+1]]) for i in range(len(index_list))])
        # sub_L x X.len x (Xm.len + 1)
        X_subset, Y_subset = self.X[subset_list], self.Y[subset_list]
        Knm = [torch.from_numpy(self.kernel(X_subset, Xm_temp[i])).to(device) for i in range(len(index_list))]
        Y_subset = torch.from_numpy(Y_subset).to(device)
        # sub_L x (Xm.len + 1) x (Xm.len + 1)
        Kmm_inv_Knm = []
        Kmm_L = []
        Kmm = []
        # sub_l x n x 1, important to only get the diagonal, expensive to get the full matrix
        Knn_diag_sum = self.kernel.diag(X_subset).sum()
        #################################################
        for i in range(len(index_list)):
            Kmm_i = torch.from_numpy(self.kernel(Xm_temp[i], Xm_temp[i])).to(device)
            Kmm.append(Kmm_i)
            try:
                c = torch.cholesky(Kmm_i)
                L_i = c
                Kmm_L.append(L_i)
            except:
                c = torch.cholesky(Kmm_i + torch.eye(Kmm_i.shape[0]).to(device)*1e-9)
                L_i = c
                Kmm_L.append(L_i)
            Kmm_inv_Knm_i = torch.cholesky_solve(Knm[i].T, c)
            Kmm_inv_Knm.append(Kmm_inv_Knm_i)

        elbo_list = []
        n, m = X_subset.shape[0], len(index_list)
        for i in range(len(index_list)):
            F0 = -n/2 * np.log(2*np.pi) - (n-m)/2 * np.log(self.noise_level**2) - 1/(2*self.noise_level**2) * torch.einsum("ij,ij->j", Y_subset, Y_subset) # (n_y,)
            F1 = torch.log(torch.diag(Kmm_L[i])).sum()

            temp_1 = self.noise_level**2*Kmm[i] + Knm[i].T @ Knm[i]

            try:
                temp_c = torch.cholesky(temp_1)
                temp_L_i = temp_c
            except:
                temp_c = torch.cholesky(temp_1 + torch.eye(temp_1.shape[0]).to(device) * 1e-9)
                temp_L_i = temp_c
            F2 = - torch.log(torch.diag(temp_L_i)).sum()
            temp_2 = (Y_subset.T @ Knm[i]) @ torch.cholesky_solve((Knm[i].T @ Y_subset), temp_c) # temp_1 (m,m), temp2 (n,n)
            F3 = 1/(2*self.noise_level**2) * torch.diag(temp_2) #(n_y,)

            F4 = -1/(2*self.noise_level**2) * Knn_diag_sum

            F5 = 1/(2*self.noise_level**2) * torch.einsum("ij,ji->", Kmm_inv_Knm[i], Knm[i])

            elbo_i = (F0 + F1 + F2 + F3 + F4 + F5).sum()

            elbo_i.detach().cpu().numpy()
            elbo_list.append(elbo_i)
        elbo_list = np.array(elbo_list).reshape(len(index_list),)

        return elbo_list

    def __E_step(self, sub_L, max_dataset, cuda, show_the_elbo=False, Xm_subset_num=200):
        ''' greedy choose one data point from subset L, which has the highest elbo
            max_dataset:        larger than 3000, then each time will only use subset of full X
            return:
                    x_i:        the index of the choosen point
        '''
        # choose sub_L random points
        index_pool = np.arange(len(self.X))
        index_pool_full = np.arange(len(self.X))
        if len(self.X) > max_dataset:
            index_pool = np.random.choice(index_pool, max_dataset, replace=False)
        Xm_index = np.array(self.Xm_index)
        index_pool = np.setdiff1d(index_pool, Xm_index, assume_unique=True)
        index_pool_full = np.setdiff1d(index_pool_full, Xm_index, assume_unique=True)
        sub_L_index = np.random.choice(index_pool_full, sub_L, replace=False) # no need to choose from index_pool
        if cuda:
            elbo_list = self.elbo_cuda(sub_L_index, index_pool, Xm_subset_num)
        else:
            elbo_list = self.elbo(sub_L_index, index_pool, Xm_subset_num)
        if show_the_elbo:
            import matplotlib.pyplot as plt
            plt.plot(sub_L_index, elbo_list, 'rx')
            plt.show()
        temp_index = np.argmax(elbo_list)
        x_i = sub_L_index[temp_index]
        return x_i

    def __M_step(self):
        ''' equivalent to self.__hyper_parameter_optimize
        '''

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
            v2 = solve_triangular(self.L, km.T, lower=True, check_finite=False) # very critical, this solve triangle system
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
            Note:
                    The LBFGSB methods is more efficient to treat box bounds than aula
        '''



if __name__ == "__main__":
    X = np.linspace(-20,20,10000)
    Y = np.sin(X) + np.sin(0.5*X)
    np.random.seed(0)
    from algpr.kernels import RBF
    import matplotlib.pyplot as plt
    from algpr.ppgpr import PPGPR

    ppgpr = PPGPR(kernel=RBF(l=[1.]), noise_level=1e-8)
    ppgpr.fit(X, Y, m=10)
    pred_pp, var_pp = ppgpr.predict(X, return_var=True)

    vigpr = VIGPR(kernel=RBF(l=[1.4]), noise_level=1e-8)
    vigpr.fit(X, Y, m=10, max_dataset=1000, sub_L=100, Xm_subset_num=25)
    pred_vi, var_vi = vigpr.predict(X, return_var=True)

    plt.plot(X, Y, '-.r', label="ground truth")
    plt.plot(X, pred_vi, '-b', label="vi gpr")
    plt.plot(X, pred_vi+var_vi, '-.b', linewidth=0.7)
    plt.plot(X, pred_vi-var_vi, '-.b', linewidth=0.7)

    plt.plot(X, pred_pp, '-c', label="pp gpr")
    plt.plot(X[vigpr.Xm_index], Y[vigpr.Xm_index], 'gv')
    plt.plot(X, pred_pp+var_pp, '-.c', linewidth=0.7)
    plt.plot(X, pred_pp-var_pp, '-.c', linewidth=0.7)
    plt.legend()
    plt.grid()
    plt.show()

    print("PP:", np.abs(Y-pred_pp.squeeze()).sum()/10000, "VI:", np.abs(Y-pred_vi.squeeze()).sum()/10000)