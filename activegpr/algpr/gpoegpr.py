import numpy as np
from scipy.optimize import minimize
import scipy, math
import copy
from algpr.gpr import GaussianProcessRegressor
from sklearn.cluster import KMeans

class GpoeGPR:
    '''
    The implementation of generalized product of expert GPR (GpoeGPR)
    Implementation follows:
            Cao, Y. and Fleet, D.J., 2014. Generalized product of experts for automatic and
            principled fusion of Gaussian process predictions. arXiv preprint arXiv:1410.7827.
    '''
    def __init__(self, kernel, max_subset_num=100):
        ''' kernel:
                    kernel instance
            max_subset_num:
                    the recommend subset number, the number of regressor will be determined by this factor
        '''
        self.kernel = kernel
        self.max_subset_num = max_subset_num

    def predict(self, x):
        ''' The prediction would be expensive than vanilla gpr
            Because in this framework, you have to output variance
        '''
        if len(x.shape) == 1:
            # in case the data is 1-d
            x = x.reshape(-1,1)
        gpr_num = len(self.regressor_l)
        sample_num = len(x)

        mean, var, prior_var = np.zeros([gpr_num, sample_num, self.output_dim]), np.zeros([gpr_num, sample_num, 1]),\
                                                                                 np.zeros([gpr_num, sample_num, 1])
        for i in range(gpr_num):
            mean_i, var_i, prior_var_i = self.regressor_l[i].predict(x, return_var=True, return_prior_std=True)
            mean[i,:,:] = mean_i
            var[i,:,:] = var_i
            prior_var[i,:,:] = prior_var_i
        # remove 0 from var
        var[np.isclose(var, np.zeros_like(var), rtol=1e-12, atol=1e-12)] = 1e-12
        # need to make sure the weights of each expert
        # \Delta H(x) = H_prior(x) - H(x)
        DeltaH = 0.5 * np.log( prior_var / var ) # shape (gpr_num, sample_num, 1), 1e-16 avoid infinity certainty in no noise process
        # normalize the DeltaH to sum to 1
        sum_res = DeltaH.sum(axis=(0,2))
        DeltaH /= sum_res[np.newaxis, :, np.newaxis]
        # vote
        temp = DeltaH * (1/var) # * precision
        var_vote = 1 / temp.sum(axis=0)
        mean_vote = (mean * temp).sum(axis=0) * var_vote # (sample_num, output_dim) * (sample_num, 1)
        return mean_vote, var_vote

    def local_predict(self, x):
        ''' use the knowledge of kmeans clusters, to dispatch the input x.
            by doing so, we can avoid calculate the variance of the prediction
        '''
        assert self.structures != None

    def fit(self, X, Y, noise_level=0., call_hyper_opt=False, style='rs'):
        ''' 1. divide the dataset
            2. train individual regressor
        '''
        self.total_sample_num, self.input_dim, self.output_dim = X.shape[0], X.shape[1], Y.shape[1]
        X_l, Y_l, gpr_num = self.__dataset_division(X, Y, style=style)
        self.regressor_l = [GaussianProcessRegressor(kernel=copy.deepcopy(self.kernel), noise_level=noise_level) for _ in range(gpr_num)]
        for i in range(gpr_num):
            self.regressor_l[i].fit(X_l[i], Y_l[i], call_hyper_opt=call_hyper_opt)
        print("Fitting finished, there are {} gpr regressor were created!".format(gpr_num))

    def step_fit(self):
        ''' TODO need to find a better way to insert the data
            in order to save kmean or tree training time
        '''

    def max_entropy_x(self):
        ''' TODO The max entropy problem need to be re-designed now
        '''

    def __dataset_division(self, X, Y, style='rs'):
        ''' style:
                rs:     Random subset
                tree:   Tree based division
                local:  Random local division
            return:
                X_l:        the divided input datasets, list object
                Y_l:        the divided output datasets, list object
                gpr_num:    the number of gpr need to create seperately 
        '''
        if style == 'rs':
            X_l, Y_l, gpr_num = self.__random_subset(X, Y)
        elif style == 'local':
            X_l, Y_l, gpr_num = self.__local(X, Y)

        return X_l, Y_l, gpr_num

    def __random_subset(self, X, Y):
        ''' random subset dataset division
            X, Y:   The big dataset
        '''
        gpr_num = math.ceil( len(X) / self.max_subset_num )
        index = np.arange(len(X))
        each_gpr_samples = len(X) // gpr_num
        np.random.shuffle(index)
        X_l, Y_l = [], []
        for i in range(gpr_num):
            if i != (gpr_num-1):
                X_l.append(X[i*each_gpr_samples:(i+1)*each_gpr_samples,:])
                Y_l.append(Y[i*each_gpr_samples:(i+1)*each_gpr_samples,:])
            else:
                X_l.append(X[i*each_gpr_samples:,:])
                Y_l.append(Y[i*each_gpr_samples:,:])
        return X_l, Y_l, gpr_num

    def __tree(self):
        ''' TODO
        '''
        pass

    def __local(self, X, Y):
        ''' Kmeans localization datasets
        '''
        gpr_num = math.ceil( len(X) / self.max_subset_num )
        self.structures = KMeans(n_clusters=gpr_num, random_state=0).fit(X)
        X_l, Y_l = [], []
        for i in range(gpr_num):
            index_i = np.where(self.structures.labels_ == i)
            X_l.append(X[index_i])
            Y_l.append(Y[index_i])
        return X_l, Y_l, gpr_num