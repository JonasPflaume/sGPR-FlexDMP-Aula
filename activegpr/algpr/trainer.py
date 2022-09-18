import numpy as np
from algpr.gpr import GaussianProcessRegressor
from algpr.kernels import RBF

from typing import Callable
################################## TODO unify the training process ###############################
class ALGPRTrainer:
    def __init__(self, Xinit, Yinit, Xvali, Yvali, kernel, regressor='full', hyperparameter_opt=True):
        ''' Xinit, Yinit:
                    The initialized training dataset
            Xvali, Yvali:
                    The validation dataset
            regressor:
                    regressor style: full:  full gaussian process regression
        '''
        self.hyperparameter_opt = hyperparameter_opt
        self.Xvali, self.Yvali = Xvali, Yvali
        if regressor == "full":
            self.regressor = GaussianProcessRegressor(kernel=kernel)
        if hyperparameter_opt:
            self.regressor.fit(Xinit, Yinit, call_hyper_opt=hyperparameter_opt)
        else:
            self.regressor.fit(Xinit, Yinit, call_hyper_opt=hyperparameter_opt)

    def validate(self, loss_func: Callable, *args):
        '''
            loss_func:
                    The loss metric, which is a function object
            args:
                    contains all the necessary variables the loss_func needed
        '''

    def predict(self, X):
        ''' The function to call the regressor to predict
        '''
        return self.regressor.predict(X)

    def query(self):
        ''' The function to trigger the active learning scheme
            env:
                    The environment return the 
        '''
        x_next = self.regressor.max_entropy_x()
        return x_next
        
