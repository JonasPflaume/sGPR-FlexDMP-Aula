from algpr.ppgpr import PPGPR
from algpr.kernels import kernel
from trajencoder.flexdmp.flexdmp import FlexDMP
import numpy as np

class PPGPRPlanner:
    def __init__(self, model_addr:str, ker:kernel, dof:int, dt=0.004) -> None:
        ''' input:
                    model_addr:     The path of model file
                    kernel:         You should define a kernel indicated by the model file
                    dof:            Degree of freedom
        '''
        self.gpr = PPGPR(kernel=ker)
        self.dof = dof
        self.gpr.load_model(model_addr)
        self.dmp_order = (self.gpr.alpha.shape[1] - 1) // dof
        
    def flexdmp_reconstruct(self, PARAM, qi, qf, length, dt=0.004):
        ''' Input:
                    PARAM:          Predicting parameters, shape (dof, DMP_ORDER)
                    DMP_ORDER:      Order of FlexDMP
                    length:         integer, predicted time length
                    dof:            Dof of robot
                    qi, qf:  The start and goal of traj
        '''
        assert PARAM.shape == (self.dof, self.dmp_order), "Input the right shape of parameters!"
        dmp = FlexDMP(self.dmp_order, length, dt, dof=self.dof)

        dmp.set_weight(PARAM, qi, qf)
        res_q, res_dq, res_ddq = dmp.trajectory()
        return res_q, res_dq, res_ddq
        
    def trajectory(self, pred, qi, qf):
        ''' pred: 1d array
            qi:   1d array
            qf:   1d array
        '''
        if len(pred.shape) == 2:
            assert pred.shape[0] == 1, "Prediction has error !"
            pred = pred.squeeze()
            
        pred_param = pred[:-1].reshape(self.dof, self.dmp_order)
        pred_length = int(pred[-1])

        res_q, dq, ddq = \
            self.flexdmp_reconstruct(pred_param, qi, qf, pred_length)
        return res_q, dq, ddq
    
    def __call__(self, qi, qf, mass):
        ''' predict the trajectory !
        '''
        if type(qi) != np.ndarray:
            qi = np.array(qi)
        if type(qf) != np.ndarray:
            qf = np.array(qf)
        x = np.concatenate([qi, qf, np.array([mass])]).reshape(1,-1)
        pred = self.gpr.predict(x)
        q, dq, ddq = self.trajectory(pred, qi, qf)
        return q, dq, ddq