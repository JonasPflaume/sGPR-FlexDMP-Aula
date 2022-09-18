from copy import copy
from algpr.ppgpr import PPGPR
from algpr.kernels import kernel
from trajencoder.flexdmp.flexdmp import FlexDMP
import numpy as np
from casadi_kinodynamics.utils import symbolic_robot

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
        sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                            tool_mass=0.,
                                            tool_frame=[0.,0.,0.0] +
                                            [0., 0., 0.],
                                            tool_cog=[0.0, 0., 0.],
                                            tool_inertia=[
                                                0., 0., 0., 0., 0., 0.],
                                            load_casadi_fnc=False)
        self.q_min = np.array( sym_robot.q_min )
        self.q_max = np.array( sym_robot.q_max )

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
        if type(mass) == np.ndarray:
            mass = mass.squeeze()[0]
        qi_ = qi.copy()
        qf_ = qf.copy()
        # qi_ -= self.q_min
        # qi_ /= (self.q_max - self.q_min)
        # qf_ -= self.q_min
        # qf_ /= (self.q_max - self.q_min)
        x = np.concatenate([qi_, qf_, np.array([mass])]).reshape(1,-1)
        pred = self.gpr.predict(x)
        q, dq, ddq = self.trajectory(pred, qi, qf)
        return q, dq, ddq
    
if __name__ == "__main__":
    # example
    from algpr.kernels import RBF
    kernel = RBF(l=[1.5]*13, anisotropic=True)  # right kernel length is critical !
    model_addr = r"/home/jiayun/MotionLearning/suboptimal_planner/planner/models/Box1ToBox2_fixedOri_variable_payload_kernel_(1.5*13).npy"
    planner = PPGPRPlanner(model_addr=model_addr, ker=kernel, dof=6)
    
    import h5py
    import time
    import matplotlib.pyplot as plt
    def data(file_addr):
        F = h5py.File(file_addr)
        keys = list(F.keys())
        for key in keys:
            traj = F[key]
            q = traj["q"]
            dq = traj["qd"]
            ddq = traj["qdd"]
            tau = traj["tau"]
            t = traj["t"]
            x = traj["x"]
            qi, qf = x[:6], x[6:-1]
            yield qi, qf, q, dq, ddq, tau, t
    
    time_l = []
    for qi, qf, q, dq, ddq, tau, t in data("Vali_1e-07_jerk_4.0_mass.hdf5"):
        mass = 4.
        s = time.time()
        q, dq, ddq = planner(qi, qf, mass)
        e = time.time()
        time_l.append(e - s)
    plt.boxplot(time_l)
    plt.show()