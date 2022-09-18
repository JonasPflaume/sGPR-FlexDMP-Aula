from trajencoder.flexdmp.flexdmp import FlexDMP
from algpr.kernels import RBF
from algpr.ppgpr import PPGPR
# active learning environment
from suboptimal_planner.data.collect_data_6dof import Env6dof

import numpy as np

Xres = np.load("/home/jiayun/MotionLearning/suboptimal_planner/data/X_mutinfoAL.npy")
rs = np.abs(Xres[:,:5]-Xres[:,5:]).sum(axis=1)
Xres = Xres[rs>0]
print(Xres.shape)
env = Env6dof(N=20)
Qf = []
Qi = []
Mass = []
for i in range(len(Xres)):
    qi = np.concatenate([Xres[i,:5], np.array([0.])])
    qf = np.concatenate([Xres[i, 5:], np.array([0.])])
    Qf.append(qi.tolist())
    Qi.append(qf.tolist())
    Mass.append(0.)
env.batch_record_multiprocessing(Qi, Qf, Mass, filename='Training_RandomToRandom_AL', 
                                 omit_qi=False, check_step=False)