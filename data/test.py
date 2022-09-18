import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch import t


time7 = []
time57 = []
time6 = []
for mass in np.linspace(0.,4.0, 9):
    time7.append([])
    time57.append([])
    time6.append([])
    FILE7 = h5py.File("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/new_dynamics/box2box_fixedOri/Vali_1e-07_jerk_{:0.1f}_mass.hdf5".format(float(mass)))
    for key in FILE7.keys():
        time7[-1].append(np.array(FILE7[key]["t"])[-1])
        
    FILE57 = h5py.File("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/new_dynamics/box2box_fixedOri/Vali_5e-07_jerk_{:0.1f}_mass.hdf5".format(float(mass)))
    for key in FILE57.keys():
        time57[-1].append(np.array(FILE57[key]["t"])[-1])
        
    FILE6 = h5py.File("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/new_dynamics/box2box_fixedOri/Vali_1e-06_jerk_{:0.1f}_mass.hdf5".format(float(mass)))
    for key in FILE6.keys():
        time6[-1].append(np.array(FILE6[key]["t"])[-1])
    
time7 = np.array(time7)
time57 = np.array(time57)
time6 = np.array(time6)

plt.plot(np.linspace(0.,4.0, 9), time7.mean(axis=1), '-r')
plt.plot(np.linspace(0.,4.0, 9), time57.mean(axis=1), '-b')
plt.plot(np.linspace(0.,4.0, 9), time6.mean(axis=1), '-c')
plt.xlabel("mass [kg]")
plt.ylabel("mean time [s]")
plt.tight_layout()
plt.savefig("/home/jiayun/Desktop/test.jpg",dpi=100)
plt.show()