from trajencoder.flexdmp.flexdmp_nnforcing import ForcingFlexDMP
import numpy as np
import h5py, os
import tqdm
import glob

file_l = glob.glob("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/Training_RandomToRandom_*")

for j, file_addr in enumerate(file_l):
    X, Y = [], []
    f = h5py.File(file_addr, 'r')
    for label in tqdm.tqdm(f.keys()):
        x = f[label]["x"]
        t, y, dy, ddy = f[label]["t"], f[label]["q"], f[label]["qd"], f[label]["qdd"]
        dmp = ForcingFlexDMP(len(t), 0.004, dof=6)
        forcing = dmp.get_forcing_ground_trueth(y, dy_demo=dy, ddy_demo=ddy)
        x = np.array(x, dtype=float)[:-1]
        for i in range(len(t)):
            xi = np.concatenate([x[6:]-y[i], y[i], dy[i], ddy[i], np.array([i], dtype=float)])
            X.append(xi)
            Y.append(forcing[i])
        
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    np.save("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/forcingdmpHuge_X_{}.npy".format(j+1), X)
    np.save("/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/forcingdmpHuge_Y_{}.npy".format(j+1), Y)
