import h5py
import numpy as np

F = h5py.File("Vali_Box1ToBox2_0.0.hdf5")

Qi = []
Qf = []
for key in F.keys():
    traj = F[key]
    x = traj['x']
    qi = np.array( x[:6] ).reshape(1,-1)
    qf = np.array( x[6:12] ).reshape(1,-1)

    Qi.append(qi)
    Qf.append(qf)

Qi = np.concatenate(Qi)
Qf = np.concatenate(Qf)
X = np.concatenate([Qi, Qf], axis=1)
np.save("X_b1tb2.npy", X)
    
