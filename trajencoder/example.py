import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('..')

from trajencoder.dmp.dmp import DMP
from trajencoder.bspline.bspline import QuinticSpline




import h5py

FILE = h5py.File('/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/new_dynamics/box2box_fixedOri/Vali_1e-07_jerk_4.0_mass.hdf5')
random_index = list(FILE.keys())[1] # 129
y_demo_1, dy_demo_1, ddy_demo_1 = np.array(FILE[random_index]['q']), np.array(FILE[random_index]['qd']), np.array(FILE[random_index]['qdd'])
t = np.linspace(0, len(y_demo_1)*0.004 * 1.2, len(y_demo_1))

# DMP
dmp_demo = DMP(50, demo_len=len(y_demo_1), demo_dt=0.004)
weight_3 = dmp_demo.get_weights(y_demo_1[:,0], dy_demo=dy_demo_1[:,0], ddy_demo=ddy_demo_1[:,0])
q, dq, ddq = dmp_demo.trajectory()
plt.plot(dq)
plt.plot(dy_demo_1[:,0])
plt.show()

# y, dy, ddy = dmp_demo.trajectory() # 0.01 * 200 ->  2s

# test set weight
#dmp_demo_ = DMP(16, demo_len=1000, demo_dt=0.002)
#dmp_demo_.set_weight(weight_3, y_demo_1[0], y_demo_1[-1])
#y, dy, ddy = dmp_demo_.trajectory()

# B spline test
# bs = QuinticSpline(knots=15, init=0., end=0.)
# bs.get_weights_through_traj(t, y_demo_1)
# y_bs, dy_bs, ddy_bs = bs.trajectory(t[-1], dt=0.004, return_dyddy=True)

# plt.figure(figsize=[7,7])
# plt.subplot(311)
# plt.plot(np.linspace(0, 0.004*len(y_demo_1), len(y_demo_1)), y_demo_1, '-.r', label="demo")
# # plt.plot(y, '-b', label="dmp")
# plt.plot(np.linspace(0, 0.004*len(y_bs), len(y_bs)), y_bs, '-c', label="bspine")
# plt.ylabel("q")
# handles, labels = plt.gca().get_legend_handles_labels()
# labels, ids = np.unique(labels, return_index=True)
# handles = [handles[i] for i in ids]
# plt.legend(handles, labels, loc='best')
# plt.grid()
# # plt.savefig("../plots/traj_encoding/DMP_BSpline.jpg", dpi=200)

# plt.subplot(312)
# plt.plot(np.linspace(0, 0.004*len(y_demo_1), len(y_demo_1)), dy_demo_1, '-.r', label="demo")
# # plt.plot(dy, '-b', label="dmp")
# plt.plot(np.linspace(0, 0.004*len(dy_bs), len(dy_bs)), dy_bs, '-c', label="bspine")
# # plt.legend()
# plt.ylabel("dq")
# plt.grid()
# # plt.savefig("../plots/traj_encoding/DMP_BSpline.jpg", dpi=200)

# plt.subplot(313)
# plt.plot(np.linspace(0, 0.004*len(y_demo_1), len(y_demo_1)), ddy_demo_1, '-.r', label="demo")
# # plt.plot(ddy, '-b', label="dmp")
# plt.plot(np.linspace(0, 0.004*len(ddy_bs), len(dy_bs)), ddy_bs, '-c', label="bspine")
# # plt.legend()
# plt.grid()
# plt.ylabel("ddq")
# plt.xlabel("Time [s]")
# plt.tight_layout()
# plt.savefig("BSpline.jpg", dpi=200)

# plt.show()