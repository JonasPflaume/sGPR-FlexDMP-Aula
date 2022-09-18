import sys, os, h5py, re
sys.path.append('..')
curr_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_path)
from data.collect_data import collect_data_with_dmp_weight
from casadi_kinodynamics.utils import symbolic_robot
import numpy as np
import matplotlib.pyplot as plt

def experiment_data():
    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB910_2r', tool_mass=0.0,
                                                    tool_frame=[0., 0., 0.3, 0., 0., 0.], load_casadi_fnc=False)
    q_min, q_max = sym_robot.q_min, sym_robot.q_max
    qd_min, qd_max = np.zeros(2), sym_robot.qd_max

    range_q_list = []
    for _ in range(200):
        start = np.random.uniform(q_min, q_max)
        end = np.random.uniform(q_min, q_max)
        if  np.any(start < q_min) or np.any(start > q_max) or np.any(end > q_max) or np.any(end < q_min):
            print("hello")
        range_q_list.append([list(np.copy(start)), list(np.copy(end))])
    for _ in range(200):
        start = np.random.uniform(q_min, q_max)
        end = start + np.random.randn(*start.shape) * 1e-3 # add a very small noise term
        end = np.clip(end, q_min, q_max)
        if  np.any(start < q_min) or np.any(start > q_max) or np.any(end > q_max) or np.any(end < q_min):
            print("hello")
        range_q_list.append([list(np.copy(start)), list(np.copy(end))])

    f = h5py.File(os.path.join(curr_path,"encoder_test.hdf5"), "w")

    for q in range_q_list:
            q1, q2 = q[0], q[1]
            qi = q1
            qf = q2
            qdi = [i for i in [0., 0.]]
            qdf = [i for i in [0., 0.]]
            print(qi, qf)
            dt, t, q, qd, qdd, tau, weight = collect_data_with_dmp_weight(sym_robot, qi, qf, qdi, qdf, DMP_ORDER=25)

            temp_group = f.create_group("{}|{}|{}|{}".format(qi[0],qi[1],qf[0],qf[1]))
            temp_group.create_dataset("dt", data=dt)
            temp_group.create_dataset("t", data=t)
            temp_group.create_dataset("q", data=q)
            temp_group.create_dataset("qd", data=qd)
            temp_group.create_dataset("qdd", data=qdd)
            temp_group.create_dataset("weight", data=weight)
    f.close()

if __name__ == "__main__":
    # experiment_data()

    # start the test
    curr_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(curr_path)

    from learning.learning_utils import DMP_reconstruct
    from trajencoder.bspline.bspline import QuinticSpline

    def split(string):
        ''' split the string seperated by |
        '''
        return list(map(float, re.split(r'\|', string)))

    print(os.getcwd())
    f = h5py.File('encoder_test.hdf5', 'r')

    error_dmp_l, error_spl_l = [], []
    for index, label in enumerate(f.keys()):
        x = split(label)
        dmp_weight, length  = np.array(f[label]["weight"]), len(f[label]["t"])
        q, t =  np.array(f[label]["q"]), np.array(f[label]["t"])
        dof = 2
        res_q_dmp, t1, t2 = DMP_reconstruct(dmp_weight, 25, x[:dof], x[dof:], length, dof)

        # spline
        spl = QuinticSpline(knots=9)
        spl.get_weights_through_traj(t, q)
        res_q_spl = spl.trajectory(t[-1], 0.01)
        t_spl = np.linspace(0, 0.004*len(t), len(res_q_spl))

        # interpolate 
        from scipy import interpolate
        res_q_spl_inter = np.zeros_like(q)
        for i in range(dof):
            f_spl = interpolate.interp1d(t_spl, res_q_spl[:,i])
            res_q_spl_inter[:,i] = f_spl(t1)

        res_q_dmp_inter = np.zeros_like(q)
        for i in range(dof):
            f_dmp = interpolate.interp1d(t2, res_q_dmp[:,i])
            res_q_dmp_inter[:,i] = f_dmp(t1)
        
        # error
        error_spl = np.linalg.norm(res_q_spl_inter - q) / len(q)
        error_dmp = np.linalg.norm(res_q_dmp_inter - q) / len(q)
        if index == 223:
            plt.plot(t2, res_q_dmp, '-r', label='dmp')
            plt.plot(t1, q, '-b', label='true')
            plt.plot(t1, res_q_spl_inter, '-c', label='spline')
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            plt.grid()
            plt.legend(handles, labels, loc='best')
            plt.savefig('./worst_case.jpg', dpi=150)
        error_spl_l.append(error_spl)
        error_dmp_l.append(error_dmp)


print("The worst dmp encoding: ", np.argmax(error_dmp_l))
print("The total encoding error: ", sum(error_dmp_l))
print("The worst spl encoding: ", np.argmax(error_spl_l))
print("The total encoding error: ", sum(error_spl_l))
print("Worst case error dmp: {}, spl: {}".format(np.max(error_dmp_l), np.max(error_spl_l)))

plt.figure()
plt.plot(error_spl_l, '.r', label='spline error')
plt.plot(error_dmp_l, '.b', label='dmp error')
plt.legend()
plt.savefig('./error.jpg', dpi=150)