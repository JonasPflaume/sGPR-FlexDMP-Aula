
import matplotlib as mpl
import numpy as np
from learning_utils import spline_reconstruct_via, flexdmp_reconstruct
from casadi_kinodynamics.utils import symbolic_robot

rc_fonts = {
    "text.usetex": True,
    'mathtext.default': 'regular',
    'axes.labelsize': 23,'axes.titlesize':14, 'legend.fontsize': 20, 'xtick.labelsize': 22, 'ytick.labelsize': 22
}
font_size = 24
mpl.rcParams.update(rc_fonts)
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

def get_traj_tau(tau_func, q, dq, ddq):
    tau = []
    for i in range(len(q)):
        tau_i = tau_func(q[i,:],dq[i,:],ddq[i,:]).full()
        tau.append(tau_i.reshape(1,-1))
    return np.concatenate(tau)

def plot_res_6dof(q_max, q_min, dq_max, ddq_max, index, pred, Xv, Yv, Trueth, dq_trueth, ddq_trueth, tau_trueth, dmp3=None, DMP_ORDER=None, dof=6, dt=0.004):
    plt.figure(figsize=[15,13])
    color_map = {0:'brown', 1:'orange', 2:'dimgray', 3:'green', 4:'c', 5:'blue'}
    # figure 1
    plt.subplot(221)

    q_real = Trueth[index]
    PARAM = pred[index,0:-1].reshape(dof, DMP_ORDER)
    length = int(pred[index, -1])
    initial, goal = q_real[0], q_real[-1]
    q, dq, ddq = flexdmp_reconstruct(PARAM, initial, goal, length, DMP_ORDER, dof)
    t = np.linspace(0, len(q)*dt, len(q))
    
    # plt.plot([t[-1], t[-1]], [np.min(Trueth[index]), np.max(Trueth[index])], '-.b') # the indicator line for the ending of prediction
    time_vec = np.linspace(0, Yv[index,-1]*dt, int(Yv[index,-1])) # time vector of the ground trueth
    # plt.plot([time_vec[-1], time_vec[-1]], [np.min(Trueth[index]), np.max(Trueth[index])], '-.r')
    plt.ylabel(r"$\mathbf{q}\ [rad]$", fontsize=font_size)
    for i in range(dof):
        plt.plot(t, q[:,i], '-', color=color_map[i], label='Joint {}'.format(i+1))
        plt.plot(time_vec, Trueth[index][:,i], '--', color=color_map[i])
        # plt.plot(t, np.ones_like(t)*q_max[i], '-.', color=color_map[i], alpha=0.4)
        # plt.plot(t, np.ones_like(t)*q_min[i], '-.', color=color_map[i], alpha=0.4)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    # plt.xlabel("time [sec]",fontsize=font_size)
    handles = [handles[i] for i in ids]
    plt.grid()
    
    # figure 2, plot the velocity
    plt.subplot(222)
    for i in range(dof):
        plt.plot(t, dq[:,i], '-', color=color_map[i])
        plt.plot(time_vec, dq_trueth[index][:,i], '--', color=color_map[i])
        plt.plot(t, np.ones_like(t)*dq_max[i], '-.', color=color_map[i], alpha=0.4)
        plt.plot(t, -np.ones_like(t)*dq_max[i], '-.', color=color_map[i], alpha=0.4)
    plt.grid()
    dq_max_line = np.max([np.max(dq_trueth[index]), np.max(dq)])
    dq_min_line = np.min([np.min(dq_trueth[index]), np.min(dq)])
    
    plt.ylim(dq_max_line+1., dq_min_line-1.)
    plt.ylabel(r"$\dot{\mathbf{q}}\ [rad/s]$", fontsize=font_size)
    # plt.xlabel("time [sec]",fontsize=font_size)
    # figure 3, plot the acceleration
    plt.subplot(223)
    for i in range(dof):
        plt.plot(t, ddq[:,i], '-', color=color_map[i])
        plt.plot(time_vec, ddq_trueth[index][:,i], '--', color=color_map[i])
        plt.plot(t, np.ones_like(t)*ddq_max[i], '-.', color=color_map[i], alpha=0.4)
        plt.plot(t, -np.ones_like(t)*ddq_max[i], '-.', color=color_map[i], alpha=0.4)
    plt.grid()
    plt.ylabel(r"$\ddot{\mathbf{q}}\ [rad/s^2]$", fontsize=font_size)
    # plt.xlabel("time [sec]",fontsize=font_size)
    # figure 4, plot the dynamics
    plt.subplot(224)
    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                          tool_mass=Xv[index, -1],
                                          tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.0],
                                          tool_inertia=[
                                              0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)
    tau_func = sym_robot.inv_dyn
    tau_max = sym_robot.tau_max
    tau = get_traj_tau(tau_func, q, dq, ddq)
    for i in range(dof):
        plt.plot(t, 100* tau[:,i] / tau_max[i], '-', color=color_map[i])
        plt.plot(time_vec, 100* tau_trueth[index][:,i] / tau_max[i], '--', color=color_map[i])
    plt.plot(t, 100*np.ones_like(tau), '-.', color='red', alpha=0.4)
    plt.plot(t, -np.ones_like(tau)*100, '-.', color='red', alpha=0.4)
    plt.grid()
    legend1 = plt.legend(handles, labels, loc='best', ncol=1, bbox_to_anchor=(1.5,1.5), labelspacing=0.1, columnspacing=.4)
    plt.plot([], [], '--k', label="Optimal")
    plt.plot([], [], '-k', label="Prediction")
    plt.plot([], [], '-.k', label="Constraints")
    plt.legend(loc='best', ncol=1, bbox_to_anchor=(1.5,0.5), labelspacing=0.1, columnspacing=.5, framealpha=1)
    plt.gca().add_artist(legend1)
    plt.xlabel("time [sec]",fontsize=font_size)
    plt.ylabel(r"\boldmath$\tau$ utilization $[\%]$", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(r'/home/jiayun/MotionLearning/suboptimal_planner/plots/ordinary/{}_dmp_baseline.svg'.format(index), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from learning_utils import prepare_data_flexdmp, calc_loss_flexdmp
    from trajencoder.flexdmp.flexdmp import FlexDMP
    from casadi_kinodynamics.utils import symbolic_robot
    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                            tool_mass=4.,
                                            tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                            tool_cog=[0.0, 0., 0.0],
                                            tool_inertia=[
                                                0., 0., 0., 0., 0., 0.],
                                            load_casadi_fnc=True)
    DMP_ORDER = 30
    Xv, Yv, Trueth, qd_trueth, qdd_trueth, tau_trueth = prepare_data_flexdmp("suboptimal_planner/data/6dof/new_dynamics/box2box_varOri/Vali_1e-07_jerk_0.00_mass.hdf5", FlexDMP, DMP_ORDER=DMP_ORDER, \
                                                             return_dyddy=True, return_tau=True, dof=6)
    
    pred = np.load("suboptimal_planner/learning/6dof/pred.npy")
    
    for i in [13]:#peak_violation_index:
        plot_res_6dof(sym_robot.q_max, sym_robot.q_min, sym_robot.qd_max, sym_robot.qdd_max, i, pred, Xv, Yv, Trueth,\
                    qd_trueth, qdd_trueth, tau_trueth, dmp3=FlexDMP, DMP_ORDER=DMP_ORDER)