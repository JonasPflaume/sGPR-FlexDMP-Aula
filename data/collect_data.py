##
#  Sloppy implementation of data collector for 2 dof
##
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from toppysail.utils import util_functions as ut
from casadi_kinodynamics.utils import symbolic_robot
from toppysail.casadi_dev import nlps
from toppysail.casadi_dev import opt_problem_formulations

import h5py, os, itertools, sys
curr_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_path)
sys.path.append('..')
from trajencoder.dmp.dmp import DMP
from trajencoder.bspline.bspline import CubicSpline

## script built the 2 dof data collector

def collect_data(sym_robot, qi, qf, qdi, qdf, N=70, jerk=1e-6, print_level=0):
    # generate a problem
    problem = opt_problem_formulations.problem_136b_X_q_qd_dt_U_qdd(sym_robot)

    nlp_params = nlps.Parameters(N=N, cost_coeff_jerk=jerk)

    # generate nlp from the problem and solve it
    sol = nlps.generate_and_solve_nlp(
        problem, qi=qi, qf=qf, qdi=qdi, qdf=qdf, p=nlp_params, solver_print_level=print_level)

    # interpolate the optimal solution
    sol_intrp = problem.interpolate_solution(sol, dt_resample=0.004)
    ut.print_solution_info(sol)

    # plots
    # ut.plot_joints2(solution=sol, plots=['q', 'qd', 'qdd', 'tau'],
    #                 solution_interp=sol_intrp, robot=sym_robot)

    # anim = ut.plot_3D(q=sol['q'], robot=sym_robot,  animate=True,
    #                 show_movej=False, plot_tool_frames=True)
    # plt.show()
    return sol_intrp["dt"], sol_intrp["t"], sol_intrp["q"], sol_intrp["qd"], sol_intrp["qdd"], sol_intrp["tau"]

def collect_data_with_dmp_weight(sym_robot, qi, qf, qdi, qdf, DMP_ORDER=25):
    """ The func for the active GPR data request
    """
    dt, t, q, qd, qdd, tau = collect_data(sym_robot, qi, qf, qdi, qdf)
    dmp_encoder = DMP(DMP_ORDER, demo_len=len(q), demo_dt=dt)
    weight_q1 = dmp_encoder.get_weights(q[:,0], dy_demo=qd[:,0], ddy_demo=qdd[:,0])
    weight_q2 = dmp_encoder.get_weights(q[:,1], dy_demo=qd[:,1], ddy_demo=qdd[:,1])
    weight = np.concatenate([weight_q1.reshape(-1,1), weight_q2.reshape(-1,1)], axis=1)

    return dt, t, q, qd, qdd, tau, weight

def collect_data_with_bspline_weight(sym_robot, qi, qf, qdi, qdf, SPL_ORDER=6):
    """ The func for the active GPR data request
    """
    dt, t, q, qd, qdd, tau = collect_data(sym_robot, qi, qf, qdi, qdf)
    bs = CubicSpline(knots=SPL_ORDER)
    weight = bs.get_weights_through_traj(t, q)

    return dt, t, q, qd, qdd, tau, weight


if __name__ == "__main__":
    
    """ To collect the data as batch with .H5 format
    """
    WHICH_WEIGHT = 'bspline' # ‘bspline‘ or dmp
    TRAINING = True
    ORDER = 9
    D = 8
    end_load = 3.00

    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB910_2r', tool_mass=end_load,
                                                tool_frame=[0., 0., 0.3, 0., 0., 0.], load_casadi_fnc=False)
    q_min, q_max = sym_robot.q_min, sym_robot.q_max
    qd_min, qd_max = np.zeros(2), sym_robot.qd_max

    if TRAINING:
        f = h5py.File(os.path.join(curr_path, "WithLoad/training_D{0}_L{1}_backoff17.hdf5".format(D, end_load)), "w")
        range_q1 = np.linspace(q_min[0], q_max[0], D)
        range_q2 = np.linspace(q_min[1], q_max[1], D)
    else:
        f = h5py.File(os.path.join(curr_path,"WithLoad/validating.hdf5"), "w")
        range_q1 = np.linspace(q_min[0], q_max[0], D)
        range_q2 = np.linspace(q_min[1], q_max[1], D)

    range_q1_list = [q1_q1 for q1_q1 in itertools.product(range_q1, range_q1) if q1_q1[0] != q1_q1[1]]
    range_q2_list = [q2_q2 for q2_q2 in itertools.product(range_q2, range_q2) if q2_q2[0] != q2_q2[1]]
    range_q_list = [q_q for q_q in itertools.product(range_q1_list, range_q2_list)]

    for q in range_q_list:
        q1, q2 = q[0], q[1]
        qi = [i for i in [q1[0], q2[0]]]
        qf = [i for i in [q1[1], q2[1]]]
        qdi = [i for i in [0., 0.]]
        qdf = [i for i in [0., 0.]]
        if not TRAINING:
            sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB910_2r', tool_mass=np.random.uniform(0., 3.),
                                                tool_frame=[0., 0., 0.3, 0., 0., 0.], load_casadi_fnc=False)
        if WHICH_WEIGHT == 'dmp':
            dt, t, q, qd, qdd, tau, weight = collect_data_with_dmp_weight(sym_robot, qi, qf, qdi, qdf, DMP_ORDER=ORDER)
        elif WHICH_WEIGHT == 'bspline':
            dt, t, q, qd, qdd, tau, weight = collect_data_with_bspline_weight(sym_robot, qi, qf, qdi, qdf, SPL_ORDER=ORDER)

        temp_group = f.create_group("{}|{}|{}|{}".format(qi[0],qi[1],qf[0],qf[1]))
        temp_group.create_dataset("dt", data=dt)
        temp_group.create_dataset("t", data=t)
        temp_group.create_dataset("q", data=q)
        temp_group.create_dataset("qd", data=qd)
        temp_group.create_dataset("qdd", data=qdd)
        temp_group.create_dataset("weight", data=weight)
        temp_group.create_dataset("mass", data=sym_robot.tool_mass)
    f.close()