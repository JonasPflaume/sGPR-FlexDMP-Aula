from casadi_kinodynamics.utils import symbolic_robot
from pathlib import Path
from sys import path
import numpy as np
from toppysail.utils import util_functions as ut
from matplotlib import pyplot as plt
from toppysail.utils import ruckig_traj_gen

# Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
build_path = Path(__file__).parent.absolute().parent / 'build'
path.insert(0, str(build_path))


if __name__ == '__main__':
    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                              tool_mass=4.0,
                                              tool_frame=[
                                                  0., 0., 0.25, 0., 0., 0.],
                                              tool_cog=[0.0, 0., 0.12],
                                              tool_inertia=[
                                                  0., 0., 0., 0., 0., 0.],
                                              load_casadi_fnc=True)

    test_joint = 3

    if test_joint == 1:
        # Joint 1 max acc estimation
        qi = [i/180.*np.pi for i in [-90., 90., -90., 0., 0., 0.]]
        qf = [i/180.*np.pi for i in [90., 90., -90., 0., 0., 0.]]

    elif test_joint == 2:
        # Joint 2 max acc estimation
        qi = [i/180.*np.pi for i in [0., 90., -90, 0., 0., 0.]]
        qf = [i/180.*np.pi for i in [0., -90, -90, 0., 0., 0.]]

    elif test_joint == 3:
        # Joint 3 max acc estimation
        qi = [i/180.*np.pi for i in [0., 0, 50., 0., 0., 0.]]
        qf = [i/180.*np.pi for i in [0., 0., -130., 0., 0., 0.]]

    elif test_joint == 4:
        # # Joint 4 max acc estimation
        qi = [i/180.*np.pi for i in [0., 0, 0., -90, 90., 0.]]
        qf = [i/180.*np.pi for i in [0., 0., 0., 90, 90., 0.]]

    elif test_joint == 5:
        # Joint 5 max acc estimation
        qi = [i/180.*np.pi for i in [0., 0, 0., 0, -90., 0.]]
        qf = [i/180.*np.pi for i in [0., 0., 0., 0, 90., 0.]]

    elif test_joint == 6:
        # Joint 6 max acc estimation
        qi = [i/180.*np.pi for i in [0., 0, 0., 0, 0., -90]]
        qf = [i/180.*np.pi for i in [0., 0., 0., 0, 0., 90]]

    qd_max = sym_robot.qd_max
    qdd_max = [23., 10, 13, 80, 62, 92]
    qddd_max = [4000.] * 6
    ruck = ruckig_traj_gen.ruckig_traj_gen(qd_max=sym_robot.qd_max,
                                           qdd_max=qdd_max, qddd_max=qddd_max)

    # Calculate the trajectory in an offline manner
    import time
    s = time.time()
    for _ in range(1000):
        trajectory = ruck.get_jerk_limited_traj(
            qi, qf, [0]*6, [0]*6, [0]*6, [0]*6)
    dt_compute = (time.time() - s)/1000
    print(f'Compute duration: {(dt_compute*1000000):0.1f} [us]')
    print(f'Trajectory duration: {(trajectory.duration*1000):0.1f} [ms]')

    dt = 0.004
    sol = ruck.get_interpolated_solution(trajectory, sym_robot, dt)
    sol['solver_stats']['t_proc_total'] = dt_compute

    ut.print_qd_tau_violation_info(
        sol, sym_robot, torque_lim_type='function')

    ut.plot_joints2(solution=sol, plots=['q', 'qd', 'qdd', 'tau'],
                    robot=sym_robot)
    plt.show()
