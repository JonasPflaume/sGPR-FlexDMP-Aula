##
#   Well orginized data collector for 6 dof scenario
##
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from toppysail.utils import util_functions as ut
from casadi_kinodynamics.utils import symbolic_robot
from toppysail.casadi_dev import nlps
from toppysail.casadi_dev import opt_problem_formulations

import h5py, os, itertools, sys, collections
import multiprocessing
curr_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_path)
sys.path.append('..')

trajectory = collections.namedtuple("trajectory", "dt mass t q qd qdd tau")

class Env6dof:
    ''' A data collector in an active learning loop or
        collecting batch data into a local file and 
        a wrapper of some helper functions
    '''
    def __init__(self, N=20, jerk=1e-7, dt=0.004):
        ''' Input:
                N:          The number of waypoints in the optimization problem
                jerk:       The coefficient term of the jerk regularization in opt problem
                print_level:The print level of the casadi solver
                dt:         Interpolating time interval
        '''
        self.N = N
        self.jerk = jerk # defined but not used!
        self.dt = dt
        self.dof = 6

    @staticmethod
    def step(qi, qf, mass, N, dt, jerk, return_dict=None, ProcNum=None):
        ''' One step trajectory generation
            Static function to ensure the multiprocessing works!
            Input:
                qi:         initial q list
                qf:         final q list
                mass:       end effector load float
                return_dict:The multiprocessing data container
                ProcNum:    The current process number
            output:
                onetraj:    one namedtuple trajectory
                or:
                None, just full the multiprocess container

        '''
        sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                          tool_mass=mass,
                                          tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.],
                                          tool_inertia=[0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)

        problem = opt_problem_formulations.problem_136b_X_q_qd_dt_U_qdd(sym_robot)
        N_variable = N
        success_solved = False
        while not success_solved:
            try:
                nlp_params = nlps.Parameters(N=N_variable, cost_coeff_jerk=jerk)

                sol = nlps.generate_and_solve_nlp(problem, qi=qi, qf=qf,  p=nlp_params, solver_print_level=0)
                # interpolate the optimal solution
                sol_intrp = problem.interpolate_solution(sol, dt_resample=dt)
            except:
                N_variable -= 5 # large N will influence the convergence
            else:
                success_solved = True

        onetraj = trajectory(dt=sol_intrp["dt"], mass=mass, t=sol_intrp["t"], q=sol_intrp["q"],
                            qd=sol_intrp["qd"], qdd=sol_intrp["qdd"], tau=sol_intrp["tau"])
        
        if return_dict != None and ProcNum != None:
            return_dict[ProcNum] = onetraj
        else:
            return onetraj

    @staticmethod
    def prepare_data_flexdmp(traj_list, FlexDMP, DMP_ORDER, dt=0.004, dof=6):
        ''' generate data for dmp3
            dmp3:
                    the dmp3 class
        '''
        Y = []
        for traj in traj_list:
            t, y, dy, ddy, dt = traj.t, traj.q, traj.qd, traj.qdd, traj.dt
            dmp = FlexDMP(DMP_ORDER, len(t), dt, dof=dof)
            yi = dmp.get_weights(y, dy_demo=dy, ddy_demo=ddy)
            yi = np.append(yi.reshape(-1,), float(len(t)))
            Y.append(yi)
        Y = np.array(Y, dtype=float)
        return Y

    def batch_record(self, Qi, Qf, Mass, filename, omit_qi=True):
        ''' The trajectories batch recorder
            Input:
                        Qi:         The start q LIST
                        Qf:         The end q LIST
                        Mass:       The load mass LIST
                        filename:   What's the name of h5 file
                        omit_qi:    Left for the fix beginning position
        '''
        num = len(Qi)
        f = h5py.File(os.path.join(curr_path, "6dof/{}.hdf5".format(filename)), "w")
        assert len(Qf) == len(Mass) == num, "Those lists should have the same length"
        for i in range(num):
            qi, qf, mass = Qi[i], Qf[i], Mass[i] # qi, qf should be list with 6 elements
            if omit_qi:
                x = np.array(qf+[mass])
            else:
                x = np.array(qi+qf+[mass])
            temp_traj = Env6dof.step(qi, qf, mass, self.N, self.dt, self.jerk)
            temp_group = f.create_group("{}".format(i))
            temp_group.create_dataset("x", data=x)
            temp_group.create_dataset("dt", data=temp_traj.dt)
            temp_group.create_dataset("t", data=temp_traj.t)
            temp_group.create_dataset("q", data=temp_traj.q)
            temp_group.create_dataset("qd", data=temp_traj.qd)
            temp_group.create_dataset("qdd", data=temp_traj.qdd)
            temp_group.create_dataset("tau", data=temp_traj.tau)
        f.close()

    def batch_record_multiprocessing(self, Qi, Qf, Mass, filename='hello', omit_qi=False, check_step=True):
        ''' The function was designed to run the self.step function in multiple process
            Input:
                    check_step:     check that we can finish the collection in one multiprocessing step
                                    And we don't need to save the trajectories in file
        '''
        num = len(Qi)
        nproc = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        assert num == len(Qf) == len(Mass), "Please align the dataset!"
        if check_step:
            traj_list = []
            X = []
            assert num <= nproc, "Please check the batch size! It might be larger than the maximum core of cpu, when you use one step data collection!"
            for ProcNum in range(nproc):
                qi, qf, mass, N, dt, jerk = Qi[ProcNum], Qf[ProcNum], Mass[ProcNum], self.N, self.dt, self.jerk
                if omit_qi:
                    x = np.array(qf + [mass])
                else:
                    x = np.array(qi + qf + [mass])
                X.append(x)
                exec("P{0} = multiprocessing.Process(target=Env6dof.step, args=(qi, qf, mass, N, dt, jerk, return_dict, ProcNum))".format(ProcNum))

            for ProcNum in range(nproc):
                exec("P{0}.start()".format(ProcNum))
            
            for ProcNum in range(nproc):
                exec("P{0}.join()".format(ProcNum))
                
            for ProcNum in range(nproc):
                traj_list.append(return_dict[ProcNum])
            return np.array(X), traj_list

        else:
            batch_num = num // nproc
            last_batch_num = num % nproc
            f = h5py.File(os.path.join(curr_path, "6dof/{}.hdf5".format(filename)), "w")

            for i_batch in range(batch_num):
                traj_list = []
                X = []
                Qi_nproc, Qf_nproc, Mass_nproc, N, dt, jerk = Qi[i_batch*nproc:(i_batch+1)*nproc], Qf[i_batch*nproc:(i_batch+1)*nproc],\
                                                    Mass[i_batch*nproc:(i_batch+1)*nproc], self.N, self.dt, self.jerk
                for ProcNum in range(nproc):
                    qi, qf, mass = Qi_nproc[ProcNum], Qf_nproc[ProcNum], Mass_nproc[ProcNum]
                    if omit_qi:
                        x = np.array(qf + [mass])
                    else:
                        x = np.array(qi + qf + [mass])
                    X.append(x)
                    exec("P{0} = multiprocessing.Process(target=Env6dof.step, args=(qi, qf, mass, N, dt, jerk, return_dict, ProcNum))".format(ProcNum))

                for ProcNum in range(nproc):
                    exec("P{0}.start()".format(ProcNum))
                
                for ProcNum in range(nproc):
                    exec("P{0}.join()".format(ProcNum))
                    
                for ProcNum in range(nproc):
                    traj_list.append(return_dict[ProcNum])

                # record the generated batch
                for ProcNum in range(nproc): 
                    temp_group = f.create_group("{}".format(i_batch*nproc+ProcNum))
                    temp_group.create_dataset("x", data=X[ProcNum])
                    temp_group.create_dataset("dt", data=traj_list[ProcNum].dt)
                    temp_group.create_dataset("t", data=traj_list[ProcNum].t)
                    temp_group.create_dataset("q", data=traj_list[ProcNum].q)
                    temp_group.create_dataset("qd", data=traj_list[ProcNum].qd)
                    temp_group.create_dataset("qdd", data=traj_list[ProcNum].qdd)
                    temp_group.create_dataset("tau", data=traj_list[ProcNum].tau)
                # clear the multiprocessing container
                return_dict.clear()
            # collect the rest trajectories
            if last_batch_num != 0:
                traj_list = []
                X = []
                Qi_nproc, Qf_nproc, Mass_nproc, N, dt, jerk = Qi[batch_num*nproc:batch_num*nproc+last_batch_num], Qf[batch_num*nproc:batch_num*nproc+last_batch_num],\
                                                    Mass[batch_num*nproc:batch_num*nproc+last_batch_num], self.N, self.dt, self.jerk
                for ProcNum in range(last_batch_num):
                    qi, qf, mass = Qi_nproc[ProcNum], Qf_nproc[ProcNum], Mass_nproc[ProcNum]
                    if omit_qi:
                        x = np.array(qf + [mass])
                    else:
                        x = np.array(qi + qf + [mass])
                    X.append(x)
                    exec("P{0} = multiprocessing.Process(target=Env6dof.step, args=(qi, qf, mass, N, dt, jerk, return_dict, ProcNum))".format(ProcNum))

                for ProcNum in range(last_batch_num):
                    exec("P{0}.start()".format(ProcNum))
                
                for ProcNum in range(last_batch_num):
                    exec("P{0}.join()".format(ProcNum))
                    
                for ProcNum in range(last_batch_num):
                    traj_list.append(return_dict[ProcNum])

                # record the generated batch
                for ProcNum in range(last_batch_num): 
                    temp_group = f.create_group("{}".format(batch_num*nproc+ProcNum))
                    temp_group.create_dataset("x", data=X[ProcNum])
                    temp_group.create_dataset("dt", data=traj_list[ProcNum].dt)
                    temp_group.create_dataset("t", data=traj_list[ProcNum].t)
                    temp_group.create_dataset("q", data=traj_list[ProcNum].q)
                    temp_group.create_dataset("qd", data=traj_list[ProcNum].qd)
                    temp_group.create_dataset("qdd", data=traj_list[ProcNum].qdd)
                    temp_group.create_dataset("tau", data=traj_list[ProcNum].tau)
                # clear the multiprocessing container
                return_dict.clear()
            f.close()

    def plot_cart_path(self, q, mass=0.5):
        '''
        We only need the via points q
        '''
        from toppysail.utils import util_pyvista as utpv

        sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                          tool_mass=mass,
                                          tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.],
                                          tool_inertia=[0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)
        plotter = utpv.pv_plot_robot(
            sym_rob=sym_robot, q=q,  plotter=None)
        # plot solution points
        utpv.plot_cart_path(q, sym_robot, plotter=plotter, tube_radius=0.002,
                            plot_line=True, plot_sphere_markers=False, marker_color='red')

        plotter.show(cpos=[(-0.5, -4, 0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)])

### inverse kinematics module
from cfree_client_utils import *
from cfree_types import *
from cfree_client import CFreePyClient
from scipy.spatial.transform import Rotation
import os.path
import subprocess
def get_cartesian_position_sample_box(cfree_client, cart_start, box_length_xyz, n_samples_xyz, orientation):
    ''' copy from toppysail example
    '''
    # container
    q_all = []
    ##
    # ------------------------------- planning ------------------------------

    xyz_start = cart_start
    points_l = []
    for x in np.linspace(xyz_start[0], xyz_start[0]+box_length_xyz[0], n_samples_xyz[0]):
        for y in np.linspace(xyz_start[1], xyz_start[1]+box_length_xyz[1], n_samples_xyz[1]):
            for z in np.linspace(xyz_start[2], xyz_start[2]+box_length_xyz[2], n_samples_xyz[2]):
                points_l.append([x, y, z])

    for points in points_l:  # points should be 3 elements list
        start_robtarg = RobTarget(conf_i=0, transform=Transform(
            points, orientation))

        start_joint_targ = cfree_client.inverse_kinematics(0, start_robtarg)
        if start_joint_targ != None:
            q_all.append(start_joint_targ.robax)
    return q_all

def get_cylinder(   
                    orientation, 
                    r_start_r_end=  np.linspace(0.1, 0.35, 6),
                    height=         np.linspace(0.02, 0.32, 7),
                    r_density=      0.05
                ):
    ''' copy from toppysail example
        r_start_r_end:          e.g. [0.15, 0.25, 0.35]
        height:                 e.g. [0.05, 0.1, 0.15, 0.2]
        r_density:              e.g. 0.1
    '''
    ## container
    q_all = []
    ##
    # ------------------------------- planning ------------------------------
    points_l = []
    for h in height:
        for r in r_start_r_end:
            for theta in np.linspace(0., 2*np.pi, int( 2*np.pi*r//r_density) ):
                point = [np.sin(theta)*r, np.cos(theta)*r, h]
                points_l.append(point)

    # from matplotlib import pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for point in points_l:
    #     ax.scatter(point[0], point[1], point[2], c='red', s=10)
    # plt.show()
    
    path_to_cfree_executable = r'C:\Users\DEJILI\code\MotionLearning\cfree\CFreeServer.1.9.172.10.Internal\bin\windows-x64\cfree_server.exe'
    # path_to_cfree_executable = r'C:\Users\DENIENA\code\CFree\CFree\cfree_src\out\build\x64-Release\src\cfree\grpc_server\cfree_server.exe'

    ip_and_port = 'localhost:50051'
    tool_tcp_pos = [0., 0.0, 0.25]

    # run the server executable
    if not os.path.isfile(path_to_cfree_executable):
        print("No file found at " + path_to_cfree_executable)
        print("Correct the CFree server executable file path or disable run_cfree_server and run the server externally.")
        exit()
    server_process = subprocess.Popen(
        [path_to_cfree_executable, ip_and_port])

    # start and use the python client
    with CFreePyClient(ip_and_port) as client:

        # ------------------------------- cell setup ------------------------------
        client.add_robot(robot_model='IRB1100_4_058', num_planning_threads=1)

        client.add_robot_tool_geometry(robot_id=0,
                                    is_convex=0,
                                    tcp_transform=Transform(tool_tcp_pos))
        plot_coordinate_frames = True
        # load the robot
        sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                                tool_mass=0.0,
                                                tool_frame=tool_tcp_pos +
                                                [0., 0., 0.],
                                                tool_cog=[0.0, 0., 0.],
                                                tool_inertia=[
                                                    0., 0., 0., 0., 0., 0.],
                                                load_casadi_fnc=True)
        for points in points_l:  # points should be 3 elements list
            start_robtarg = RobTarget(conf_i=0, transform=Transform(
                points, orientation))

            start_joint_targ = client.inverse_kinematics(0, start_robtarg)
            if start_joint_targ != None:
                q_all.append(start_joint_targ.robax)
    return q_all

def test():
    ''' simple test env
    '''
    env = Env6dof()
    qi = [i/180.*np.pi for i in [-80., 80., -60., 0., 60., 0.]]
    qf = [i/180.*np.pi for i in [120., 80., -60., 0., 60., 0.]]
    mass = 2.
    qi_ = qi
    qi_[1] -= 0.1
    qf_ = qf
    qf_[1] -= 0.1
    Qi = [qi for _ in range(6)] + [qi_ for _ in range(7)]
    Qf = [qf for _ in range(6)] + [qf_ for _ in range(7)]
    Mass = [mass for _ in range(13)]
    env.batch_record_multiprocessing(Qi, Qf, Mass, check_step=False)

    f = h5py.File('./6dof/hello.hdf5', 'r')
    assert len(f.keys()) == 13
    for i in range(13):
        assert str(i) in f.keys()

    print("Pass the test...")

def create_data_from_X(file_path="./X.npy", mass=None, jerk=1e-7):
    if type(file_path) == np.ndarray:
        X = file_path
    elif type(file_path) == str:
        X = np.load(file_path)
    if X.shape[1] == 12:
        X = np.concatenate( [X, mass * np.ones([X.shape[0], 1])], axis=1 )
    Qi, Qf, Mass = X[:, :6], X[:, 6:12], X[:, 12]
    rs = np.abs(Qi - Qf).sum(axis=1)
    index = rs > 0.
    X_filtered = X[index]
    Qi, Qf, Mass = X_filtered[:, :6].tolist(), X_filtered[:, 6:12].tolist(), X_filtered[:, 12].tolist()
    env = Env6dof(N=30, jerk=jerk)
    env.batch_record_multiprocessing(Qi, Qf, Mass, filename="Training_{0}_jerk_{1:.2f}_mass".format(jerk, mass), omit_qi=False, check_step=False)


if __name__ == "__main__":
    # pass
    # import h5py
    # F = h5py.File("6dof/new_dynamics/box2box_fixedOri/normal_data/Training_1e-07_jerk_0.0_mass.hdf5")
    # X = []
    # for key in F.keys():
    #     xi = np.array(F[key]["x"]).reshape(1,-1)
    #     X.append(xi[:,:12])
    # X = np.concatenate(X)
    # print(X.shape)
    # np.save("/home/jiayun/Desktop/X_fixedOri.npy", X)
    # for jerk in [1e-7]:
    #     for mass in np.linspace(3.,4.,3):

    #         create_data_from_X(X, mass, jerk=jerk)
    
    from itertools import product
    box1 = np.load("6dof/box_oppo/box1_pool.npy")
    index1 = np.load("6dof/box_oppo/box1_picked_index.npy")[:70]
    # index1 = np.random.choice( np.arange(len(box1)), 15)
    box2 = np.load("6dof/box_oppo/box2_pool.npy")
    index2 = np.load("6dof/box_oppo/box2_picked_index.npy")[:70]
    # index2 = np.random.choice( np.arange(len(box2)), 15)
    box1, box2 = box1[index1], box2[index2]
    b12b2 = np.array( list( product(box1, box2))).reshape(-1,12)
    X = b12b2
    print(X.shape)
    for mass in np.linspace(0.,4.,9):
        # X = []
        # for i in range(225):
        #     index1, index2 = np.random.randint(len(box1)), np.random.randint(len(box2))
        #     qi = box1[index1]
        #     qf = box2[index2]
        #     # from itertools import product
        #     X.append(np.concatenate([qi, qf]).reshape(1,-1))
        # # b12b2 = np.array( list( product(box1, box2))).reshape(-1,12)
        # X = np.concatenate(X)
        # print(X.shape)

        # for mass in np.linspace(1.5,4.,6):
        create_data_from_X(X, mass, 1e-7)