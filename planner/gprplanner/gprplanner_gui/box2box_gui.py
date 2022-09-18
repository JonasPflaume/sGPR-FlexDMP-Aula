from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
from gprplanner.ppgprplanner import PPGPRPlanner
from algpr.kernels import RBF

from matplotlib.backend_bases import MouseButton
from pyrsistent import v
# inverse kinematics
from cfree_client_utils import *
from cfree_types import *
from cfree_client import CFreePyClient
from scipy.spatial.transform import Rotation
import os.path
import subprocess
from casadi_kinodynamics.utils import symbolic_robot
import time

import casadi as cs
from casadi_kinodynamics.utils import symbolic_robot
from toppysail.casadi_dev import nlps
from toppysail.casadi_dev import opt_problem_formulations

class Box2BoxGui:
    def __init__(self, model_path, path_to_cfree_executable):
        '''
            the wrapper of the gpr planner, including the inverse kinematics and a rough gui
        '''
        kernel = RBF(l=[0.4]*13, anisotropic=True)
        self.planner = PPGPRPlanner(model_addr=model_path, ker=kernel, dof=6)
        self.path_to_cfree_executable = path_to_cfree_executable

    def __init_IK_server(self, mass):
        tool_tcp_pos = [0., 0.0, 0.25]
        self.sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                            tool_mass=mass,
                                            tool_frame=tool_tcp_pos +
                                            [0., 0., 0.],
                                            tool_cog=[0.0, 0., 0.],
                                            tool_inertia=[
                                                0., 0., 0., 0., 0., 0.],
                                            load_casadi_fnc=True)
        if not os.path.isfile(self.path_to_cfree_executable):
            print("No file found at " + self.path_to_cfree_executable)
            print("Correct the CFree server executable file path or disable run_cfree_server and run the server externally.")
            exit()
        ip_and_port = 'localhost:50051'
        server_process = subprocess.Popen(
            [self.path_to_cfree_executable, ip_and_port])

        self.client = CFreePyClient(ip_and_port)
        self.client.add_robot(robot_model='IRB1100_4_058', num_planning_threads=1)

        self.client.add_robot_tool_geometry(robot_id=0,
                                    is_convex=0,
                                    tcp_transform=Transform(tool_tcp_pos))

    def start(self, pi_input, zf, mass, run_traj_call_back, get_current_robot_pos, opt_solver=False):
        ''' Input:
                    run_traj_call_back: Take the q, dq, ddq and run the trajs
                    pi_input:           initial position
                    zf:                 the fixed z coordinate of the next position
                    mass:               the mass of the payload
                    which_planner:      using this variable can switch the planner
                    plot_trajectory:    when true, the planned trajectories will be plotted 
        '''
        self.__init_IK_server(mass)
        # cart_begin_1 = [0.25, -0.1, 0.05]
        # box_length_1 = [0.2, 0.35, 0.22]
        # cart_begin_2 = [-0.2, -0.45, 0.05]
        # box_length_2 = [0.35, 0.2, 0.22]
        cart_begin_1 = [0.3, -0.1, 0.05]
        box_length_1 = [0.2, 0.2, 0.22]
        cart_begin_2 = [-0.1, -0.45, 0.05]
        box_length_2 = [0.2, 0.2, 0.22]

        # assert (pi[2] <= cart_begin_1[2] + box_length_1[2]), "z1 should be less than or equal to {}".format(cart_begin_1[2] + box_length_1[2])
        # assert (pi[2] >= cart_begin_1[2]), "z1 should be greater than or equal to {}".format(cart_begin_2[2])
        # assert (zf <= cart_begin_2[2] + box_length_2[2]), "z2 should be less than or equal to {}".format(cart_begin_2[2] + box_length_2[2])
        # assert (zf >= cart_begin_2[2]), "z2 should be greater than or equal to {}".format(cart_begin_2[2])

        global start_with_box1
        start_with_box1 = False
        global marker_position_l
        marker_position_l = [[pi_input[0], pi_input[1]]]
        fig, ax = plt.subplots()

        global pf
        global pi
        pi = pi_input
        pf = [0.,0.]
        plt.ion()

        def background():
            rectangle1 = plt.Rectangle((cart_begin_1[0],cart_begin_1[1]), box_length_1[0], box_length_1[1], fc='red', ec="blue")
            plt.gca().add_patch(rectangle1)
            ax.annotate("Box1", (cart_begin_1[0]+0.5*box_length_1[0],cart_begin_1[1]+0.5*box_length_1[1]), color='w', weight='bold', 
                    fontsize=14, ha='center', va='center')

            rectangle2 = plt.Rectangle((cart_begin_2[0], cart_begin_2[1]), box_length_2[0], box_length_2[1], fc='blue', ec="red")
            plt.gca().add_patch(rectangle2)
            ax.annotate("Box2", (cart_begin_2[0]+0.5*box_length_2[0],cart_begin_2[1]+0.5*box_length_2[1]), color='w', weight='bold', 
                    fontsize=14, ha='center', va='center')

            robot = plt.Circle((0., 0.), radius=0.07, fc='c', ec='y')
            ax.annotate("Robot", (0.,0.), color='w', weight='bold', 
                    fontsize=14, ha='center', va='center')
            plt.gca().add_patch(robot)
            for marker_position_i in marker_position_l:
                p1_marker = plt.Circle((marker_position_i[0], marker_position_i[1]), radius=0.01, fc='c', ec='y')
                plt.gca().add_patch(p1_marker)
            plt.axis("scaled")

        def on_move(event):
            # get the x and y pixel coords
            global pf
            x, y = event.x, event.y
            if event.inaxes:
                ax = event.inaxes  # the axes instance
                pf[0] = event.xdata
                pf[1] = event.ydata
        
        def check_outside_box(p):
            if start_with_box1:
                judge1 = p[0] > (cart_begin_1[0] + box_length_1[0])
                judge2 = p[0] < cart_begin_1[0]
                judge3 = p[1] > (cart_begin_1[1] + box_length_1[1])
                judge4 = p[1] < cart_begin_1[1]
                return (judge1 or judge2 or judge3 or judge4)
            else:
                judge1 = p[0] > (cart_begin_2[0] + box_length_2[0])
                judge2 = p[0] < cart_begin_2[0]
                judge3 = p[1] > (cart_begin_2[1] + box_length_2[1])
                judge4 = p[1] < cart_begin_2[1]
                return (judge1 or judge2 or judge3 or judge4)

        def predict_traj(pf, mass):
            p_1 = pi
            p_2 = pf + [zf]
            #
            qi, qf = self.__inverse_kinematics(p_1, p_2, mass)
            print(f"qf =  {np.round( np.array(qf)*180/np.pi, 2)}")

            start = time.time()
            if opt_solver:
                q, dq, ddq = self.opt_solver(qi, qf, mass)
            else:
                if not start_with_box1:
                    q, dq, ddq = self.planner(qi, qf, mass)
                ###### Reverse trajectory here !!! TODO remove it #########
                else:
                    q, dq, ddq = self.planner(qf, qi, mass)
                    q, dq, ddq = np.flip(q, 0), np.flip(dq, 0), np.flip(ddq, 0)
                    dq, ddq = -1*dq, ddq
            ###########################################################

            print("Calculating time: ", time.time()-start, " Trajectory time: ", len(q)*0.004)
            return q, dq, ddq, p_1, p_2

        def render_traj(q, dq, ddq, p_1, p_2):
            X = []
            for i in range(len(q)):
                xi = self.sym_robot.for_kine(q[i])
                X.append(xi[-1][:3,-1].full().reshape(1,3))
            X = np.concatenate(X)
            ax.clear()
            
            background()
            plt.plot(X[:,0], X[:,1], "-.g")
            dx = X[len(X)//2 + 1, 0] - X[len(X)//2, 0]
            dy = X[len(X)//2 + 1, 1] - X[len(X)//2, 1]
            plt.arrow(X[len(X)//2, 0], X[len(X)//2, 1], dx, dy, shape='full', lw=0, color="g", length_includes_head=True, head_width=.03)
            ax.annotate("qi: {0:.3f}, {1:.3f}, qf: {2:.3f}, {3:.3f}".format(p_1[0],p_1[1],p_2[0],p_2[1]), (0.08, 0.1), color='k', weight='bold', 
                fontsize=9, ha='center', va='center')

        def on_click(event): # main loop
            global marker_position_l
            global start_with_box1
            global pi
            global binding_id_move
            global binding_id_click
            if event.button is MouseButton.LEFT:
                outsidebox_indicator = check_outside_box(pf)
                if outsidebox_indicator:
                    print("Choose inside the start box please!")
                    return
                else:
                    # we already get the pf here... pf is a global variable
                    
                    ### get current pos callback
                    pi = get_current_robot_pos()
                    ###
                    # stop the mouse serves
                    plt.disconnect(binding_id_move)
                    plt.disconnect(binding_id_click)
                    ax.clear()
                    marker_position_l.append([pf[0], pf[1]])
                    q, dq, ddq, p_1, p_2 =  predict_traj(pf, mass)
                    render_traj(q, dq, ddq, p_1, p_2)
                    # update plot
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    ### run trajetory call back
                    run_traj_call_back(q, dq)
                    ###
                    start_with_box1 = not start_with_box1

                    # start the mouse serves again
                    binding_id_move = plt.gcf().canvas.mpl_connect('motion_notify_event', on_move)
                    binding_id_click = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

        background()
        global binding_id_move, binding_id_click
        binding_id_move = plt.gcf().canvas.mpl_connect('motion_notify_event', on_move)
        binding_id_click = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

        while True:
            plt.pause(0.01)

    def __inverse_kinematics(self, p1, p2, mass):
        orient1 = Rotation.from_euler('y', 180, degrees=True)
        orient2 = Rotation.from_euler('y', 180, degrees=True)

        print(f"Start position = {p1}")
        print(f"Goal position = {p2}")
        start_robtarg = RobTarget(conf_i=0, transform=Transform(
                p1, orient1))
        qi = self.client.inverse_kinematics(0, start_robtarg).robax

        start_robtarg = RobTarget(conf_i=0, transform=Transform(
                p2, orient2))
        qf = self.client.inverse_kinematics(0, start_robtarg).robax
        return qi, qf

    def opt_solver(self, qi, qf, mass):
        N = 20
        sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                          tool_mass=mass,
                                          tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.],
                                          tool_inertia=[0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)

        # generate a problem
        problem = opt_problem_formulations.problem_136b_X_q_qd_dt_U_qdd(sym_robot)

        nlp_params = nlps.Parameters(N=N, cost_coeff_jerk=1e-7)

        # generate nlp from the problem and solve it
        sol = nlps.generate_and_solve_nlp(problem, qi=qi, qf=qf,  p=nlp_params)

        # interpolate the optimal solution
        sol_intrp = problem.interpolate_solution(sol, dt_resample=0.004)

        return sol_intrp["q"], sol_intrp["qd"], sol_intrp["qdd"]
