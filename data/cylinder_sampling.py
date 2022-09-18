#
# using feature space inverse kinematics to generate the cylinder-like configuration space sampling
#
from toppysail.utils import util_pyvista as utpv
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from toppysail.utils import util_functions as ut
from casadi_kinodynamics.utils import symbolic_robot


def cylinder_sampling(sym_robot, r_range:list, height:float, grid_num:list, q_start, q_samples:list, append_positioning=False):
    ''' given a sym_robot object, target radius start to end and a starting point q_start and a container q_samples
        We are computing a cylinder sampling in the robot configurations space.
    '''
    def circle_start_positioning(target, q_start, z_specific=None):
        ''' target is target radius
        '''
        if z_specific == None:
            z_target = sym_robot.for_kine(q_start)[-1][0:3,3][2]
        else:
            z_target = z_specific
        while True:
            J_full = sym_robot.jacobian(q_start).full()
            position = sym_robot.for_kine(q_start)[-1][0:3,3]
            x_f, y_f, z_f = position[0], position[1], position[2]
            J_f1 = 2*x_f*J_full[0:1,:] + 2*y_f*J_full[1:2,:] # feature jacobian
            J_f2 = J_full[2:3,:] # feature jacobian
            feature = np.array([x_f ** 2 + y_f ** 2, z_f, 0, 0, 0])
            feature_control = -0.3 * (feature - np.array([target**2, z_target, 0, 0, 0], dtype=float)) # p control
            J_f = np.concatenate([J_f1, J_f2, J_full[3:,:]])
            q_start += np.linalg.pinv(J_f) @ (feature_control)
            if append_positioning:
                q_samples.append(q_start.copy().tolist())
            if np.all(np.abs(feature_control) < 1e-6):
                break
        if not append_positioning:
            q_samples.append(q_start.copy().tolist())
        return q_start

    def one_circle(q_start, r, dr=grid_num[2], inverse=False):
        N = int(r / dr)
        dt = 2 * np.pi / N
        # this ode defined a rotating with angular speed = -1 rad/s
        for _ in range(N-1):
            J_full = sym_robot.jacobian(q_start).full()
            position = sym_robot.for_kine(q_start)[-1][0:3,3]
            x_f, y_f, z_f = position[0], position[1], position[2]
            if inverse:
                twist = np.array([-y_f, x_f, 0., 0., 0., 0.])
            else:
                twist = np.array([y_f, -x_f, 0., 0., 0., 0.])
            q_start += dt*np.linalg.pinv(J_full) @ twist
            q_samples.append(q_start.copy().tolist())
        return q_start
    r_start, r_end = r_range
    # initial positioning
    q_start = circle_start_positioning(r_start, q_start)
    curr_z = sym_robot.for_kine(q_start)[-1][0:3,3][2]
    
    h_start, h_end = curr_z, curr_z + height
    r_num, h_num = grid_num[0], grid_num[1]
    r_range = np.linspace(r_start, r_end, r_num)
    h_range = np.linspace(h_start, h_end, h_num)

    radius_height_product = []
    for h_i in range(0, len(h_range)-1, 2):
        radius_height_product += [(r_range[r_i], h_range[h_i]) for r_i in range(len(r_range))]
        radius_height_product += [(r_range[r_i], h_range[h_i+1]) for r_i in reversed(range(len(r_range)))]
    if len(h_range) % 2 != 0:
        radius_height_product += [(r_range[r_i], h_range[-1]) for r_i in range(len(r_range))]

    inverse=False
    for r_target, z_target in radius_height_product:
            q_start = circle_start_positioning(r_target, q_start, z_specific=z_target)
            q_start = one_circle(q_start, r_target, inverse=inverse)
            inverse = not inverse
    return q_samples

if __name__ == "__main__":
    plot_coordinate_frames = False
    # load the robot
    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                            tool_mass=.5,
                                            tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                            tool_cog=[0.0, 0., 0.0],
                                            tool_inertia=[
                                                0., 0., 0., 0., 0., 0.],
                                            load_casadi_fnc=True)

    q_start = [i/180.*np.pi for i in [180., 23.4, 31.2, 0., 35.4, -31.1]]
    q_samples = [q_start]
    # sym_robot, r_range, height, grid_num, q_start:current q, q_samples:container
    q_samples = cylinder_sampling(sym_robot, [0.25, 0.5], 0.25, [7, 7, 0.01], q_start, q_samples)
    q_samples = np.array(q_samples)
    # print(q_samples)
    violate_joint_limit = False
    if np.any(sym_robot.q_min > np.min(q_samples, axis=0)) or np.any(sym_robot.q_max < np.max(q_samples, axis=0)):
        violate_joint_limit = True
    print("Number of sampling: ", len(q_samples), ", if joint was violated: ", violate_joint_limit)

    plotter = utpv.pv_plot_visual_mesh_robot(robot=sym_robot, q=q_start, plotter=None)
    plotter = utpv.pv_plot_visual_mesh_robot(
        robot=sym_robot, q=q_start, plotter=None)
    if plot_coordinate_frames:
        pose_traj = ut.get_ee_cart_pose_from_joint_array(
            sym_robot.for_kine, q_samples)
        for pose in pose_traj:
            utpv.pv_plot_corrdinate_axes(pose, plotter=plotter, length=0.01)

    else:
        utpv.plot_cart_path(q_samples, sym_robot, plotter, plot_line=False, plot_sphere_markers=True,
                            tube_radius=0.0005, marker_color=[0.8, 0., 0.2], plot_coordinate_frames=False)

    plotter.show(cpos=[(-0.5, -4, 0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)])
