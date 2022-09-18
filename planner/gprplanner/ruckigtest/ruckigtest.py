from pathlib import Path
from sys import path
import numpy as np
 
# Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
build_path = Path(__file__).parent.absolute().parent / 'build'
path.insert(0, str(build_path))
 
from ruckig import InputParameter, Ruckig, Trajectory, Result
 
from casadi_kinodynamics.utils import symbolic_robot
 
if __name__ == '__main__':
    sym_bot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                          tool_mass=0.0,
                                          tool_frame=[0., 0., 0.25, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.12],
                                          tool_inertia=[
                                              0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)
    qd_max = sym_bot.qd_max
    qdd_max = sym_bot.qdd_max
    # qdd_max[0] = 40.
    qdd_max[0] = 25.
    qddd_max = [1000.] * 6

    inp = InputParameter(6)
 
    qi = [i/180.*np.pi for i in [-90., 90, -90., 0., 0., 0.]]  
    qf = [i/180.*np.pi for i in [90.,90., -90., 0., 0., 0.]]
    # qi = [i/180.*np.pi for i in [-90., 0., -0., 0., 0., 0.]]
    # qf = [i/180.*np.pi for i in [90., 0., -0., 0., 0., 0.]]

    inp.current_position = qi
    inp.current_velocity = [0.0] * 6
    inp.current_acceleration = [0.0] * 6
 
    inp.target_position = qf
    inp.target_velocity = [0.0] * 6
    inp.target_acceleration = [0.0] * 6
 
    inp.max_velocity = qd_max
    inp.max_acceleration = qdd_max
    inp.max_jerk = qddd_max
 
    # Set different constraints for negative direction
    inp.min_velocity = (np.array(qd_max)*-1).tolist()
    inp.min_acceleration = (np.array(qdd_max)*-1).tolist()
 
    # We don't need to pass the control rate (cycle time) when using only offline features
    otg = Ruckig(6)
    trajectory = Trajectory(6)
 
    # Calculate the trajectory in an offline manner
    import time
    s = time.time()
    for _ in range(100):
        result = otg.calculate(inp, trajectory)
        if result == Result.ErrorInvalidInput:
            raise Exception('Invalid input!')
    e = time.time()
    print((e - s)/100)
    print(f'Trajectory duration: {trajectory.duration:0.4f} [s]')
 
    new_time = 0.1
 
    # Then, we can calculate the kinematic state at a given time
    new_position, new_velocity, new_acceleration = trajectory.at_time(new_time)
 
    print(f'Position at time {new_time:0.4f} [s]: {new_position}')
 
    # Get some info about the position extrema of the trajectory
    print(f'Position extremas are {trajectory.position_extrema}')