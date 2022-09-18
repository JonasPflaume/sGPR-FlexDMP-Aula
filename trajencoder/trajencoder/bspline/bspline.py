import numpy as np

class Bspline(object):
    ''' bspline base class
    '''
    def __init__(self, knots, init=0., end=0.):
        self.init = init
        self.end = end
        self.knots = knots
        self.order = 0 # init polynomial character

    def get_weights_through_traj(self, T, Y):
        '''
            Input:
                    T: the time vector
                    Y: traj value, has shape (len(T), n), n for number of channel
        '''
        traj_length, channel = Y.shape
        assert len(T) == traj_length, "The time length dosen't match the trajectory."
        # subsample, including two boundary points
        dt = T[1] - T[0]
        indx = [0] + [i*(traj_length-traj_length%(self.knots+1))//(self.knots+1) for i in range(1, self.knots+1)] + [traj_length-1]
        t = [T[i] for i in indx]
        y = [Y[i].reshape(1,-1) for i in indx]
        y = np.concatenate(y) # y has shape (knots+2, n)
        self.weights = np.zeros([channel, self.knots+1, self.order]) # 4 for cubic spline
        for c in range(channel):
            # calc weight for each channel/joints
            weight = self._calc_weight(t, y[:, c], c).reshape(self.knots+1, self.order)
            self.weights[c, :, :] = weight
        return self.weights

    def subsample(self, T, Y, include_start_end=False):
        ''' subsamp the traj with the given knots number
        '''
        traj_length = len(T)
        if include_start_end:
            indx = [0] + [i*(traj_length-traj_length%(self.knots+1))//(self.knots+1) for i in range(1, self.knots+1)] + [traj_length-1]
        else:
            indx = [i*(traj_length-traj_length%(self.knots+1))//(self.knots+1) for i in range(1, self.knots+1)]
        t = [T[i] for i in indx]
        y = [Y[i].reshape(1,-1) for i in indx]
        y = np.concatenate(y)
        t = np.array(t)
        return t, y

    def get_weights_through_via(self, T, Y):
        ''' via point interpolating
            Input:
                    T: Time vector has the same dim as Y
                    Y: Via point value
            Output:
                    weights
        '''
        point_num, channel = Y.shape
        assert len(T) == point_num, "The time length dosen't match the trajectory."
        assert point_num == self.knots + 2, "The knots number + 2 should be equal to the point number"
        # subsample, including two boundary points
        self.weights = np.zeros([channel, self.knots+1, self.order]) # e.g 4 for cubic spline
        for c in range(channel):
            # calc weight for each channel/joints
            weight = self._calc_weight(T, Y[:, c], c).reshape(self.knots+1, self.order)
            self.weights[c, :, :] = weight
        return self.weights

    def set_weight(self, w):
        ''' To set the learning weights
        '''
        assert w.shape[2] == self.order and w.shape[1] == (self.knots + 1), "The preset knots number might be wrong"
        self.weights = w

    def trajectory(self, time, dt, return_dyddy=False):
        ''' reconstruct the trajectory from encoding
            Input:
                    time: the length of trajectory in second
                    dt:   reconstruct time interval
            output:
                    traj.T: the trajectory with shape (T, c), T = time/dt, c channel
                            if return_dyddy=False, then return q, qd, qdd
        '''
        time_vec = np.linspace(0., time, int(time/dt))
        traj_length = len(time_vec)
        traj = np.zeros([len(self.weights), traj_length])
        indx = [0] + [i*(traj_length-traj_length%(self.knots+1))//(self.knots+1) for i in range(1, self.knots+1)] + [traj_length]
        if return_dyddy:
            traj_d = np.zeros([len(self.weights), traj_length])
            traj_dd = np.zeros([len(self.weights), traj_length])
            for i in range(self.knots + 1):
                y, dy, ddy = self._polyfunc(self.weights[:,i,:], time_vec[indx[i]:indx[i+1]], return_dyddy=True)
                traj[:, indx[i]:indx[i+1]] = y
                traj_d[:, indx[i]:indx[i+1]] = dy
                traj_dd[:, indx[i]:indx[i+1]] = ddy
            return traj.T, traj_d.T, traj_dd.T
        else:
            for i in range(self.knots + 1):
                traj[:, indx[i]:indx[i+1]] = self._polyfunc(self.weights[:,i,:], time_vec[indx[i]:indx[i+1]])
            return traj.T

    def _calc_weight(self, t, y):
        ''' solve the linear system and return the weight for one channel
            input:
                t:  1-d numpy array
                y:  1-d numpy array
            output:
                weight: 2-d numpy array, shape=[knots+1, order]
        '''
        raise NotImplementedError("The calc weight should be defined for each spline class")

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, order)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        raise NotImplementedError("The poly func should be defined for each spline class")

class CubicSpline(Bspline):
    ''' 1. This class is going to implement the bspline interpolating by given
        the first derivative boundary condition, for evenly divided traj
        2. It seems the init velocity and end velocity is not the prime pursue of least square approach
    '''
    def __init__(self, knots, init=0., end=0.):
        ''' given init and end velocity value
            given knots number
            example: ----|----|----|----, knots=3 and there are 4 group of parameters
        '''
        super().__init__(knots, init=init, end=end)
        self.order = 4 # cubic poly is order 4
    
    def _calc_weight(self, t, y, dof):
        ''' solve the linear system and return the weight for one channel
            input:
                t:  1-d numpy array
                y:  1-d numpy array
            output:
                weight: 2-d numpy array, shape=[knots+1, 4]
        '''
        # A_diag = [self._build_A_block(t[i], t[i+1]) for i in range(0, self.knots)]
        assert len(t) == (self.knots + 2)
        width = self.order * (self.knots + 1) # width of A matrix
        A = []
        for i in range(self.knots):
            temp = np.zeros([self.order, self.order * (self.knots + 1)]) # start building matrix
            temp[:, i*4:i*4+8] = self._build_A_block(t[i], t[i+1])
            A.append(temp)
        A = np.concatenate(A, axis=0)
        # add boundary condition
        A_boundary = np.zeros([4, width], dtype=float)
        A_boundary[0, -4:] = np.array([1., t[-2], t[-2]**2, t[-2]**3], dtype=float)
        A_boundary[1, -4:] = np.array([1., t[-1], t[-1]**2, t[-1]**3], dtype=float)
        A_boundary[2, :4]  = np.array([0., t[0], 2*t[0], 3*t[0]**2], dtype=float) # boundary init velocity
        A_boundary[3, -4:] = np.array([0., t[-1], 2*t[-1], 3*t[-1]**2], dtype=float) # boundary end velocity
        A = np.append(A, A_boundary, axis=0)
        assert A.shape == (4*(self.knots+1), 4*(self.knots+1)), "A might be mistakenly initialized.."
        b = [np.array([y[i], y[i+1],0 ,0], dtype=float) for i in range(0, self.knots)]
        b = np.concatenate(b)
        # add boundary condition
        b_boundary = np.array([y[-2], y[-1], self.init[dof], self.end[dof]], dtype=float)
        b = np.concatenate([b, b_boundary]).reshape(-1,1)
        #return weight
        weight = np.linalg.inv( A.T @ A + np.eye(A.shape[1])*1e-3 ) @ A.T @ b
        return weight

    def _build_A_block(self, x, x_next):
        ''' helper function to build the dignonal block of A matrix in _calc_weight
        '''
        A_i =   np.array(
                    [[1.,   x,      x**2,       x**3,       0.,     0.,     0.,         0.],
                     [1.,   x_next, x_next**2,  x_next**3,  0.,     0.,     0.,         0.],
                     [0.,   1.,     2*x_next,   3*x_next**2,0.,    -1.,    -2*x_next,  -3*x_next**2],
                     [0.,   0.,     2.,         6*x_next,   0.,     0.,    -2.,        -6*x_next]], 
                    dtype=float)
        return A_i

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, 4)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        t = np.repeat(t.reshape(1,-1), len(self.weights), axis=0)
        y = w_i[:,0:1] + w_i[:,1:2] * t + w_i[:,2:3] * t**2 + w_i[:,3:4] * t**3
        if return_dyddy:
            dy = w_i[:,1:2] + 2 * w_i[:,2:3] * t + 3*w_i[:,3:4] * t**2
            ddy = 2 * w_i[:,2:3] + 6*w_i[:,3:4] * t
            return y, dy, ddy
        else:
            return y

class Order5Spline(Bspline):
    ''' 1. This class is going to implement the bspline interpolating by given
        the velocity boundary conditions for both side and the acceleration condition for one side, for evenly divided traj.
        In order to make the 5n linear equations well defined
        2. It seems the init velocity and end velocity is not the prime pursue of least square approach
    '''
    def __init__(self, knots, init=0., end=0.):
        ''' given init and end velocity value
            given knots number
            example: ----|----|----|----, knots=3 and there are 4 group of parameters
        '''
        super().__init__(knots, init=init, end=end)
        self.order = 5 # the order of quintic poly
    
    def _calc_weight(self, t, y):
        ''' solve the linear system and return the weight for one channel
            here we define an extra init condition qdd = 0, in order to make 5n equations
            input:
                t:  1-d numpy array, (knots+2, dof)
                y:  1-d numpy array, (knots+2, dof)
            output:
                weight: 2-d numpy array, shape=[knots+1, 5]
        '''
        width = self.order * (self.knots + 1) # width of A matrix
        A = []
        for i in range(self.knots):
            temp = np.zeros([self.order, width]) # start building matrix
            temp[:, i*self.order:i*self.order+2*self.order] = self._build_A_block(t[i], t[i+1])
            A.append(temp)
        A = np.concatenate(A, axis=0)
        # add boundary condition
        A_boundary = np.zeros([self.order-1, width])
        A_boundary[0, -self.order:] = np.array([1., t[-2], t[-2]**2, t[-2]**3, t[-2]**4], dtype=float)
        A_boundary[1, -self.order:] = np.array([1., t[-1], t[-1]**2, t[-1]**3, t[-1]**4], dtype=float)
        A_boundary[2, :self.order]  = np.array([0., 0., 2, 6*t[0], 12*t[0]**2], dtype=float)                 # init acc
        A_boundary[3, -self.order:] = np.array([0., 0., 2., 6*t[-1], 12*t[-1]**2], dtype=float)              # final acc
        # A_boundary[4, :self.order] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # init vel is zero
        # A_boundary[5, -self.order:] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # final vel is zero
        A = np.append(A, A_boundary, axis=0)
        # assert A.shape == (5*(self.knots+1), 5*(self.knots+1)), "A might be mistakenly initialized.."
        b = [np.array([y[i], y[i+1], 0., 0., 0.], dtype=float) for i in range(0, self.knots)]
        b = np.concatenate(b)
        # add boundary condition
        b_boundary = np.array([y[-2], y[-1], self.init, self.end], dtype=float)
        b = np.concatenate([b, b_boundary]).reshape(-1,1)
        #return weight
        weight = np.linalg.pinv(A) @ b
        return weight

    def _build_A_block(self, x, x_next):
        ''' helper function to build the dignonal block of A matrix in _calc_weight
        '''
        A_i =   np.array(
                    [[1.,   x,      x**2,       x**3,       x**4,           0.,     0.,     0.,         0.,             0.],
                     [1.,   x_next, x_next**2,  x_next**3,  x_next**4,      0.,     0.,     0.,         0.,             0.],
                     [0.,   1.,     2*x_next,   3*x_next**2,4*x_next**3,    0.,    -1.,    -2*x_next,  -3*x_next**2,    -4*x_next**3],
                     [0.,   0.,     2.,         6*x_next,   12*x_next**2,   0.,     0.,    -2.,        -6*x_next,       -12*x_next**2],
                     [0.,   0.,     0.,         6.,         24*x_next,      0.,     0.,     0.,        -6.,             -24*x_next]], 
                    dtype=float)
        return A_i

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, 4)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        t = np.repeat(t.reshape(1,-1), len(self.weights), axis=0)
        y = w_i[:,0:1] + w_i[:,1:2] * t + w_i[:,2:3] * t**2 + w_i[:,3:4] * t**3 + w_i[:,4:5] * t**4
        if return_dyddy:
            dy = w_i[:,1:2] + 2 * w_i[:,2:3] * t + 3*w_i[:,3:4] * t**2 + 4*w_i[:,4:5] * t**3
            ddy = 2 * w_i[:,2:3] + 6*w_i[:,3:4] * t + 12*w_i[:,4:5] * t**2
            return y, dy, ddy
        else:
            return y

class QuinticSpline(Bspline):
    ''' 1. This class is going to implement the bspline interpolating by given
        the velocity boundary conditions for both side and the acceleration condition for one side, for evenly divided traj.
        In order to make the 5n linear equations well defined
        2. It seems the init velocity and end velocity is not the prime pursue of least square approach
    '''
    def __init__(self, knots, init=0., end=0.):
        ''' given init and end velocity value
            given knots number
            example: ----|----|----|----, knots=3 and there are 4 group of parameters
        '''
        super().__init__(knots, init=init, end=end)
        self.order = 6 # the order of quintic poly
    
    def _calc_weight(self, t, y, c):
        ''' solve the linear system and return the weight for one channel
            here we define an extra init condition qdd = 0, in order to make 5n equations
            input:
                t:  1-d numpy array, (knots+2, dof)
                y:  1-d numpy array, (knots+2, dof)
            output:
                weight: 2-d numpy array, shape=[knots+1, 6]
        '''
        width = self.order * (self.knots + 1) # width of A matrix
        A = []
        for i in range(self.knots):
            temp = np.zeros([self.order, width]) # start building matrix
            temp[:, i*self.order:i*self.order+2*self.order] = self._build_A_block(t[i], t[i+1])
            A.append(temp)
        A = np.concatenate(A, axis=0)
        # add boundary condition
        A_boundary = np.zeros([self.order-2, width])
        A_boundary[0, -self.order:] = np.array([1., t[-2], t[-2]**2, t[-2]**3, t[-2]**4, t[-2]**5], dtype=float)
        A_boundary[1, -self.order:] = np.array([1., t[-1], t[-1]**2, t[-1]**3, t[-1]**4, t[-1]**5], dtype=float)
        A_boundary[2, :self.order]  = np.array([0., 0., 2, 6*t[0], 12*t[0]**2, 20*t[0]**3], dtype=float)                 # init acc
        A_boundary[3, -self.order:] = np.array([0., 0., 2., 6*t[-1], 12*t[-1]**2, 20*t[-1]**3], dtype=float)              # final acc
        # A_boundary[4, :self.order] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # init vel is zero
        # A_boundary[5, -self.order:] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # final vel is zero
        A = np.append(A, A_boundary, axis=0)
        # assert A.shape == (5*(self.knots+1), 5*(self.knots+1)), "A might be mistakenly initialized.."
        b = [np.array([y[i], y[i+1], 0., 0., 0., 0.], dtype=float) for i in range(0, self.knots)]
        b = np.concatenate(b)
        # add boundary condition
        b_boundary = np.array([y[-2], y[-1], self.init, self.end], dtype=float)
        b = np.concatenate([b, b_boundary]).reshape(-1,1)
        #return weight
        weight = np.linalg.pinv(A) @ b
        return weight

    def _build_A_block(self, x, x_next):
        ''' helper function to build the dignonal block of A matrix in _calc_weight
        '''
        A_i =   np.array(
                    [[1.,   x,      x**2,       x**3,       x**4,           x**5,           0.,     0.,     0.,         0.,             0.,    0.],
                     [1.,   x_next, x_next**2,  x_next**3,  x_next**4,      x_next**5,      0.,     0.,     0.,         0.,             0.,    0.],
                     [0.,   1.,     2*x_next,   3*x_next**2,4*x_next**3,    5*x_next**4,    0.,    -1.,    -2*x_next,  -3*x_next**2,    -4*x_next**3,   -5*x_next**4],
                     [0.,   0.,     2.,         6*x_next,   12*x_next**2,   20*x_next**3,   0.,     0.,    -2.,        -6*x_next,       -12*x_next**2,  -20*x_next**3],
                     [0.,   0.,     0.,         6,   24*x_next,   60*x_next**2,   0.,     0.,    0.,        -6,       -24*x_next,  -60*x_next**2],
                     [0.,   0.,     0.,         0.,   24,   120*x_next,   0.,     0.,    0.,        0.,       -24,  -120*x_next]
                     ], 
                    dtype=float)
        return A_i

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, 4)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        t = np.repeat(t.reshape(1,-1), len(self.weights), axis=0)
        y = w_i[:,0:1] + w_i[:,1:2] * t + w_i[:,2:3] * t**2 + w_i[:,3:4] * t**3 + w_i[:,4:5] * t**4 + w_i[:,5:6] * t**5
        if return_dyddy:
            dy = w_i[:,1:2] + 2 * w_i[:,2:3] * t + 3*w_i[:,3:4] * t**2 + 4*w_i[:,4:5] * t**3 + 5*w_i[:,5:6] * t**4
            ddy = 2 * w_i[:,2:3] + 6*w_i[:,3:4] * t + 12*w_i[:,4:5] * t**2 + 20*w_i[:,5:6] * t**3
            return y, dy, ddy
        else:
            return y

class Order7spline(Bspline):
    def __init__(self, knots, init=0., end=0.):
        ''' given init and end velocity value
            given knots number
            example: ----|----|----|----, knots=3 and there are 4 group of parameters
        '''
        super().__init__(knots, init=init, end=end)
        self.order = 7 # the order of poly
    
    def _calc_weight(self, t, y):
        ''' solve the linear system and return the weight for one channel
            here we define an extra init condition qdd = 0, in order to make 5n equations
            input:
                t:  1-d numpy array, (knots+2, dof)
                y:  1-d numpy array, (knots+2, dof)
            output:
                weight: 2-d numpy array, shape=[knots+1, 6]
        '''
        width = self.order * (self.knots + 1) # width of A matrix
        A = []
        for i in range(self.knots):
            temp = np.zeros([self.order, width]) # start building matrix
            temp[:, i*self.order:i*self.order+2*self.order] = self._build_A_block(t[i], t[i+1])
            A.append(temp)
        A = np.concatenate(A, axis=0)
        # add boundary condition
        A_boundary = np.zeros([self.order-3, width])
        A_boundary[0, -self.order:] = np.array([1., t[-2], t[-2]**2, t[-2]**3, t[-2]**4, t[-2]**5, t[-2]**6], dtype=float)
        A_boundary[1, -self.order:] = np.array([1., t[-1], t[-1]**2, t[-1]**3, t[-1]**4, t[-1]**5, t[-1]**6], dtype=float)
        A_boundary[2, :self.order]  = np.array([0., 0., 2, 6*t[0], 12*t[0]**2, 20*t[0]**3, 30*t[0]**4], dtype=float)                 # init acc
        A_boundary[3, -self.order:] = np.array([0., 0., 2., 6*t[-1], 12*t[-1]**2, 20*t[-1]**3, 30*t[-1]**4], dtype=float)              # final acc
        # A_boundary[4, :self.order] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # init vel is zero
        # A_boundary[5, -self.order:] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # final vel is zero
        A = np.append(A, A_boundary, axis=0)
        # assert A.shape == (5*(self.knots+1), 5*(self.knots+1)), "A might be mistakenly initialized.."
        b = [np.array([y[i], y[i+1], 0., 0., 0., 0., 0.], dtype=float) for i in range(0, self.knots)]
        b = np.concatenate(b)
        # add boundary condition
        b_boundary = np.array([y[-2], y[-1], self.init, self.end], dtype=float)
        b = np.concatenate([b, b_boundary]).reshape(-1,1)
        #return weight
        weight = np.linalg.pinv(A) @ b
        return weight

    def _build_A_block(self, x, x_next):
        ''' helper function to build the dignonal block of A matrix in _calc_weight
        '''
        A_i =   np.array(
                    [[1.,   x,      x**2,       x**3,       x**4,           x**5,       x**6,      0.,       0.,     0.,     0.,         0.,             0.,    0.],
                     [1.,   x_next, x_next**2,  x_next**3,  x_next**4,      x_next**5,  x_next**6,   0.,     0.,     0.,     0.,         0.,             0.,    0.],
                     [0.,   1.,     2*x_next,   3*x_next**2,4*x_next**3,    5*x_next**4,6*x_next**5,    0.,    -1.,    -2*x_next,  -3*x_next**2,    -4*x_next**3,   -5*x_next**4, -6*x_next**5],
                     [0.,   0.,     2.,         6*x_next,   12*x_next**2,   20*x_next**3,30*x_next**4,   0.,     0.,    -2.,        -6*x_next,       -12*x_next**2,  -20*x_next**3, -30*x_next**4],
                     [0.,   0.,     0.,         6,   24*x_next,   60*x_next**2, 120*x_next**3,   0.,     0.,    0.,        -6,       -24*x_next,  -60*x_next**2, -120*x_next**3],
                     [0.,   0.,     0.,         0.,   24,   120*x_next, 360*x_next**2,   0.,     0.,    0.,        0.,       -24,  -120*x_next, -360*x_next**2],
                     [0.,   0.,     0.,         0.,   0.,   120,    720*x_next, 0.,     0.,    0.,        0.,       0.,  -120, -720*x_next]
                     ], 
                    dtype=float)
        return A_i

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, 4)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        t = np.repeat(t.reshape(1,-1), len(self.weights), axis=0)
        y = w_i[:,0:1] + w_i[:,1:2] * t + w_i[:,2:3] * t**2 + w_i[:,3:4] * t**3 + w_i[:,4:5] * t**4 + w_i[:,5:6] * t**5 + w_i[:,6:7] * t**6
        if return_dyddy:
            dy = w_i[:,1:2] + 2 * w_i[:,2:3] * t + 3*w_i[:,3:4] * t**2 + 4*w_i[:,4:5] * t**3 + 5*w_i[:,5:6] * t**4 + 6*w_i[:,6:7] * t**5
            ddy = 2 * w_i[:,2:3] + 6*w_i[:,3:4] * t + 12*w_i[:,4:5] * t**2 + 20*w_i[:,5:6] * t**3 + 30*w_i[:,6:7] * t**4
            return y, dy, ddy
        else:
            return y


class Order8spline(Bspline):
    def __init__(self, knots, init=0., end=0.):
        ''' given init and end velocity value
            given knots number
            example: ----|----|----|----, knots=3 and there are 4 group of parameters
        '''
        super().__init__(knots, init=init, end=end)
        self.order = 8 # the order of poly
    
    def _calc_weight(self, t, y):
        ''' solve the linear system and return the weight for one channel
            here we define an extra init condition qdd = 0, in order to make 5n equations
            input:
                t:  1-d numpy array, (knots+2, dof)
                y:  1-d numpy array, (knots+2, dof)
            output:
                weight: 2-d numpy array, shape=[knots+1, 6]
        '''
        width = self.order * (self.knots + 1) # width of A matrix
        A = []
        for i in range(self.knots):
            temp = np.zeros([self.order-3, width]) # start building matrix
            temp[:, i*self.order:i*self.order+2*self.order] = self._build_A_block(t[i], t[i+1])
            A.append(temp)
        A = np.concatenate(A, axis=0)
        # add boundary condition
        A_boundary = np.zeros([self.order-4, width])
        A_boundary[0, -self.order:] = np.array([1., t[-2], t[-2]**2, t[-2]**3, t[-2]**4, t[-2]**5, t[-2]**6, t[-2]**7], dtype=float)
        A_boundary[1, -self.order:] = np.array([1., t[-1], t[-1]**2, t[-1]**3, t[-1]**4, t[-1]**5, t[-1]**6, t[-1]**7], dtype=float)
        A_boundary[2, :self.order]  = np.array([0., 0., 2, 6*t[0], 12*t[0]**2, 20*t[0]**3, 30*t[0]**4, 42*t[0]**5], dtype=float)                 # init acc
        A_boundary[3, -self.order:] = np.array([0., 0., 2., 6*t[-1], 12*t[-1]**2, 20*t[-1]**3, 30*t[-1]**4, 42*t[-1]**5], dtype=float)              # final acc
        # A_boundary[4, :self.order] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3, 5*t[0]**4, 6*t[0]**5, 7*t[0]**6], dtype=float)                 # init vel is zero
        # A_boundary[5, -self.order:] = np.array([0., 1., 2*t[-1], 3*t[-1]**2, 4*t[-1]**3, 5*t[-1]**4, 6*t[-1]**5, 7*t[-1]**6], dtype=float)          # final vel is zero
        A = np.append(A, A_boundary, axis=0)
        # assert A.shape == (5*(self.knots+1), 5*(self.knots+1)), "A might be mistakenly initialized.."
        b = [np.array([y[i], y[i+1], 0., 0., 0.], dtype=float) for i in range(0, self.knots)]
        b = np.concatenate(b)
        # add boundary condition
        b_boundary = np.array([y[-2], y[-1], self.init, self.end], dtype=float)
        b = np.concatenate([b, b_boundary]).reshape(-1,1)
        #return weight
        weight = np.linalg.pinv(A) @ b
        return weight

    def _build_A_block(self, x, x_next):
        ''' helper function to build the dignonal block of A matrix in _calc_weight
        '''
        A_i =   np.array(
                    [[1., x, x**2, x**3, x**4, x**5, x**6, x**7,  0., 0., 0., 0., 0., 0., 0., 0.],
                     [1., x_next, x_next**2, x_next**3, x_next**4, x_next**5, x_next**6, x_next**7, 0., 0., 0., 0., 0., 0., 0.,0.],
                     [0., 1., 2*x_next, 3*x_next**2, 4*x_next**3,  5*x_next**4, 6*x_next**5, 7*x_next**6, 0., -1., -2*x_next, -3*x_next**2, -4*x_next**3, -5*x_next**4, -6*x_next**5, -7*x_next**6],
                     [0., 0., 2., 6*x_next, 12*x_next**2, 20*x_next**3, 30*x_next**4, 42*x_next**5, 0., 0., -2., -6*x_next, -12*x_next**2, -20*x_next**3, -30*x_next**4, -42*x_next**5],
                     [0., 0., 0., 6, 24*x_next, 60*x_next**2, 120*x_next**3, 210*x_next**4, 0.,     0.,    0.,        -6,       -24*x_next,  -60*x_next**2, -120*x_next**3,-210*x_next**4],
                    #  [0., 0., 0., 0., 24, 120*x_next, 360*x_next**2, 840*x_next**3,   0.,     0.,    0.,        0.,       -24,  -120*x_next, -360*x_next**2, -840*x_next**3],
                    #  [0., 0., 0., 0., 0., 120, 720*x_next, 2520*x_next**2, 0., 0., 0., 0., 0., -120, -720*x_next, -2520*x_next**2],
                    #  [0., 0., 0., 0., 0., 0., 720., 5040*x_next, 0., 0., 0., 0., 0., 0., -720., -5040*x_next]
                     ],
                    dtype=float)
        return A_i

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, 4)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        t = np.repeat(t.reshape(1,-1), len(self.weights), axis=0)
        y = w_i[:,0:1] + w_i[:,1:2] * t + w_i[:,2:3] * t**2 + w_i[:,3:4] * t**3 + w_i[:,4:5] * t**4 + w_i[:,5:6] * t**5 + w_i[:,6:7] * t**6 + w_i[:,7:8] * t**7 
        if return_dyddy:
            dy = w_i[:,1:2] + 2 * w_i[:,2:3] * t + 3*w_i[:,3:4] * t**2 + 4*w_i[:,4:5] * t**3 + 5*w_i[:,5:6] * t**4 + 6*w_i[:,6:7] * t**5 + 7*w_i[:,7:8] * t**6
            ddy = 2 * w_i[:,2:3] + 6*w_i[:,3:4] * t + 12*w_i[:,4:5] * t**2 + 20*w_i[:,5:6] * t**3 + 30*w_i[:,6:7] * t**4 + 42*w_i[:,7:8] * t**5
            return y, dy, ddy
        else:
            return y



if __name__ == "__main__":
    ### test run on multiple dimensional traj ###
    bs = Order8spline(20)

    t = np.linspace(0,2*np.pi,1000).reshape(-1,1)
    y = np.sin(t) + 2*np.sin(2*t)
    dy = np.cos(t) + 4*np.cos(2*t)
    ddy = -np.sin(t) - 8*np.sin(2*t)
    y = np.repeat(y, 2, axis=1)

    bs.get_weights_through_traj(t, y)
    traj, traj_d, traj_dd = bs.trajectory(2*np.pi, 2*np.pi/1000, return_dyddy=True)
    print("There are %i parameters." % bs.weights.size)
    import matplotlib.pyplot as plt
    plt.figure(figsize=[12,8])
    plt.plot(y[:, 0], '-.r', label="true_q")
    plt.plot(dy[:, 0], '-.c', label="true_dq")
    plt.plot(ddy[:, 0], '-.b', label="true_ddq")
    plt.plot(traj[:, 0], 'r', label="q")
    plt.plot(traj_d[:, 0], 'c', label="dq")
    plt.plot(traj_dd[:, 0], 'b', label="ddq")
    plt.legend()
    plt.grid()
    plt.savefig("/home/jiayun/Desktop/spline.jpg", dpi=200)
    plt.show()