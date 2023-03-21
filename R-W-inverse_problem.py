import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt
import io
import re

#Unknown parameters
w_r = dde.Variable(1.0)   # Real part
w_i = dde.Variable(-1.0)  # Imaginary part

#Domain
bdr_inf = 0.9
tf.random.get_seed(5)

#Exact solutions
#Soluciones analiticas real e imaginaria 
def func_r(y):
    '''
    Soluciones dependiendo el modo 
    n = 0 : (tf.math.sqrt(tf.math.cosh(x)) * tf.math.cos(tf.math.log(tf.math.cosh(x)) * 0.5))
    n = 1 : n(0) * tf.math.sinh(x) 
    n = 2 : n(0) * (3.0*(tf.math.sinh(x)**2) + 1.0) //// im : tf.math.sinh(x)**2
    n = 3 : n(0) * (5/3 * tf.math.sinh(x)**3 + tf.math.sinh(x))
    '''
    sol1 = np.real((((5/3 + 1j/3)*y**3)/((1 - y**2)**(3/2)) + y/((1 - y**2)**(1/2)))*(1 - y**2)**(-1/4 - 1j/4))
    return  sol1

def func_i(y):
    '''
    Soluciones dependiendo el modo 
    n = 0 : (tf.math.sqrt(tf.math.cosh(x)) * tf.math.sin(tf.math.log(tf.math.cosh(x)) * 0.5))
    n = 1 : n(0) * tf.math.sinh(x) 
    n = 2 : n(0) * tf.math.sinh(x)**2
    n = 3 : n(0) * (1/3 * tf.math.sinh(x)**3)
    '''
    sol1 = np.imag ((((5/3 + 1j/3)*y**3)/((1 - y**2)**(3/2)) + y/((1 - y**2)**(1/2)))*(1 - y**2)**(-1/4 - 1j/4))
    return  sol1 

#Training data
def gen_traindata_r(N):
    """ Create the data from real exact solution
        INPUTS:
            N : number of data 
        Outputs:
            yvals: tensor with the reference points taken (independent values)
            phi_vals: tensor with the analytical values   phi_vals = ψ(y_vals)
    """
    yvals = (np.linspace(-bdr_inf, bdr_inf, N)).reshape(N, 1)
    phi_vals = func_r(yvals)

    return yvals, phi_vals

def gen_traindata_i(N):
    """ Create the data from imaginary exact solution 
        INPUTS:
            N : number of data 
        Outputs:
            yvals: tensor with the reference points taken (independent values)
            phi_vals: tensor with the analytical values   phi_vals = ψ(y_vals)
    """
    yvals = (np.linspace(-bdr_inf, bdr_inf, N)).reshape(N, 1)
    phi_vals = func_i(yvals)

    return yvals, phi_vals


# Differential equation
def PDE(y,x):
    '''Definition of Diferential problem divided in imaginary
       and real part
    ---------------------------------------------------
        INPUTS: 
            x : dependent variables
            y : independent variables 
        OUTPUTS:
            two scalar tensor wit the calculation of functions
            real and imaginary
    ----------------------------------------------------
    '''
    # Real and Imaginary parts of the function
    u, v = x[:, 0:1], x[:, 1:2]

    # Independent variable
    Y = y[:,0:1]

    #Neccesary derivatives
    u_y = dde.grad.jacobian(x, y, i=0, j=0)
    v_y = dde.grad.jacobian(x, y, i=1, j=0)
    u_yy = dde.grad.hessian(x, y, component=0, i=0, j=0)
    v_yy = dde.grad.hessian(x, y, component=1, i=0, j=0)  

    # Differential equations (divided in real/imaginary)
    f_r = ( (1 - Y**2)**2 * u_yy
           - 2 * Y * (1 - Y**2) * u_y
           + (w_r**2 - w_i**2 - 0.5 * (1 - Y**2)) * u - 2 * w_r * w_i * v
    )
    f_i = ( (1 - Y**2)**2 * v_yy
           - 2 * Y * (1 - Y**2) * v_y
           + (w_r**2 - w_i**2 - 0.5 * (1 - Y**2)) * v + 2 * w_r * w_i * u
    )

    return  [f_r, f_i] 

#Domain to solutions
geom = dde.geometry.Interval(-0.9, 0.9) ### Just define a 1D spacial domain

# Boundary Conditions
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], -0.9) 

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.9) 

#Define the boundary
bc1 = dde.DirichletBC(geom, func_r, lambda _, boundary_l : boundary_l, component=0) 
bc2 = dde.DirichletBC(geom, func_r, lambda _, boundary_r : boundary_r, component=0)
bc3 = dde.DirichletBC(geom, func_i, lambda _, boundary_l : boundary_l, component=1) 
bc4 = dde.DirichletBC(geom, func_i, lambda _, boundary_r : boundary_r, component=1)

# "Experimental" data

## Real
y_values, u_values = gen_traindata_r(50) 
exp_data_r = dde.icbc.PointSetBC(y_values, u_values, component=0) # valores w real 

## Imaginary
y_values, v_values = gen_traindata_i(50)
exp_data_i = dde.icbc.PointSetBC(y_values, v_values, component=1) #valores w imaginario


#Discritized Problem
data = dde.data.PDE(geom,
                    PDE,
                    [bc1, bc2, bc3, bc4 ,exp_data_r, exp_data_i], 
                    num_domain=100,
                    num_boundary=2,
                    anchors=y_values)

#Neural network
layer_size = [1] + [20] * 3 + [2]
activation = 'tanh'
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

#Model
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([w_r,w_i], period=1000)

#Compile
model.compile("adam", lr=1e-3, external_trainable_variables=[w_i,w_r])
loss_history, train_state = model.train(iterations=10000, callbacks=[variable])

# Optimizer L-BFGS to improved solution
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS", external_trainable_variables=[w_i,w_r])
losshistory, train_state = model.train(callbacks=[variable])


#Parameters calculated
N = 3                 #Normal Mode
w_r_exp = 1/2         #Real value expected
w_i_exp = -(N + 1/2)  #Imaginary value expected


print('Expected: ','\t ω_r = ',w_r_exp, "\t\t ω_i =", w_i_exp)
w_r_est, w_i_est = variable.get_value()
print('Predicted: ','\t ω_r = ',w_r_est, "\t ω_i =", w_i_est)


#Grafics

x = geom.uniform_points(1000)
y = model.predict(x)

u_pred, v_pred = y[:, 0:1], y[:, 1:2]
"""
with tf.Session() as sess: #pasar el tensor a arreglo de numpy
    array_u = sess.run(func_r(x))
    array_v = sess.run(func_i(x))
"""
f_u = np.array(func_r(x))
f_v = np.array(func_i(x))

plt.figure()
plt.plot(x, f_u, "-", label="u_true")
plt.plot(x, u_pred, "--", label="u_pred")

plt.title("Solutions to Regger-Wheeler Quasinormal Modes real part")
plt.xlabel("y")
plt.ylabel('{\phy}')
plt.legend()
plt.grid()
plt.savefig("Comparison_real_part")

plt.figure()
plt.plot(x, f_v, "-", label="v_true")
plt.plot(x, v_pred, "--", label="v_pred")
plt.title("Solutions to Regger-Wheeler Quasinormal Modes imaginary ")
plt.xlabel("y")
plt.ylabel('{phy}')
plt.legend()
plt.grid()
plt.savefig("Comparison_imaginary_part")


plt.figure()
plt.plot(x, f_u, "-", label="u_true")
plt.plot(x, u_pred, "--", label="u_pred")
plt.plot(x, f_v, "-", label="v_true")
plt.plot(x, v_pred, "--", label="v_pred")
plt.title("Solutions to Regger-Wheeler Quasinormal Modes")
plt.legend()
plt.grid()
plt.savefig("both grafics")
plt.show()
