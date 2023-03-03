import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt

#parametros a encontrar
w_r = dde.Variable(2.0) # Variable para ajustar
w_i = dde.Variable(-1.0) # Variable para ajustar
#frecuencia como suma de real e imaginaria
#W2 = w_r**2 + w_i**2

#funciones para extraer valores de la sulucion y agregarlos al 
#entrenamiento para ajustar las w
def gen_traindata_r(N):
    yvals = (np.linspace(-0.9, 0.9, N)).reshape(N, 1)
    phi_vals = func_r(yvals)
    return yvals, phi_vals

def gen_traindata_i(N):
    yvals = (np.linspace(-0.9, 0.9, N)).reshape(N, 1)
    phi_vals = func_i(yvals)
    return yvals, phi_vals

#Soluciones analiticas real e imaginaria 
def func_r(y):
    x = tf.math.atanh(y)
    return tf.math.sqrt(tf.cosh(x)) * tf.math.cos(tf.math.log(tf.math.cosh(x))) #* np.sinh(y)

def func_i(y):
    x = tf.math.atanh(y)
    return tf.math.sqrt(tf.cosh(x)) * tf.math.sin(tf.math.log(tf.math.cosh(x))) #* np.sinh(y)

#Ecuacion diferencial
def PDE(y,x):
    # funcion real e imaginaria
    u, v = x[:, 0:1], x[:, 1:2]

    #derivadas de cada funcion
    u_y = dde.grad.jacobian(x, y, i=0, j=0)
    v_y = dde.grad.jacobian(x, y, i=1, j=0)
    u_yy = dde.grad.hessian(x, y, component=0, i=0, j=0)
    v_yy = dde.grad.hessian(x, y, component=1, i=0, j=0)  

    Y = y[:,0:1] #Variable independiente (respecto a la que se deriva)

    # Division en parte real e imaginaria de la ecuaciones
    f_r = ( (1 - Y**2)**2 * u_yy
           - 2 * Y * (1 - Y**2) * u_y
           + (w_r**2 - w_i**2 - 0.5 * (1 - Y**2)) * u - 2 * w_r * w_i * v
    )
    f_i = ( (1 - Y**2)**2 * v_yy
           - 2 * Y * (1 - Y**2) * v_y
           + (w_r**2 - w_i**2 - 0.5 * (1 - Y**2)) * v
    )

    return  [f_r, f_i] 

#Definicion del intervalo de solucion
geom = dde.geometry.Interval(-0.99,0.99)

#condiciones de frontera
ic_u = dde.icbc.DirichletBC(geom, func_r,  lambda _,on_boundary: on_boundary, component=0)
ic_v = dde.icbc.DirichletBC(geom, func_i,  lambda _,on_boundary: on_boundary, component=1)

#datos experimentales para hayar el parametro
y_values, u_values = gen_traindata_r(500) 
exp_data_r = dde.icbc.PointSetBC(y_values, u_values, component=0) # valores w real 

y_values, v_values = gen_traindata_i(500)
exp_data_i = dde.icbc.PointSetBC(y_values, v_values, component=1) #valores w imaginario

#Datos De entrenamiento
data = dde.data.PDE(geom, PDE, [ic_u, ic_v, exp_data_r, exp_data_i], 1000, 200, num_test=100,
                    train_distribution='pseudo', anchors=y_values)

#Red neuronal
layer_size = [1] + [100] * 4 + [2]
activation = 'tanh'
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

#Modelo
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([w_r,w_i], period=1000)

# Optimizador Adam para ajustar el valor de w^2
model.compile("adam", lr=1e-3, external_trainable_variables=[w_i,w_r])
loss_history, train_state = model.train(iterations=14000, callbacks=[variable])

# Optimizador L-BFGS para ajustar la ecuación diferencial (no ajusta w^2)
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS", external_trainable_variables=[w_i,w_r])
losshistory, train_state = model.train(callbacks=[variable])

#Graficas
dde.utils.external.plot_loss_history(losshistory)

t = geom.uniform_points(1000)
y = model.predict(t)
W_exp = 1/2 # w= \pm 1 -i(n+1/2) usando n=0 por que si :V

plt.figure()
plt.plot(t, y[:, 0],"-", label="real_solution")
plt.plot(t, y[:, 1],"-", label="imaginary solution")

#plt.plot(t, func_r,'-',label="Exacta u")
#plt.plot(t, func_i,'-',label="Exacta v")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

w_r_est, w_i_est = variable.get_value()
print('w_r_ = ',w_r_est, "  w_i =", w_i_est)
W_r_est = 1/2
W_i_est = -1/2
print('W_r_est = ',W_r_est, "  W_i_est =", W_i_est)