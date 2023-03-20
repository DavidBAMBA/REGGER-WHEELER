import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt

#parametros a encontrar
w_r = dde.Variable(1.0) # Variable para ajustar
w_i = dde.Variable(-1.0) # Variable para ajustar
bdr_inf = 0.9

#funciones para extraer valores de la sulucion y agregarlos al 
#entrenamiento para ajustar las w
def gen_traindata_r(N):
    yvals = (np.linspace(-bdr_inf, bdr_inf, N)).reshape(N, 1)
    phi_vals = func_r(yvals)
    return yvals, phi_vals

def gen_traindata_i(N):
    yvals = (np.linspace(-bdr_inf, bdr_inf, N)).reshape(N, 1)
    phi_vals = func_i(yvals)
    return yvals, phi_vals

#Soluciones analiticas real e imaginaria 
def func_r(y):
    '''
    Soluciones dependiendo el modo 
    n = 0 : (tf.math.sqrt(tf.math.cosh(x)) * tf.math.cos(tf.math.log(tf.math.cosh(x)) * 0.5))
    n = 1 : n(0) * tf.math.sinh(x) 
    n = 2 : n(0) * (3.0*(tf.math.sinh(x)**2) + 1.0) //// im : tf.math.sinh(x)**2
    n = 3 : n(0) * (5/3 * tf.math.sinh(x)**3 + tf.math.sinh(x))
    '''
    x = tf.math.atanh(y)
    return  (tf.math.sqrt(tf.math.cosh(x)) * tf.math.cos(tf.math.log(tf.math.cosh(x)) * 0.5)) * (3.0*(tf.math.sinh(x)**2) + 1.0)

def func_i(y):
    '''
    Soluciones dependiendo el modo 
    n = 0 : (tf.math.sqrt(tf.math.cosh(x)) * tf.math.sin(tf.math.log(tf.math.cosh(x)) * 0.5))
    n = 1 : n(0) * tf.math.sinh(x) 
    n = 2 : n(0) * tf.math.sinh(x)**2
    n = 3 : n(0) * (1/3 * tf.math.sinh(x)**3)
    '''
    x = tf.math.atanh(y)
    return  tf.math.sqrt(tf.math.cosh(x)) * tf.math.sin(tf.math.log(tf.math.cosh(x)) * 0.5) * tf.math.sinh(x)**2
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
           + (w_r**2 - w_i**2 - 0.5 * (1 - Y**2)) * v + 2 * w_r * w_i * u
    )

    return  [f_r, f_i] 

#Definicion del intervalo de solucion
geom = dde.geometry.Interval(-bdr_inf,bdr_inf)

#condiciones de frontera
ic_u = dde.icbc.DirichletBC(geom, func_r,  lambda _,on_boundary: on_boundary, component=0)
ic_v = dde.icbc.DirichletBC(geom, func_i,  lambda _,on_boundary: on_boundary, component=1)

#datos experimentales para hayar el parametro
y_values, u_values = gen_traindata_r(50) 
exp_data_r = dde.icbc.PointSetBC(y_values, u_values, component=0) # valores w real 

y_values, v_values = gen_traindata_i(50)
exp_data_i = dde.icbc.PointSetBC(y_values, v_values, component=1) #valores w imaginario

#Datos De entrenamiento
data = dde.data.PDE(geom, PDE, [ic_u, ic_v ,exp_data_r, exp_data_i], num_domain=100, num_boundary=30, num_test=200,
                    train_distribution='pseudo', anchors=y_values)

#Red neuronal
layer_size = [1] + [20] * 3 + [2]
activation = 'tanh'
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

#Modelo
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([w_r,w_i], period=1000)

# Optimizador Adam para ajustar el valor de w^2
#pde_resampler = dde.callbacks.PDEPointResampler(period=100)

model.compile("adam", lr=1e-3, external_trainable_variables=[w_i,w_r])
loss_history, train_state = model.train(iterations=10000, callbacks=[variable])

# Optimizador L-BFGS para ajustar la ecuación diferencial (no ajusta w^2)
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS", external_trainable_variables=[w_i,w_r])
losshistory, train_state = model.train(callbacks=[variable])

#Parametros calculados
N = 2
W_r_esp = 1/2
W_i_esp = -(N + 1/2)
print('Esperados: ','W_r_est = ',W_r_esp, "  W_i_est =", W_i_esp)
w_r_est, w_i_est = variable.get_value()
print('Predichos: ','w_r_ = ',w_r_est, "  w_i =", w_i_est)


#Graficas
dde.utils.external.plot_loss_history(losshistory)

x = geom.uniform_points(1000)
y = model.predict(x)

u_pred, v_pred = y[:, 0:1], y[:, 1:2]

with tf.Session() as sess: #pasar el tensor a arreglo de numpy
    array_u = sess.run(func_r(x))
    array_v = sess.run(func_i(x))

f_u = np.array(array_u)
f_v = np.array(array_v)

plt.figure()
plt.plot(x, f_u, "-", label="u_true")
plt.plot(x, u_pred, "--", label="u_pred")

plt.title("Solutions to Regger-Wheeler Quasinormal Modes real part")
plt.xlabel("y")
plt.ylabel('{\phy}')
plt.legend()
plt.grid()
plt.savefig("Comparison_real_part")
plt.show()

plt.figure()
plt.plot(x, f_v, "-", label="v_true")
plt.plot(x, v_pred, "--", label="v_pred")
plt.title("Solutions to Regger-Wheeler Quasinormal Modes imaginary ")
plt.xlabel("y")
plt.ylabel('{phy}')
plt.legend()
plt.grid()
plt.savefig("Comparison_imaginary_part")
plt.show()

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


