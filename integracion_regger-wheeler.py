
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

w_r = 0.5
w_i = 1.5

def ode(X,y):
    u, A, v ,B = X
    return [A,
            (2 * y )/(1-y**2) * A
            - ( (w_r**2 - w_i**2 - 0.5 * (1-y**2))/(1-y**2)**2) * u
            + (2 * w_i * w_r * v)/(1-y**2)**2,
            B,
            (2 * y )/(1-y**2) * B
            -( (w_r**2 - w_i**2 - 0.5 * (1-y**2)) / (1-y**2)**2) * v
    ]
# time points
time = np.linspace(0, 5, 100)

# solve ODE
y0 = [1.0, 1.0, 1.0, 5.0]
sol = odeint(ode,y0,time)

# Reacomodar los arrays para más tarde
time = time.reshape(len(time),1)
sol_u = sol[:, 0:1].reshape(len(time),1)
sol_v = sol[:, 2:3].reshape(len(time),1)

plt.plot(time, sol_u + sol_v, label='u+v')
#plt.plot(time, sol_v, label='v')
plt.legend(loc='right')
plt.xlabel('t')
plt.show()