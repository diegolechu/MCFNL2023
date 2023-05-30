# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:01:01 2023

@author: Chema & Benji & Lechuga
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.fft import fft, fftfreq

#%%
#Parametros electromagneticos (fijan unidades):
eps = 1.0 
mu = 1.0
c0 = 1/np.sqrt(eps*mu)
eta0 = np.sqrt(mu/eps)
#Discretización del dominio espacio temporal:
L = 40
n_puntos = 400
x = np.linspace(0.0, L, n_puntos)
xDual = (x[1:] + x[:-1])/2
dx = x[1] - x[0]
CFL = 0.9
dt = CFL * dx / c0
tmax = 55
tRange = np.arange(0, tmax, dt)
#Parámetros del panel conductor:
panel1 = 20
d = 5
panel2 = panel1 + d
val_sigma = 0.5 #aqui el valor de la conductividad en el panel
sigma = np.zeros(x.shape)
sigma[x>=panel1] = val_sigma
sigma[x>=panel2] = 0
val_eps = 1.0 #aqui el valor de permitividad en el medio 
eps = np.ones(x.shape)
eps[x>=panel1] = val_eps
eps[x>=panel2] = 1.0
complex_vec = np.vectorize(complex)
def compl_eps(omega):
    return complex(val_eps, -val_sigma/omega)
def eta(omega):
    return np.sqrt(mu/compl_eps(omega))
def gamma(omega):
    return complex(0, omega*np.sqrt(mu*compl_eps(omega)))
def phi_11(omega):
    return np.cosh(gamma(omega)*d)
def phi_12(omega):
    return eta(omega)*np.sinh(gamma(omega)*d)
def phi_21(omega):
    return np.sinh(gamma(omega)*d)/eta(omega)
def phi_22(omega):
    return np.cosh(gamma(omega)*d)
def T_teo(omega):
    return 2*eta0/(phi_11(omega)*eta0 + phi_12(omega) + phi_21(omega)*eta0*eta0 + eta0*phi_22(omega))
def R_teo(omega):
    num = phi_11(omega)*eta0 + phi_12(omega) - phi_21(omega)*eta0*eta0 - eta0*phi_22(omega)
    den = phi_11(omega)*eta0 + phi_12(omega) + phi_21(omega)*eta0*eta0 + eta0*phi_22(omega)
    return num/den
#Parámetros para el esquema fdtd con conductividad:
alpha = sigma/2 + eps/dt
beta  = sigma/2 - eps/dt
#Esto controla la velocidad de visualización:
t_pause = 0.02
#Para el plot:
ymin = -1.1
ymax = 1.1
plot = False
#Parametros del paquete gaussiano:
x0 = 5
s0 = 0.7
#Puntos donde se va a grabar el valor de e:
record_left = 5
record_right = 35
#Tiempo a partir del cual se empiezan a guardar valores:
time_record = 15
#%%
k = 5
e = np.exp( -(x-x0)**2 / (2*s0**2) ) #* np.cos(k*(x-x0))
h = np.exp( -(xDual - c0*dt/2 - x0)**2 / (2*s0**2) ) #* np.cos(k*(xDual-x0-c0*dt/2))
refl = []
trans = []

#Esto para ver que índices corresponden con record rigth y left:
for i, x_ in enumerate(x):
    if x_ > record_left:
        index_left = i
        break
for i, x_ in enumerate(x):
    if x_ > record_right:
        index_right = i
        break

#Aquí se guardan el valor máximo y posición de e para ver el skin depth
maxs = []
indices =  []

plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 90
plt.figure('Panel conductivo')

for t in tRange:
    eMur_izq = e[1]
    eMur_der = e[-2]
    e[1:-1] = (-(h[1:]-h[:-1])/dx - beta[1:-1]*e[1:-1])/alpha[1:-1]
    e[0] = eMur_izq + (c0*dt-dx)/(c0*dt+dx)*(e[1]-e[0]) # Mur izquierda
    e[-1] = eMur_der + (c0*dt-dx)/(c0*dt+dx)*(e[-2]-e[-1]) # Mur derecha
    h[:] = (-dt/dx/mu) * (e[1:] - e[:-1]) + h[:]
    if t>time_record:
        refl.append(e[index_left])
        trans.append(e[index_right])
        if plot:
            plt.plot([record_left, record_right], [1, 1], marker='o', linestyle='')
    maxs.append(max(e))
    indices.append(np.argmax(e))
    if plot:
        plt.plot(x, e, '.-', label='E', color='Orange')
        plt.axvline(panel1, color='black')
        plt.axvline(panel2, color='black')
        plt.plot(xDual, h, '.-', label='H', color='Blue')
        plt.legend()
        plt.grid()
        plt.ylim(ymin, ymax)
        plt.xlim(x[0],x[-1])
        plt.show()
        #plt.pause(t_pause)
        plt.cla()
if plot:
    plt.show()

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 900
plt.figure('FFT componentes')
time = [t for t in tRange if t>time_record]
n = len(time)
transformada_refl = np.abs(fft(refl))[:n//2]

frecuencia_ = fftfreq(n, dt)
frecuencia = frecuencia_[:n//2]
transformada_trans = np.abs(fft(trans))[:n//2]
#Transformada de Fourier teórica del paquete incidente
incid_teorico = [s0*np.exp(-(2*np.pi*f*s0/(np.sqrt(2)*c0))**2)/c0 for f in frecuencia]
#incid_teorico = [s0*(np.sinh(2*k*s0*s0*2*np.pi*f/c0)+np.cosh(2*k*s0*s0*2*np.pi*f/c0)+1)*np.exp(-s0*s0*(c0*k+2*np.pi*f)**2/(2*c0*c0))/(2*c0) for f in frecuencia]
#Normalizamos las fft
transformada_refl *= np.abs(R_teo(frecuencia[1]))*incid_teorico[0]/transformada_refl[0]
transformada_trans *= np.abs(T_teo(frecuencia[1]))*incid_teorico[0]/transformada_trans[0]
plt.plot(frecuencia, transformada_refl, label='Reflejado', color='blue')
plt.plot(frecuencia, transformada_trans, label='Transmitido', color='red')
plt.plot(frecuencia, incid_teorico, label='Incidente teórico', color='black')
plt.xlim(0, 1)
plt.xlabel('Frecuencia')
plt.grid()
plt.legend()

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 900

T = transformada_trans/incid_teorico
R = transformada_refl/incid_teorico

plt.figure('Coeficientes R y T')
plt.plot(frecuencia, T, label='T', color='orange', linewidth=3)
plt.plot(frecuencia, R, label='R', color='turquoise', linewidth=3)
plt.plot(frecuencia, [np.abs(T_teo(2*np.pi*f)) for f in frecuencia],
         label='T teórico', color='springgreen', linestyle='dashed', linewidth=3)
plt.plot(frecuencia, [np.abs(R_teo(2*np.pi*f)) for f in frecuencia],
         label='R teórico', color='orchid', linestyle='dashed', linewidth=3)
#plt.plot(frecuencia, T*T+R*R, label=r'$T^2+R^2$', color='yellowgreen', linewidth=3)
x_lim = 1.0
y_lim = 1.2
plt.xlabel('Frecuencia')
plt.xlim(0, x_lim)
plt.ylim(0, y_lim)
plt.axhline(1, 0, x_lim, linestyle='dashed', color='black')
plt.legend()


#%%
#Skin depth
for i, maximo in enumerate(maxs):
    if maximo < (1/np.e):
        index_delta = i
        break
delta = x[indices[index_delta]] - panel1
print(f'Skin depth={delta}')