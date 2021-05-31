# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:31:37 2021

@author: elske
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
#from models_tf import Eikonal2DnetCV2
from model_cardiac3 import Eikonal2DnetCV2
#from model_cardiac2 import my_Eikonal2D

#%% Exact solution

def exact(X, Y):
    return np.minimum(np.sqrt(X**2 + Y**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2))

def CVexact(X, Y):
    mask = np.less_equal(np.sqrt(X**2 + Y**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2))
    return mask*1.0 + ~mask*1.0/0.7

#%% Plotting the exact solutions and training points
N_grid = 50         
x = y = np.linspace(0,1,N_grid)[:,None]
N_train = 50

X_m, Y_m = np.meshgrid(x,y)             #creating meshgrid for exact solution
X = X_m.flatten()[:,None]
Y = Y_m.flatten()[:,None]
T = exact(X,Y)
CV = CVexact(X,Y)

X_train_all = lhs(2, N_train)           #creating training points
X_train = X_train_all[:,:1]
Y_train = X_train_all[:,1:]
T_train = exact(X_train, Y_train)

fig = plt.figure()                      #Plot of the activation time
fig.set_size_inches((12,5))
plt.subplot(121)
plt.contourf(X_m, Y_m, T.reshape(X_m.shape))
plt.colorbar()
plt.scatter(X_train, Y_train, facecolors = 'none', edgecolor = 'k') 
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('exact activation times')

plt.subplot(122)                        #plot of the Velocity
plt.contourf(X_m, Y_m, CV.reshape(X_m.shape))
plt.colorbar()
plt.scatter(X_train, Y_train, facecolors = 'none', edgecolor = 'k') 
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('exact conduction velocity')
     
plt.tight_layout()

#%% Architecture of the network

layers = [2,20,20,20,20,20,1]
CVlayers = [2,5,5,5,5,1]

# collocation points
X_pde = X
Y_pde = Y

# maximum value for the conduction velocity
CVmax = 1.5

#%% model used in article
model = Eikonal2DnetCV2(X_pde, Y_pde, X_train, Y_train, T_train, 
                        layers,CVlayers, C = CVmax, alpha = 1e-5, alphaL2 = 1e-9)

#%% Own model

model = my_Eikonal2D(X_pde, Y_pde, X_train, Y_train, T_train, 
                        layers,CVlayers, C = CVmax, alpha = 1e-5, alphaL2 = 1e-9, epochs=50000)


#%%
#losses = model.train_Adam_minibatch(50000)
model.train_Adam(10000)
#model.train() # BFGS training

#%%
T_star, CV_star = model.predict(X,Y)

fig = plt.figure()
fig.set_size_inches((12,5))
plt.subplot(121)
plt.contourf(X_m, Y_m, T_star.reshape(X_m.shape))
plt.colorbar()
plt.scatter(X_train, Y_train, facecolors = 'none', edgecolor = 'k') 
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('predicted activation times')

plt.subplot(122)
plt.contourf(X_m, Y_m, CV_star.reshape(X_m.shape))
plt.colorbar()
plt.scatter(X_train, Y_train, facecolors = 'none', edgecolor = 'k') 
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('predicted conduction velocity')
     
plt.tight_layout()