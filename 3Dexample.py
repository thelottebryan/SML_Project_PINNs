# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from models_tf_3D import Eikonal3DnetCV2
from mpl_toolkits import mplot3d

#%%

def exact(X, Y, Z):
    return np.minimum(np.sqrt(X**2 + Y**2 + Z**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2) + (Z - 1)**2)

def CVexact(X, Y, Z):
    mask = np.less_equal(np.sqrt(X**2 + Y**2 + Z**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2 + (Z - 1)**2))
    return mask*1.0 + ~mask*1.0/0.7

#%%

N_grid = 50         
x = y = z = np.linspace(0,1,N_grid)[:,None]

N_train = 50

X_m, Y_m, Z_m = np.meshgrid(x,y,z)
X = X_m.flatten()[:,None]
Y = Y_m.flatten()[:,None]
Z = Z_m.flatten()[:,None]
T = exact(X,Y,Z)
CV = CVexact(X,Y,Z)

X_train_all = lhs(3, N_train)
X_train = X_train_all[:,0][:,None]
Y_train = X_train_all[:,1][:,None]
Z_train = X_train_all[:,2][:,None]
T_train = exact(X_train, Y_train, Z_train)
#%%
#fig = plt.figure()
#fig.set_size_inches((12,5))

#ax=plt.axes(projection='3d')

#plt.subplot(121)
#ax.contour3D(X_m, Y_m, Z_m, T.reshape(X_m.shape))
#plt.colorbar()
#ax.scatter_3d(X_train, Y_train, Z_train, facecolors = 'none', edgecolor = 'k') 
#ax.xlim([0,1])
#ax.ylim([0,1])
#ax.zlim([0,1])
#ax.title('exact activation times')
#%%
#plt.subplot(122)
#plt.contour3D(X_m, Y_m, Z_m, CV.reshape(X_m.shape))
#plt.colorbar()
#plt.scatter_3d(X_train, Y_train, Z_train, facecolors = 'none', edgecolor = 'k') 
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.zlim([0,1])
#plt.title('exact conduction velocity')
     
#plt.tight_layout()

#%%
layers = [3,20,20,20,20,20,1]
CVlayers = [3,5,5,5,5,1]

# collocation points
X_pde = X
Y_pde = Y
Z_pde = Z

# maximum value for the conduction velocity
CVmax = 1.5

model = Eikonal3DnetCV2(X_pde, Y_pde, Z_pde, X_train, Y_train, Z_train, T_train, 
                        layers,CVlayers, C = CVmax, alpha = 1e-7, alphaL2 = 1e-9)
#%%

model.train_Adam_minibatch(1000, size = 100)
#model.train() # BFGS training
#%%

T_star, CV_star = model.predict(X,Y,Z)

#fig = plt.figure()
#fig.set_size_inches((12,5))
#plt.subplot(121)
#plt.contourf(X_m, Y_m, T_star.reshape(X_m.shape))
#plt.colorbar()
#plt.scatter(X_train, Y_train, facecolors = 'none', edgecolor = 'k') 
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.title('predicted activation times')

#plt.subplot(122)
##plt.colorbar()
#plt.scatter(X_train, Y_train, facecolors = 'none', edgecolor = 'k') 
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.title('predicted conduction velocity')
     
#plt.tight_layout()