# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:11:38 2021

@author: elske
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from cardiac_3Dmodel import my_Eikonal3D

#%%

def exact(X, Y, Z):
    return np.minimum(np.sqrt(X**2 + Y**2 + Z**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2) + (Z - 1)**2)
    # return np.sqrt((X-0.5)**2 + (Y)**2 + (Z-0.5)**2)
def CVexact(X, Y, Z):
    mask = np.less_equal(np.sqrt(X**2 + Y**2 + Z**2), 0.7*np.sqrt((X - 1)**2 + (Y - 1)**2 + (Z - 1)**2))
    return mask*0.5 + ~mask*1.0/0.7


##Punten en trainingspunten genereren
N_grid = 50         
x = y = z = np.linspace(0,1,N_grid)
X_m, Y_m, Z_m = np.meshgrid(x,y,z)

X = np.array([X_m[0,:,:], X_m[0,:,:], X_m[:,0,:], X_m[:,-1,:], X_m[:,:,0], X_m[:,:,0]]).reshape(-1,50).flatten()
Y = np.array([Y_m[0,:,:], Y_m[-1,:,:], Y_m[:,0,:], Y_m[:,0,:], Y_m[:,0,:], Y_m[:,0,:]]).reshape(-1,50).flatten()
Z = np.array([Z_m[0,:,:], Z_m[0,:,:], Z_m[0,:,:], Z_m[0,:,:], Z_m[:,:,0], Z_m[:,:,-1]]).reshape(-1,50).flatten()
T = exact(X,Y,Z)
CV = CVexact(X,Y,Z)

index_value = random.sample(range(len(T)), 200)
X_train = X[index_value][:, None]
Y_train = Y[index_value][:, None]
Z_train = Z[index_value][:, None]
T_train = T[index_value][:, None]

## T en CV plotten in de kubus
fig = plt.figure(1)
ax = plt.axes(projection = '3d')
for i in range(0,300,50):
    ax.plot_surface(X.reshape(300,50)[i:i+50,:], Y.reshape(300,50)[i:i+50,:], Z.reshape(300,50)[i:i+50,:], facecolors=cm.viridis(T[i*50:i*50+50*50].reshape(50,50)))
    
m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(T)
plt.colorbar(m)
ax.scatter(X_train, Y_train, Z_train, color = 'k')
plt.show()

fig = plt.figure(2)
ax = plt.axes(projection = '3d')
for i in range(0,300,50):
    ax.plot_surface(X.reshape(300,50)[i:i+50,:], Y.reshape(300,50)[i:i+50,:], Z.reshape(300,50)[i:i+50,:], facecolors=cm.viridis(CV[i*50:i*50+50*50].reshape(50,50)))

m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(CV)
plt.colorbar(m)
ax.scatter(X_train, Y_train, Z_train, color = 'k')
plt.show()


##Model parameters
layers = [3,20,20,20,20,20,1]
CVlayers = [3,5,5,5,5,1]

# collocation points
X_pde = X[:,None]
Y_pde = Y[:,None]
Z_pde = Z[:,None]

epochs_value_T = 1000
epochs_value_CV = 1000


# maximum value for the conduction velocity
CVmax = 1.5

#%% Model aanroepen
model = my_Eikonal3D(X_pde, Y_pde, Z_pde, X_train, Y_train, Z_train, T_train, 
                        layers, CVlayers, C = CVmax, alpha = 1e-7, alphaL2 = 1e-9, epochs_T=epochs_value_T, epochs_CV=epochs_value_CV)

#%% T trainen
T1_losses, T23_losses, total_losses = model.trainT()
#%% CV trainen
T1_lossesCV, T23_lossesCV, total_lossesCV = model.trainCV()

#%% Predictions plotten in de kubus
T_star, CV_star = model.predict(X[:,None],Y[:,None],Z[:,None])

fig = plt.figure(1)
ax = plt.axes(projection = '3d')
for i in range(0,300,50):
    ax.plot_surface(X.reshape(300,50)[i:i+50,:], Y.reshape(300,50)[i:i+50,:], Z.reshape(300,50)[i:i+50,:], facecolors=cm.jet(T_star[i*50:i*50+50*50,:].reshape(50,50)))
    if i == 250:
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(T_star)
        plt.colorbar(m)
ax.scatter(X_train, Y_train, Z_train, color = 'k')
plt.show()

fig = plt.figure(2)
ax = plt.axes(projection = '3d')
for i in range(0,300,50):
    ax.plot_surface(X.reshape(300,50)[i:i+50,:], Y.reshape(300,50)[i:i+50,:], Z.reshape(300,50)[i:i+50,:], facecolors=cm.viridis(CV_star[i*50:i*50+50*50,:].reshape(50,50)))
    if i== 0:
        m = cm.ScalarMappable(cmap=cm.viridis)
        m.set_array(CV_star)
        plt.colorbar(m)
ax.scatter(X_train, Y_train, Z_train, color = 'k')
plt.show()







