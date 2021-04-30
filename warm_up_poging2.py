# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:08:48 2021

@author: elske
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch
import matplotlib.pyplot as plt

#%%
class my_model(tf.Module):
    def __init__(self,layers):
        self.depth = len(layers)
        self.activation = tf.nn.tanh
        
        self.model = tf.keras.Sequential()
        
        self.model.add(tf.keras.Input(shape=2))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.Dense(layers[i], activation=self.activation))
    
    def forward(self, x):
        out = self.model(x)
        return out
    
class PINN:
    def __init__(self, layers,N):
        self.model = my_model(layers)
        self.optimizer = tf.keras.optimizers.Adam(
                                          learning_rate=0.5,
                                          beta_1=0.99,
                                          epsilon=1e-1)
        
        self.N = N
    
    def generate_train_data_grid(self):
        # Boundary
        x = tf.linspace(0, 1, self.N)
        bc1 = np.stack((tf.zeros(self.N), x)).T
        bc2 = np.stack((tf.ones(self.N), x)).T
        bc3 = np.stack((x, tf.zeros(self.N))).T
        bc4 = np.stack((x, tf.ones(self.N))).T
        self.boundary = tf.cast( tf.convert_to_tensor(np.stack((bc1, bc2, bc3, bc4))), tf.float32)
        
        # Interior
        xx,yy = tf.meshgrid(x[1:-1], x[1:-1])
        xx = tf.reshape(xx, [1,len(x[1:-1])**2]) 
        yy = tf.reshape(yy, [1,len(x[1:-1])**2])
        self.interior = tf.cast( tf.transpose(tf.concat([xx,yy], axis=0)), tf.float32)
        return self.boundary, self.interior
    
    def u(self, x):
        return self.model.forward(x)
 
    def f(self, x):
        fun = np.zeros(len(x))
        for i in range(len(x)):
            if x[i,1] >= 2*x[i,0]:
                fun[i] = 1 
            else:
                fun[i] = -1
        return tf.convert_to_tensor(fun, dtype= float)
        
    def loss(self):
        self.generate_train_data_grid()
        interior_tensor = self.interior
        boundary_tensor = self.boundary
                
        # Using GradientTape for computing the derivatives of the model output
        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(interior_tensor)
            tape.watch(boundary_tensor)
            
            u_interior = self.u(interior_tensor)
            #u_boundary = self.u(boundary_tensor)
            
            grad_u_interior   = tape.gradient(u_interior, interior_tensor)
            u_interior_x      = grad_u_interior[:,0]
            u_interior_y      = grad_u_interior[:,1]
            u_interior_xx = tape.gradient(u_interior_x, interior_tensor)[:,0]
            u_interior_yy = tape.gradient(u_interior_y, interior_tensor)[:,1]
            
            diffusion     = tf.math.scalar_mul(-0.01,tf.math.add(u_interior_xx,u_interior_yy))
            advection     = tf.math.add(tf.math.scalar_mul((0.5*np.sqrt(0.8)),u_interior_x),tf.math.scalar_mul(1 * np.sqrt(0.8),u_interior_y))
        
        f_values = self.f(interior_tensor)
            
        interior_loss = tf.reduce_mean(tf.square(diffusion + advection - f_values))
        boundary_loss = tf.reduce_mean(tf.square(self.u(boundary_tensor)))
        
        loss = interior_loss + boundary_loss
       
        return loss
    
    def train(self):
        self.generate_train_data_grid()
        
        losses = []
        
        for i in range(100):
            with tf.GradientTape() as tape:
                loss = self.loss()
            variables = self.model.trainable_variables
            grads = tape.gradient(loss,variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            losses.append(loss.numpy())
        return losses
    
    def plot_solution(self):
        #model.dnn.cpu()
        x = tf.linspace(0, 1, self.N)
        xx,yy = tf.meshgrid(x, x)
        xx = tf.reshape(xx, [1,len(x)**2]) 
        yy = tf.reshape(yy, [1,len(x)**2])
        self.data = tf.cast( tf.transpose(tf.concat([xx,yy], axis=0)), tf.float32)
        u = np.array(self.model.forward(self.data))
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
        h = ax.pcolormesh(x, x, tf.reshape(u,[self.N,self.N]), shading="auto")
        fig.colorbar(h)
        #ax.plot(self.interior[:, 0], self.interior[:, 1], "kx", markersize=4, clip_on=False, alpha=0.5)
        #ax.plot(self.boundary[:, 0], self.boundary[:, 1], "kx", markersize=4, clip_on=False, alpha=0.5)        
        plt.show()
    

layers = [10,10,10,1]   
N = 50
PINN_model = PINN(layers,N) 
losses = PINN_model.train()
PINN_model.plot_solution()
