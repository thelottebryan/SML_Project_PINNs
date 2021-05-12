# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

#%%
   
class PINN(object):
    def __init__(self, layers, N, Nbc=[20,20,20,20], random=False,epochs=100,int_weight=1,bound_weight=1):
        
        self.depth = len(layers)
        self.activation = tf.nn.tanh
        
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=2))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.Dense(layers[i], activation=self.activation))
        
        self.optimizer = tf.keras.optimizers.Adam(
                                          learning_rate=0.1,
                                          beta_1=0.99,
                                          epsilon=1e-1)
        
        self.N = N
        self.Nbc = Nbc
        self.random=random
        self.epochs=epochs
        self.int_weight=int_weight
        self.bound_weight=bound_weight
        
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
    
    def generate_train_data_grid(self):
        # Boundary
        xbc1 = tf.linspace(0, 1, self.Nbc[0])
        xbc2 = tf.linspace(0, 1, self.Nbc[1])
        xbc3 = tf.linspace(0, 1, self.Nbc[2])
        xbc4 = tf.linspace(0, 1, self.Nbc[3])
        bc1 = np.stack((tf.zeros(self.Nbc[0]), xbc1)).T
        bc2 = np.stack((tf.ones(self.Nbc[1]), xbc2)).T
        bc3 = np.stack((xbc3, tf.zeros(self.Nbc[2]))).T
        bc4 = np.stack((xbc4, tf.ones(self.Nbc[3]))).T
        self.boundary = tf.cast(tf.convert_to_tensor(np.concatenate((bc1, bc2, bc3, bc4))),tf.float32)
        
        # Interior
        xint = tf.linspace(0, 1, self.N)
        xx,yy = tf.meshgrid(xint[1:-1], xint[1:-1])
        xx = tf.reshape(xx, [1,len(xint[1:-1])**2]) 
        yy = tf.reshape(yy, [1,len(xint[1:-1])**2])
        self.interior = tf.cast( tf.transpose(tf.concat([xx,yy], axis=0)), tf.float32)
        
        return self.boundary, self.interior
    
    def generate_random_data_grid(self):
        #Boundary
        xbc1 = tf.linspace(0, 1, self.Nbc[0])
        xbc2 = tf.linspace(0, 1, self.Nbc[1])
        xbc3 = tf.linspace(0, 1, self.Nbc[2])
        xbc4 = tf.linspace(0, 1, self.Nbc[3])
        
        randbc1  = (np.random.rand(self.Nbc[0])-1/2)/(self.Nbc[0]-1)
        randbc2  = (np.random.rand(self.Nbc[1])-1/2)/(self.Nbc[1]-1)
        randbc3  = (np.random.rand(self.Nbc[2])-1/2)/(self.Nbc[2]-1)
        randbc4  = (np.random.rand(self.Nbc[3])-1/2)/(self.Nbc[3]-1)
        
        randbc1[[0,-1]]=0
        randbc2[[0,-1]]=0
        randbc3[[0,-1]]=0
        randbc4[[0,-1]]=0
        
        bc1 = np.stack((tf.zeros(self.Nbc[0]), xbc1+randbc1)).T
        bc2 = np.stack((tf.ones(self.Nbc[1]), xbc2+randbc2)).T
        bc3 = np.stack((xbc3+randbc3,tf.zeros(self.Nbc[2]))).T
        bc4 = np.stack((xbc4+randbc4,tf.ones(self.Nbc[3]))).T
        
        self.boundary = tf.cast( tf.convert_to_tensor(np.concatenate((bc1, bc2, bc3, bc4))), tf.float32)

        #Interior
        randint = (np.random.rand(self.N,2)-1/2)/(self.N-1)
        xint=tf.linspace(0,1,self.N)+randint[:,0]
        yint=tf.linspace(0,1,self.N)+randint[:,1]
        xx,yy = tf.meshgrid(xint[1:-1], yint[1:-1])
        xx = tf.reshape(xx, [1,len(xint[1:-1])**2]) 
        yy = tf.reshape(yy, [1,len(yint[1:-1])**2])
        self.interior = tf.cast( tf.transpose(tf.concat([xx,yy], axis=0)), tf.float32)
        
        return self.boundary, self.interior
    
    def u(self, x):
        return self.model.(x)
 
    def f(self, x):
        fun = np.zeros(len(x))
        for i in range(len(x)):
            if x[i,1] >= 2*x[i,0]:
                fun[i] = 1 
            else:
                fun[i] = -1
        return tf.convert_to_tensor(fun, dtype= float)
        
    def loss(self):
        if self.random:
            self.generate_random_data_grid()
        else:    
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
        
        del tape 
        
        diffusion     = tf.math.scalar_mul(-0.01,tf.math.add(u_interior_xx,u_interior_yy))
        advection     = tf.math.add(tf.math.scalar_mul((0.5*np.sqrt(0.8)),u_interior_x),tf.math.scalar_mul(1 * np.sqrt(0.8),u_interior_y))
        
        f_values = self.f(interior_tensor)
        
        self.interior_loss = tf.reduce_mean(tf.square(diffusion + advection - f_values))
        self.boundary_loss = tf.reduce_mean(tf.square(self.u(boundary_tensor)))
        
        self.total_loss = self.interior_loss+self.boundary_loss
       
        return self.int_weight*self.interior_loss + self.bound_weight*self.boundary_loss
    
    def train(self):
        if self.random:
            self.generate_random_data_grid()
        else:
            self.generate_train_data_grid()
        
        interior_losses = []
        boundary_losses = []
        total_losses = []
        
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                loss = self.loss()
            variables = self.model.trainable_variables
            grads = tape.gradient(loss,variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            interior_losses.append(self.interior_loss.numpy())
            boundary_losses.append(self.boundary_loss.numpy())
            total_losses.append(self.total_loss.numpy())
            print('Run ',i)
            
            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % 10 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                print("loss {:1.2f}".format(loss.numpy()))
                
        return interior_losses, boundary_losses, total_losses
    
    def plot_solution(self):
        #model.dnn.cpu()
        x = tf.linspace(0, 1, self.N)
        xx,yy = tf.meshgrid(x, x)
        xx = tf.reshape(xx, [1,len(x)**2]) 
        yy = tf.reshape(yy, [1,len(x)**2])
        self.data = tf.cast( tf.transpose(tf.concat([xx,yy], axis=0)), tf.float32)
        u = np.array(self.model(self.data))
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        
        h = ax.pcolormesh(x, x, tf.reshape(u,[self.N,self.N]), shading="auto")
        fig.colorbar(h)
        #ax.plot(self.interior[:, 0], self.interior[:, 1], "kx", markersize=4, clip_on=False, alpha=0.5)
        #ax.plot(self.boundary[:, 0], self.boundary[:, 1], "kx", markersize=4, clip_on=False, alpha=0.5)        
        plt.show()
    

layers = [10,10,1]   
epochs = 500
N = 100
Nbc = [75,75,150,150]
random = True
int_weight=1
bound_weight=5
PINN_model = PINN(layers,N,Nbc,random,epochs,int_weight,bound_weight) 
losses = PINN_model.train()
PINN_model.plot_solution()
#%%
x=np.arange(epochs)
plt.plot(x,losses[0],x,losses[1],x,losses[2])
plt.legend(['Interior Loss','Boundary Loss','Total Loss'])
plt.show()
