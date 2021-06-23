# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:47:18 2021

@author: elske
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
#%%
   
class my_Eikonal3D():
        # Initialize the class
    def __init__(self, x, y, z, x_e, y_e, z_e, T_e, layers, CVlayers, C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, epochs_T=100, epochs_CV=1000):
        
        X = np.concatenate([x, y, z], 1)
        #X_e = np.concatenate([x_e, t_e], 1)        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        #self.X_e = X_e
        
        self.x = x  #location of the collocation points to enforce the residual penalty (random samples). Each of them must be of shape N_colloc, 1
        self.y = y
        self.z = z
        
        self.T_e = T_e  #activation times data. Must be of shape N_data, 1
        self.x_e = x_e  #location of the data points (exact values). Each of them must be of shape N_data, 1
        self.y_e = y_e
        self.z_e = z_e
        
        self.x_tf = tf.Variable(self.x, dtype=tf.float32)
        self.y_tf = tf.Variable(self.y, dtype=tf.float32)
        self.z_tf = tf.Variable(self.z, dtype=tf.float32)
        
        self.T_e_tf = tf.Variable(self.T_e, dtype=tf.float32)
        self.x_e_tf = tf.Variable(self.x_e, dtype=tf.float32)
        self.y_e_tf = tf.Variable(self.y_e, dtype=tf.float32)
        self.z_e_tf = tf.Variable(self.z_e, dtype=tf.float32)
        
        self.layers = layers        
        self.CVlayers = CVlayers
        self.depth = len(layers)
        self.CVdepth =  len(CVlayers)
        
        self.activation = 'tanh'
        
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=3))
        for i in range(1,self.depth):
            self.model.add(tf.keras.layers.Dense(layers[i], activation = self.activation))
        
        self.CVmodel = tf.keras.models.Sequential()
        self.CVmodel.add(tf.keras.Input(shape=3))
        for i in range(1,self.CVdepth-1):
            self.CVmodel.add(tf.keras.layers.Dense(CVlayers[i], activation = self.activation))
        self.CVmodel.add(tf.keras.layers.Dense(CVlayers[-1]))
        
        
        
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                                          # learning_rate=0.1,
                                          # beta_1=0.99,
                                          # epsilon=1e-1)
        self.CVoptimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)                                 
        
        self.epochs_T = epochs_T
        self.epochs_CV = epochs_CV
                
        self.C = tf.constant(C)     # maximum conduction velocity.
        self.alpha = tf.constant(alpha)   # regularization coefficient for the conduction velocity.
        self.alphaL2 = tf.constant(alphaL2)      #regularization coefficient for the weights of the neural network.
        
        # self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
        # self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)
        
        
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        
    
    def loss(self):
        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf, self.z_tf)
        self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf, self.z_e_tf)
        
        L1 = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred))
        L2 = tf.reduce_mean(tf.square(self.f_T_e_pred))
        L3 = tf.reduce_mean(tf.square(self.f_T_pred))
        L4 = self.alpha*tf.reduce_mean(tf.square(self.f_CV_e_pred))
        L5 = self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred))
        
        # weights = self.model.trainable_variables
        # L6 = sum([self.alphaL2* tf.nn.l2_loss(w)for w in weights])
        T1_loss = L1
        T2_loss = L2 +L3
        # T3_loss = L3
        CV_loss = (L4 + L5)
        total_loss = T1_loss + T2_loss + CV_loss # + T3_loss+ CV_loss        
        return T1_loss, T2_loss, total_loss #, L6 CV_loss, T3_loss, 
    
      
    def net_eikonal(self, x, y, z):  #Alle afgeleides berekenen
        
        with tf.GradientTape(persistent=True) as tape:
            C = self.C              # maximum conduction velocity.
            T = self.model(tf.concat([x,y,z], 1))         #predicted activation time
            CV = self.CVmodel(tf.concat([x,y,z], 1))       #predicted velocities
            CV = C*tf.sigmoid(CV)
            
        T_x = tape.gradient(T, x)         #afgeleide van T in x-dir
        T_y = tape.gradient(T, y)         #afgeleide van T in y-dir
        T_z = tape.gradient(T, z) 
    
        CV_x = tape.gradient(CV, x)       #afgeleiden van V in x-dir
        CV_y = tape.gradient(CV, y)       #afgeleiden van V in y-dir
        CV_z = tape.gradient(CV, z)
        del tape

        f_T = tf.sqrt(T_x**2 + T_y**2 + T_z**2) - 1.0/CV  #Deze functies minimaliseren in de loss function
        f_CV = tf.sqrt(CV_x**2 + CV_y**2 + CV_z**2)        
        
        return T, CV, f_T, f_CV
    
            

    def trainT(self):
        T1_losses = []
        T2_losses = []
        total_losses = []
        for i in range(self.epochs_T):
            
            variables1 = self.model.trainable_variables
            with tf.GradientTape(persistent=True) as tape:
                 tape.watch(variables1)
                 T1_loss, T2_loss, total_loss = self.loss()  
            grads1 = tape.gradient(T1_loss,variables1)
            
            
            del tape
            self.optimizer.apply_gradients(zip(grads1, variables1))
           
            T1_losses.append(T1_loss.numpy())
            T2_losses.append(T2_loss.numpy())
            # T3_losses.append(T3_loss.numpy())
            # CV_losses.append(CV_loss.numpy())
            total_losses.append(total_loss.numpy())
            
            print('Run ',i)
            
            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % 10 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                print("loss {:1.2f}".format(total_loss.numpy()))
                
        return T1_losses, T2_losses, total_losses #CV_losses, 

    def trainCV(self):
        T1_losses = []
        T2_losses = []
        total_losses = []
        
        for i in range(self.epochs_CV):
            
            variables1 = self.model.trainable_variables
            variables2 = self.CVmodel.trainable_variables
            with tf.GradientTape(persistent=True) as tape:
                 tape.watch(variables1)
                 tape.watch(variables2)
                 T1_loss, T2_loss, total_loss = self.loss()  
            grads1 = tape.gradient(total_loss,variables1)
            grads2 = tape.gradient(total_loss,variables2)
            
            
            del tape
            self.CVoptimizer.apply_gradients(zip(grads1, variables1))
            self.CVoptimizer.apply_gradients(zip(grads2, variables2))
           
            T1_losses.append(T1_loss.numpy())
            T2_losses.append(T2_loss.numpy())
            # T3_losses.append(T3_loss.numpy())
            # CV_losses.append(CV_loss.numpy())
            total_losses.append(total_loss.numpy())
            
            print('Run ',i)
            
            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % 10 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                print("loss {:1.2f}".format(total_loss.numpy()))
                
        return T1_losses, T2_losses, total_losses #CV_losses, 

 

    
    def predict(self, x_star, y_star, z_star):
        T_star = self.model(tf.concat([x_star,y_star, z_star], 1))        
        #T_star = T_star.eval(session=tf.compat.v1.Session())
        T_star = T_star.numpy()
        CV_star = self.C*tf.sigmoid(self.CVmodel(tf.concat([x_star,y_star, z_star], 1)))        
        #CV_star = CV_star.eval(session=tf.compat.v1.Session())
        CV_star = CV_star.numpy()
        return T_star, CV_star
 