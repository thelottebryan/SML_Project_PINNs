# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:53:15 2021

@author: elske
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
#%%
   
class my_Eikonal2D():
        # Initialize the class
    def __init__(self, x, y, x_e, y_e, T_e, layers, CVlayers, C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, epochs=1000):
        
        X = np.concatenate([x, y], 1)
        #X_e = np.concatenate([x_e, t_e], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        #self.X_e = X_e
        
        self.x = x  #location of the collocation points to enforce the residual penalty (random samples). Each of them must be of shape N_colloc, 1
        self.y = y
        
        self.T_e = T_e  #activation times data. Must be of shape N_data, 1
        self.x_e = x_e  #location of the data points (exact values). Each of them must be of shape N_data, 1
        self.y_e = y_e
        
        self.x_tf = tf.Variable(self.x, dtype=tf.float32)
        self.y_tf = tf.Variable(self.y, dtype=tf.float32)
        
        self.T_e_tf = tf.Variable(self.T_e, dtype=tf.float32)
        self.x_e_tf = tf.Variable(self.x_e, dtype=tf.float32)
        self.y_e_tf = tf.Variable(self.y_e, dtype=tf.float32)
        
        #model
        branch_width       = layers[1:-1]
        branch_width_CV    = CVlayers[1:-1]
        activ_function = tf.nn.tanh
       
        inputs    = tf.keras.layers.Input(shape = 2)

        v1_branch = tf.keras.layers.Dense(branch_width[0], activation = activ_function,kernel_initializer = 'glorot_normal')(inputs)

        for width in branch_width[1:]:

            v1_branch = tf.keras.layers.Dense(
                width, activation = activ_function,
                kernel_initializer = 'glorot_normal')(v1_branch)

    
        v1_branch = tf.keras.layers.Dense(
            1, activation = activ_function,
            kernel_initializer = 'glorot_normal')(v1_branch)

        v2_branch = tf.keras.layers.Dense(
                branch_width_CV[0], activation = activ_function,
                kernel_initializer = 'glorot_normal')(inputs)
        
        for width in branch_width_CV[1:]:
        
            v2_branch = tf.keras.layers.Dense(
                width, activation = activ_function,
                kernel_initializer = 'glorot_normal')(v2_branch)
        
            
        v2_branch = tf.keras.layers.Dense(
            1, activation = activ_function,
            kernel_initializer = 'glorot_normal')(v2_branch)

        self.model = tf.keras.models.Model(inputs = inputs, outputs = [v1_branch, v2_branch], name = 'Eikonal')
        
        
        self.optimizer = tf.keras.optimizers.Adam(
                                          learning_rate=0.1,
                                          beta_1=0.99,
                                          epsilon=1e-1)
        
        self.epochs=epochs
                
        self.C = tf.constant(C)     # maximum conduction velocity.
        self.alpha = tf.constant(alpha)   # regularization coefficient for the conduction velocity.
        self.alphaL2 = alphaL2      #regularization coefficient for the weights of the neural network.
        
        #self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
        #self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)
        self.weights = self.model.get_weights()
        
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        
        
    def loss(self):
        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
        self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)
        
        L1 = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred))
        L2 = tf.reduce_mean(tf.square(self.f_T_e_pred))
        L3 = tf.reduce_mean(tf.square(self.f_T_pred))
        L4 = self.alpha*tf.reduce_mean(tf.square(self.f_CV_e_pred))        
        L5 = self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred))          
        
        #L6 = sum([self.alphaL2* tf.nn.l2_loss(w)for w in self.weights])
        return L1 + L2 + L3 + L4 + L5 #+ L6 
    
    def neural_net(self, x):
        return self.model(x)
 
    def net_eikonal(self, x, y):  #Alle afgeleides berekenen
        
        with tf.GradientTape(persistent=True) as tape:
            C = self.C              # maximum conduction velocity.
            T, CV = self.neural_net(tf.concat([x,y], 1))         #predicted activation time
            CV = C*tf.sigmoid(CV)
            
        T_x = tape.gradient(T, x)         #afgeleide van T in x-dir
        T_y = tape.gradient(T, y)         #afgeleide van T in y-dir
    
        CV_x = tape.gradient(CV, x)       #afgeleiden van V in x-dir
        CV_y = tape.gradient(CV, y)       #afgeleiden van V in y-dir
        del tape

        f_T = tf.sqrt(T_x**2 + T_y**2) - 1.0/CV  #Deze functies minimaliseren in de loss function
        f_CV = tf.sqrt(CV_x**2 + CV_y**2)        
        
        return T, CV, f_T, f_CV
    
    def train(self):
        losses = []
        
        for i in range(self.epochs):
            
            variables = self.model.trainable_variables
            with tf.GradientTape(persistent=True) as tape:
                loss = self.loss()
            grads = tape.gradient(loss,variables)
            
            del tape
            
            self.optimizer.apply_gradients(zip(grads, variables))
            
            self.ckpt.step.assign_add(1)
            if int(self.ckpt.step) % 10 == 0:
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                print("loss {:1.2f}".format(loss.numpy()))
                
        return losses
    
    
    def predict(self, x_star, y_star):
        T_star, CV_star = self.neural_net(tf.concat([x_star,y_star], 1))        
        T_star = T_star.numpy()
        CV_star = CV_star.numpy()
        return T_star, CV_star