# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:02:27 2021

@author: elske
"""
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import numpy as np
import time
from pyDOE import lhs

#%%
tf.random.set_seed(1234)
np.random.seed(1234)

#%%
class Eikonal2DnetCV2(object):
    # Initialize the class
    def __init__(self, x, y, x_e, y_e, T_e, layers, CVlayers, C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, jobs = 4):
        
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
        
        self.layers = layers
        self.CVlayers = CVlayers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)  
        
        self.C = tf.constant(C)     # maximum conduction velocity.
        self.alpha = tf.constant(alpha)   # regularization coefficient for the conduction velocity.
        self.alphaL2 = alphaL2      #regularization coefficient for the weights of the neural network.
        

        
        
        self.x_tf = tf.Variable(self.x, dtype=tf.float32)
        self.y_tf = tf.Variable(self.y, dtype=tf.float32)
        
        self.T_e_tf = tf.Variable(self.T_e, dtype=tf.float32)
        self.x_e_tf = tf.Variable(self.x_e, dtype=tf.float32)
        self.y_e_tf = tf.Variable(self.y_e, dtype=tf.float32)
        

        #self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf)
        #self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf)
                        
        
                    

                    
        self.optimizer_Adam = tf.keras.optimizers.Adam(
                                          learning_rate=0.1,
                                          beta_1=0.99,
                                          epsilon=1e-1)
      
        
        
        
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)      #Waarom bij de laatste laag geen activatie functie?
        return Y
    
    def loss(self):
        
        with tf.GradientTape(persistent=True) as tape:
            C = self.C              # maximum conduction velocity.
            
            T = self.neural_net(tf.concat([self.x_tf, self.y_tf], 1), self.weights, self.biases)         #predicted activation time
            CV = self.neural_net(tf.concat([self.x_tf, self.y_tf], 1), self.CVweights, self.CVbiases)    #predicted velocities
            CV = C*tf.sigmoid(CV)
            
            T_e = self.neural_net(tf.concat([self.x_e_tf, self.y_e_tf], 1), self.weights, self.biases)         #predicted activation time
            CV_e = self.neural_net(tf.concat([self.x_e_tf, self.y_e_tf], 1), self.CVweights, self.CVbiases)    #predicted velocities
            CV_e = C*tf.sigmoid(CV_e)
        
        T_x = tape.gradient(T, self.x_tf)         #afgeleide van T in x-dir
        T_y = tape.gradient(T, self.y_tf)         #afgeleide van T in y-dir
        
        T_e_x = tape.gradient(T_e, self.x_e_tf)
        T_e_y = tape.gradient(T_e, self.y_e_tf)
        
        CV_x = tape.gradient(CV, self.x_tf)       #afgeleiden van V in x-dir
        CV_y = tape.gradient(CV, self.y_tf)       #afgeleiden van V in y-dir
        
        CV_e_x = tape.gradient(CV_e, self.x_e_tf)
        CV_e_y = tape.gradient(CV_e, self.y_e_tf)
        del tape
        
        f_T = tf.sqrt(T_x**2 + T_y**2) - 1.0/CV  #Deze functies minimaliseren in de loss function
        f_CV = tf.sqrt(CV_x**2 + CV_y**2)   
        
        f_T_e = tf.sqrt(T_e_x**2 + T_e_y**2) - 1.0/CV_e  #Deze functies minimaliseren in de loss function
        f_CV_e = tf.sqrt(CV_e_x**2 + CV_e_y**2)
        
        #print(T_x, T_y, T_e_x, T_e_y, f_T, f_CV, f_T_e, f_CV_e)
        
        loss = tf.reduce_mean(tf.square(self.T_e_tf - T_e)) + \
                    tf.reduce_mean(tf.square(f_T_e)) + \
                    tf.reduce_mean(tf.square(f_T)) + \
                    self.alpha*tf.reduce_mean(tf.square(f_CV_e)) + \
                    self.alpha*tf.reduce_mean(tf.square(f_CV))  + \
                    sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights])
        
        return loss 
    
    def callback(self, loss):               
        self.lossit.append(loss)     #Wat is lossit
        print('Loss: %.5e' % (loss))
        
  
    def train_Adam(self, nIter): 
        
        self.lossit = []
  
        start_time = time.time()
        for it in range(nIter):
            with tf.GradientTape(persistent=True) as tape:
                loss_value = self.loss()
            grads = tape.gradient(loss_value, self.weights)
            #CVgrads = tape.gradient(loss_value, self.CVweights)
            
            del tape
            
            self.optimizer_Adam.apply_gradients(zip(grads, self.weights))
            self.optimizer_Adam.apply_gradients(zip(CVgrads, self.CVweights))
           
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = tf.exp(self.C)
                print('It: %d, Loss: %.3e, C: %.3f, Time: %.2f' % 
                      (it, loss_value, C_value, elapsed))
                start_time = time.time()
            
        
    
    
    def predict(self, x_star, y_star):
        x_star_tf = tf.Variable(x_star, dtype=tf.float32)
        y_star_tf = tf.Variable(y_star, dtype=tf.float32)
        T_star = self.neural_net(tf.concat([x_star_tf, y_star_tf], 1), self.weights, self.biases)
        CV_star = self.neural_net(tf.concat([x_star_tf, y_star_tf], 1), self.CVweights, self.CVbiases)

        T_star = T_star.numpy()
        CV_star = CV_star.numpy()
        return T_star, CV_star
