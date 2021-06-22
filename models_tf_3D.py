# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:34:18 2021

@author: elske
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import time
from pyDOE import lhs

#%%
tf.random.set_random_seed(1234)
np.random.seed(1234)

#%%
class Eikonal3DnetCV2(object):
    # Initialize the class
    def __init__(self, x, y, z,x_e, y_e, z_e, T_e, layers, CVlayers, C = 1.0, alpha = 1e-5, alphaL2 = 1e-6, jobs = 4):
        
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
        
        self.layers = layers
        self.CVlayers = CVlayers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  
        self.CVweights, self.CVbiases = self.initialize_NN(CVlayers)  
        
        self.C = tf.constant(C)     # maximum conduction velocity.
        self.alpha = tf.constant(alpha)   # regularization coefficient for the conduction velocity.
        self.alphaL2 = alphaL2      #regularization coefficient for the weights of the neural network.
        

        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True,
                                                     intra_op_parallelism_threads=jobs,
                                                     inter_op_parallelism_threads=jobs,
                                                     device_count={'CPU': jobs}))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        
        self.T_e_tf = tf.placeholder(tf.float32, shape=[None, self.T_e.shape[1]]) 
        self.x_e_tf = tf.placeholder(tf.float32, shape=[None, self.x_e.shape[1]]) 
        self.y_e_tf = tf.placeholder(tf.float32, shape=[None, self.y_e.shape[1]]) 
        self.z_e_tf = tf.placeholder(tf.float32, shape=[None, self.z_e.shape[1]]) 
        

        self.T_pred, self.CV_pred, self.f_T_pred, self.f_CV_pred = self.net_eikonal(self.x_tf, self.y_tf, self.z_tf)
                
        self.T_e_pred, self.CV_e_pred, self.f_T_e_pred, self.f_CV_e_pred = self.net_eikonal(self.x_e_tf, self.y_e_tf, self.z_tf)
                        
        self.loss = tf.reduce_mean(tf.square(self.T_e_tf - self.T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_e_pred)) + \
                    tf.reduce_mean(tf.square(self.f_T_pred)) + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_e_pred)) + \
                    self.alpha*tf.reduce_mean(tf.square(self.f_CV_pred))  + \
                    sum([self.alphaL2*tf.nn.l2_loss(w) for w in self.weights])
                    

                    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
        # Define optimizer (use L-BFGS for better accuracy)       
        #self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
        #                                                        method = 'L-BFGS-B', 
        #                                                        options = {'maxiter': 10000,
        #                                                                   'maxfun': 50000,
        #                                                                   'maxcor': 50,
        #                                                                   'maxls': 50,
        #                                                                   'ftol' : 1.0 * np.finfo(float).eps})
        
        self.optimizer = tf.train.AdamOptimizer()
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
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
        #for l in range(0,num_layers-1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)      #Waarom bij de laatste laag geen activatie functie?
        return Y
    
    def net_eikonal(self, x, y,z):
        C = self.C              # maximum conduction velocity.
        T = self.neural_net(tf.concat([x,y,z], 1), self.weights, self.biases)
        #T = self.neural_net(tf.concat([x,y,z], 1), self.weights, self.biases)         #predicted activation time
        CV = self.neural_net(tf.concat([x,y,z], 1), self.CVweights, self.CVbiases)    #predicted velocities
        CV = C*tf.sigmoid(CV)
        
        T_x = tf.gradients(T, x)[0]         #afgeleide van T in x-dir
        T_y = tf.gradients(T, y)[0] 
        T_z = tf.gradients(T, z)[0]         #afgeleide van T in y-dir
        
        CV_x = tf.gradients(CV, x)[0]       #afgeleiden van V in x-dir
        CV_y = tf.gradients(CV, y)[0]
        CV_z = tf.gradients(CV, z)[0]       #afgeleiden van V in y-dir
        
        f_T = tf.sqrt(T_x**2 + T_y**2 +T_z**2) - 1.0/CV  #Deze functies minimaliseren in de loss function
        f_CV = tf.sqrt(CV_x**2 + CV_y**2 + CV_z**2)        
        
        return T, CV, f_T, f_CV
    
    def callback(self, loss):               
        self.lossit.append(loss)     #Wat is lossit
        print('Loss: %.5e' % (loss))
        
    def train(self):                #Dit snap ik nog niet

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z, 
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.z_e_tf: self.z_e, 
                   self.T_e_tf: self.T_e}
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
        
    def train_Adam(self, nIter): 
        
        self.lossit = []

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.z_e_tf: self.z_e,
                   self.T_e_tf: self.T_e}        
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.lossit.append(loss_value)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = np.exp(self.sess.run(self.C))
                print('It: %d, Loss: %.3e, C: %.3f, Time: %.2f' % 
                      (it, loss_value, C_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
    
    def train_Adam_minibatch(self, nIter, size = 50): 
        self.lossit = []
       
        start_time = time.time()
        for it in range(nIter):
            X = lhs(2, size)
            tf_dict = {self.x_tf: X[:,0], self.y_tf: X[:,1], self.z_tf: X[:,2],
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.z_e_tf: self.z_e, self.T_e_tf: self.T_e} 
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            self.lossit.append(loss_value)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                C_value = np.exp(self.sess.run(self.C))
                print('It: %d, Loss: %.3e, C: %.3f, Time: %.2f' % 
                      (it, loss_value, C_value, elapsed))
                start_time = time.time()
                
    
    def predict(self, x_star, y_star, z_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star,
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.z_e_tf: self.z_e}
        
        T_star = self.sess.run(self.T_pred, tf_dict)
        CV_star = self.sess.run(self.CV_pred, tf_dict)

        
        return T_star, CV_star
    
    def get_adaptive_points(self, N = 1000, M = 10):
        
        X = lhs(2, N)
        tf_dict = {self.x_tf: X[:,0], self.y_tf: X[:,1], self.z_tf: X[:,2],
                   self.x_e_tf: self.x_e, self.y_e_tf: self.y_e, self.z_e_tf: self.z_e}
        
        f_T_star = self.sess.run(self.f_T_pred, tf_dict)
        
        ind = f_T_star[:,0].argsort()[-M:]
        
        return X[ind], f_T_star[ind]