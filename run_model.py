# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:02:45 2016

@author: rahulkumar
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import corpus_handel 
import os, json


##Hyper parameters
SCALE_NUM_TRIPS = 100000
numiter = 100000
modelfile = os.getcwd() +'/trained_model'
npredictors = 7
noutputs = 1 
nhidden = 3


def model(requirement= [ ]):
    
    jsondata = { 
               'Prediction' : 0.0
               }
               

    data = pd.read_excel('Raw_data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
    x, v, _ = corpus_handel.load_data(data)
    

    
    requirement = requirement[0].strip()
    requirement = corpus_handel.clean_str(requirement)
    requirement = requirement.split(" ")
    
    num_padd = x[0].shape[1] - len(requirement)
    requirement = requirement + ["<PAD/>"] * num_padd
    
    
    for word in requirement:
        if not v.has_key(word):
            requirement[requirement.index(word)] = "<PAD/>"
    
#    print 'Processed req=>', requirement
    x = np.array([v[word] for word in requirement])
    
    input = pd.DataFrame(np.array([x]))
    
        
    with tf.Session() as sess:
        filename = modelfile + '-' + str(numiter)
        feature_data = tf.placeholder("float", [None, npredictors])
      
        weights1 = tf.Variable(tf.truncated_normal([npredictors, nhidden], stddev=0.01))
        weights2 = tf.Variable(tf.truncated_normal([nhidden, noutputs], stddev=0.01))
      
        biases1 = tf.Variable(tf.ones([nhidden]))
        biases2 = tf.Variable(tf.ones([noutputs]))
        
        saver = tf.train.Saver({'weights1' : weights1, 'biases1' : biases1, 'weights2' : weights2, 'biases2' : biases2})
    
        saver.restore(sess, filename)
    
        feature_data = tf.placeholder("float", [None, npredictors])
        predict_operation = (tf.matmul(tf.nn.relu(tf.matmul(feature_data, weights1) + biases1), weights2) + biases2) * SCALE_NUM_TRIPS
        predicted = sess.run(predict_operation, feed_dict = {
            feature_data : input.values
          })
    
    jsondata = { 'Prediction' : predicted[0]}
    
    jsonreturn = json.dumps(jsondata)                    
    return jsonreturn

#print model(requirement=['MEERUT SDC','OKHLA SDC','2'])