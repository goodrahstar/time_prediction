# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:02:45 2016

@author: rahulkumar
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import corpus_handel 



data = pd.read_excel('Raw_data.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
print 'data loded'

#data['producttype'] = data['producttype'].map({'T1':1.0,'T2':2.0,'T3':3.0,'T4':4.0,'T148':148.0,'T172':172.0,'T196':196.0,'T21400':21400.0,'T21100':21100.0})

data = data.fillna(0.0)
data = data[data.destinationbranch != 0.0]
data = data.reset_index(drop=True)

#data = data.drop(['Index','details','Delivered date', 'Delivered time', 'Dispatch date','Dispatch time'],1)

x, v, v_in = corpus_handel.load_data(data)
print 'data encoded'

prod=pd.DataFrame(data=np.array(x[0]))
size = len(prod.columns)

result = pd.concat([prod, data.iloc[0:len(x[0]),2:len(data.columns)]], axis=1)

print 'frame prepared of size = ', size


predictors = result.iloc[:,0:size+1]
targets = result.iloc[:,-1]
del data, v, v_in,x,prod 


print ' shuffles and input data ready'


SCALE_NUM_TRIPS = 100000
trainsize = int(len(result['producttype']) * 0.8)
testsize = len(result['producttype']) - trainsize
npredictors = len(predictors.columns)
noutputs = 1 #number of classes 
nhidden = 5
numiter = 10000
modelfile = '/tmp/trained_model'

with tf.Session() as sess:
  feature_data = tf.placeholder("float", [None, npredictors])
  target_data = tf.placeholder("float", [None, noutputs])
  
  weights1 = tf.Variable(tf.truncated_normal([npredictors, nhidden], stddev=0.01), name='weight1')
  weights2 = tf.Variable(tf.truncated_normal([nhidden, noutputs], stddev=0.01),name='weight2')
  
  biases1 = tf.Variable(tf.ones([nhidden]))
  biases2 = tf.Variable(tf.ones([noutputs]))
  
  model = (tf.matmul(tf.nn.relu(tf.matmul(feature_data, weights1) + biases1), weights2, name= 'output') + biases2) * SCALE_NUM_TRIPS

  cost = tf.nn.l2_loss(model - target_data, name='loss')

  training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
  
#  summaries = tf.merge_all_summaries()
#  summary_writer = tf.train.SummaryWriter('log_simple_stats', sess.graph)

  init = tf.initialize_all_variables()
  sess.run(init)

  saver = tf.train.Saver({'weights1' : weights1, 'biases1' : biases1, 'weights2' : weights2, 'biases2' : biases2})
  for iter in xrange(0, numiter):
    sess.run(training_step, feed_dict = {
        feature_data : predictors[:trainsize].values,
        target_data : targets[:trainsize].values.reshape(trainsize, noutputs)
      })
#    summary_writer.add_summary(sess.run(summaries), iter)
    
    if iter%1000 == 0:
      print '{0} error={1}'.format(iter, np.sqrt(cost.eval(feed_dict = {
          feature_data : predictors[:trainsize].values,
          target_data : targets[:trainsize].values.reshape(trainsize, noutputs)
      }) / trainsize))
    
  filename = saver.save(sess, modelfile, global_step=numiter)
  print 'Model written to {0}'.format(filename)

  print 'testerror={0}'.format(np.sqrt(cost.eval(feed_dict = {
          feature_data : predictors[trainsize:].values,
          target_data : targets[trainsize:].values.reshape(testsize, noutputs)
      }) / testsize))
      
      
      
#
#
#input = predictors
#
##input = pd.DataFrame.from_dict(data = 
##                               {'Document Type' : [8],
##                                'Amount in doc. curr.' : [-55900.0],
##                                'Amount in local currency' : [-12049938.72],
##                                })
#with tf.Session() as sess:
#    filename = modelfile + '-' + str(numiter)
#    saver = tf.train.Saver({'weights1' : weights1, 'biases1' : biases1, 'weights2' : weights2, 'biases2' : biases2})
#    saver.restore(sess, filename)
#    feature_data = tf.placeholder("float", [None, npredictors])
#    predict_operation = (tf.matmul(tf.nn.relu(tf.matmul(feature_data, weights1) + biases1), weights2) + biases2) * SCALE_NUM_TRIPS
#    predicted = sess.run(predict_operation, feed_dict = {
#        feature_data : input.values
#      })
#
#print predicted