
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.framework import ops




sess=tf.InteractiveSession()
x = tf.placeholder(tf.float64) 

sess=tf.InteractiveSession()
initial = tf.truncated_normal([1,5], stddev=0.1,dtype=tf.float64)  
y = tf.Variable(initial) 

keep_prob = tf.placeholder(tf.float64) 
dx = tf.nn.dropout(x*y, keep_prob)
sess.run(tf.initialize_all_variables())
res = sess.run(dx, feed_dict={x : np.array([1.0, 2.0, 3.0, 4.0, 5.0]),keep_prob: 0.5})
sess.close()
        
        
        
        
        
        
        