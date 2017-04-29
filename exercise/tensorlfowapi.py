
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.framework import ops


def test_scatter_update():
    a = tf.Variable(initial_value=[2, 5, -4, 0])
    b = tf.scatter_update(a, [2,2], [9,100])
    return b

def test_scatter_nd():
    indices = tf.constant([[3]])
    updates = tf.constant([[9,10,11,12]])
    shape = tf.constant([8,4])
    scatter = tf.scatter_nd(indices, updates, shape)

    return scatter

def test_scatter_nd_2():
    gt_bboxes = tf.constant([[0,0,1,2],[1,0,3,4],[100,100,105,102.5]])
    gt_labels = tf.constant([1,2,6])
  
    
    gt_anchors_labels = tf.Variable([100,100,100,100], trainable=False,collections=[ops.GraphKeys.LOCAL_VARIABLES])
    gt_anchors_bboxes=tf.Variable([[100,100,105,105],[2,1,3,3.5],[0,0,10,10],[0.5,0.5,0.8,1.5]], trainable=False,collections=[ops.GraphKeys.LOCAL_VARIABLES],dtype=tf.float32)
    
    max_inds = [1,0,3]
    
    gt_anchors_labels = tf.scatter_update(gt_anchors_labels, max_inds,gt_labels)
    gt_anchors_bboxes = tf.scatter_update(gt_anchors_bboxes, max_inds,gt_bboxes)
    
    
    
    
   
    return gt_anchors_labels,gt_anchors_bboxes

def test_scatter_nd_3():
    gt_bboxes = tf.constant([[0,0,1,2],[1,0,3,4],[100,100,105,102.5]])
    gt_labels = tf.constant([1,2,6])
  
    jaccard = tf.constant( [[ 0. ,     0.  ,    0.02,    0.15  ],[ 0. ,     0.3125 , 0.08,    0.    ],[ 0.5 ,    0. ,     0.  ,    0.    ]])
    gt_anchors_scores = tf.constant([0.0,0.,0.,0.])
    gt_anchors_labels = tf.constant([100,100,100,100])
    gt_anchors_bboxes=tf.constant([[100,100,105,105],[2,1,3,3.5],[0,0,10,10],[0.5,0.5,0.8,1.5]])
    
    max_inds = tf.cast(tf.argmax(jaccard, axis=1),tf.int32)
    
    def cond(i,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores):
        r = tf.less(i, tf.shape(gt_labels)[0])
        return r
    def body(i,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores):
        
        #upate gt_anchors_labels
        updates = tf.reshape(gt_labels[i], [-1])
        indices = tf.reshape(max_inds[i],[1,-1])
        shape = tf.reshape(tf.shape(gt_anchors_bboxes)[0],[-1])
        
        
        new_labels = tf.scatter_nd(indices, updates, shape)
        new_mask = tf.cast(new_labels, tf.bool)
        gt_anchors_labels = tf.where(new_mask, new_labels, gt_anchors_labels)
        
        #update gt_anchors_bboxes
        updates = tf.reshape(gt_bboxes[i], [1,-1])
        indices = tf.reshape(max_inds[i],[1,-1])
        shape = tf.shape(gt_anchors_bboxes)
        new_bboxes = tf.scatter_nd(indices, updates, shape)
        gt_anchors_bboxes = tf.where(new_mask, new_bboxes, gt_anchors_bboxes)
        
        #update gt_anchors_scores
        updates = tf.reshape(jaccard[i, max_inds[i]], [-1])
        indices = tf.reshape(max_inds[i],[1,-1])
        shape = tf.reshape(tf.shape(gt_anchors_bboxes)[0],[-1])
        new_scores = tf.scatter_nd(indices, updates, shape)
        gt_anchors_scores = tf.where(new_mask, new_scores, gt_anchors_scores)
        

        
        return [i+1,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores]
    
    
    i = 0
    [i,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores] = tf.while_loop(cond, body,[i,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores])
    
    
    
    
   
    return gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores


with tf.Graph().as_default():

#     i = tf.constant(0)
#     c = lambda i: tf.less(i, 10)
#     b = lambda i: tf.add(i, 1)
#     r = tf.while_loop(c, b, [i])
    
#     jaccard= tf.constant([0])
#     glabels = tf.constant([32],dtype=tf.int32)
#     rlabel = tf.constant([16])
#     jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)
#     jaccard = tf.Print(jaccard, [jaccard,glabels, rlabel], "in body")
    

    
    # Best fit, checking it's above threshold.
#     idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)

#     ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
    
#     my_var = tf.zeros(10,dtype=tf.float32)
#     values= tf.constant([32,5],dtype=tf.float32)
#     indices = tf.constant([2,9], dtype=tf.int64)
#     
#     delta = tf.SparseTensor(indices, values, [10])
#     
#     result = my_var + tf.sparse_tensor_to_dense(delta)
    
#     my_var = tf.scatter_update(my_var,inds,valuses)

#     indices = tf.constant([[3], [3], [1], [7]])
#     updates = tf.constant([9, 10, 11, 12])
#     shape = tf.constant([8])
#     
#     delta = tf.SparseTensor(indices, updates, shape)
#     scatter = tf.sparse_tensor_to_dense(delta)
    
    
#     scatter = tf.scatter_nd(indices, updates, shape)


#     indices = [[5], [3], [1], [7]] # A list of coordinates to update.
# 
#     values = [9, 10, 11, 12]  # A list of values corresponding to the respective
#                     # coordinate in indices.
#     
#     shape = [8]  # The shape of the corresponding dense tensor, same as `c`.
#     
#     delta = tf.SparseTensor(indices, values, shape)
#     scatter = tf.sparse_tensor_to_dense(delta)

   
    
    


    
#     res = test_scatter_nd()
    res = test_scatter_nd_3()
#     res = test_scatter_update()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
#         res = sess.run(res)
        
        gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores = sess.run(res)
        print(gt_anchors_labels)
        
        
        
        
        
        
        