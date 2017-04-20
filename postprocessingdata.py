import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import numpy as np
import math
from preparedata import PrepareData
from nets.ssd import g_ssd_model
import tf_extended as tfe
import time
from tensorflow.python.ops import math_ops


class PostProcessingData(object):
    def __init__(self):
       
        
        
        return
    def get_mAP_tf(self,predictions, localisations,glabels, gbboxes,gdifficults):
        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
        
            # Detected objects from SSD output.
            localisations = g_ssd_model.decode_bboxes_all_ayers_tf(localisations)
            
            rscores, rbboxes = g_ssd_model.detected_bboxes(predictions, localisations)
            
            # Compute TP and FP statistics.
            num_gbboxes, tp, fp, rscores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          glabels, gbboxes, gdifficults)
            
            for c in rscores.keys():
            
                #reshape data
                num_gbboxes = math_ops.to_int64(num_gbboxes)
                scores = math_ops.to_float(rscores)
                stype = tf.bool
                tp = tf.cast(tp, stype)
                fp = tf.cast(fp, stype)
                # Reshape TP and FP tensors and clean away 0 class values.(difficult bboxes)
                scores = tf.reshape(scores, [-1])
                tp = tf.reshape(tp, [-1])
                fp = tf.reshape(fp, [-1])
                
                # Remove TP and FP both false.
                mask = tf.logical_or(tp, fp)
        
                rm_threshold = 1e-4
                mask = tf.logical_and(mask, tf.greater(scores, rm_threshold))
                scores = tf.boolean_mask(scores, mask)
                tp = tf.boolean_mask(tp, mask)
                fp = tf.boolean_mask(fp, mask)
                
                num_gbboxes = tf.reduce_sum(num_gbboxes)
                num_detections = tf.size(scores, out_type=tf.int32)
                
                # Precison and recall values.
                prec, rec = tfe.precision_recall(num_gbboxes, num_detections, tp, fp, scores)
            
            
        return
    
    
   
    
    def run(self):
        
        
       
        
        
        return
    
    
g_post_processing_data = PostProcessingData()

if __name__ == "__main__":   
    obj= PostProcessingData()
    obj.run()