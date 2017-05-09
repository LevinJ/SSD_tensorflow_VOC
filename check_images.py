from datasets import pascalvoc_datasets
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
# from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import cv2
from utility import visualization
from nets.ssd import g_ssd_model
from preprocessing.ssd_vgg_preprocessing import np_image_unwhitened
from preprocessing.ssd_vgg_preprocessing import preprocess_for_train
from preprocessing.ssd_vgg_preprocessing import preprocess_for_eval
import tf_utils
import math
from preparedata import PrepareData


class CheckImages(PrepareData):
    def __init__(self):
        
        PrepareData.__init__(self)
        return
    
    def run(self):
        
        
        with tf.Graph().as_default():
#             batch_data= self.get_voc_2007_train_data(is_training_data=False)
            batch_data = self.get_voc_2007_test_data()
#             batch_data = self.get_voc_2012_train_data()
#             batch_data = self.get_voc_2007_2012_train_data(is_training_data = True)


#             return self.iterate_file_name(batch_data)
           
            num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
            target_filenames = []
            target_object = 9
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
                    i = 0
                    while i < num_batches:  
                         
                        image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores = sess.run(list(batch_data))
                        pos_sample_inds = (glabels == target_object).nonzero()
                        target_filenames.extend(list(filename[np.unique(pos_sample_inds[0])]))
#                         print(filename)
#                         print(glabels)
                        
                         
                       
                        i += 1
                    print(target_filenames)
                            
                        
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= CheckImages()
    obj.run()