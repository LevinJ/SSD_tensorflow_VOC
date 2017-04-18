from datasets import dataset_factory
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



class PrepareData():
    def __init__(self):
        self.dataset_name = None
        self.dataset_split_name = None
        self.dataset_dir = None
        
        self.batch_size = 32
        self.labels_offset = 0
      
        self.num_preprocessing_threads = 4
        self.is_training_data = True
        
        
        return
    def __preprocess_data(self, image, labels, bboxes):
        out_shape = g_ssd_model.img_shape
        if self.is_training_data:
            image, labels, bboxes = preprocess_for_train(image, labels, bboxes, out_shape = out_shape)
        else:
            image, labels, bboxes, _ = preprocess_for_eval(image, labels, bboxes, out_shape = out_shape)
        return image, labels, bboxes
    def __get_images_labels_bboxes(self):
        dataset = dataset_factory.get_dataset(
                self.dataset_name, self.dataset_split_name, self.dataset_dir)
        
        self.dataset = dataset
        
        if self.is_training_data:
            
            shuffle = True
            #make sure most samples can be fetched in one epoch
            self.num_readers = 1
        else:
            #make sure data is fetchd in sequence
            shuffle = False
            self.num_readers = 1
            
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=shuffle,
                    num_readers=self.num_readers,
                    common_queue_capacity=20 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        
        # Get for SSD network: image, labels, bboxes.
        [image, shape, format, filename, glabels, gbboxes,gdifficults] = provider.get(['image', 'shape', 'format','filename',
                                                         'object/label',
                                                         'object/bbox',
                                                         'object/difficult'])
        glabels -= self.labels_offset
        
        
        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = self.__preprocess_data(image, glabels, gbboxes)

        # Assign groundtruth information for all default/anchor boxes
        gclasses, glocalisations, gscores = g_ssd_model.tf_ssd_bboxes_encode(glabels, gbboxes)
        
        
        return self.__batching_data(image, glabels, format, filename, gbboxes, gdifficults, gclasses, glocalisations, gscores)
    def __batching_data(self,image, glabels, format, filename, gbboxes, gdifficults,gclasses, glocalisations, gscores):
        
       
        #Batch the samples
        if self.is_training_data:
            
            dynamic_pad = False
            batch_shape = [1,1] + [len(gclasses), len(glocalisations), len(gscores)]
            tensors = [image, filename, gclasses, glocalisations, gscores]
        else:
            #in the case of evaluatation data, we will want to batch original glabels and gbboxes
            #this information is still useful even if they are padded after dequeuing
            dynamic_pad = True
            batch_shape = [1,1,1,1,1] + [len(gclasses), len(glocalisations), len(gscores)]
            tensors = [image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores]
            
            # to make sure data is fectched in sequence during evaluation
            self.num_preprocessing_threads = 1
            
        #tf.train.batch accepts only list of tensors, this batch shape can used to
        #flatten the list in list, and later on convet it back to list in list.
        batch = tf.train.batch(
                tf_utils.reshape_list(tensors),
                batch_size=self.batch_size,
                num_threads=self.num_preprocessing_threads,
                dynamic_pad=dynamic_pad,
                capacity=5 * self.batch_size)
        
        if self.is_training_data:
            #speed up batch featching during training
            batch = slim.prefetch_queue.prefetch_queue(
                    batch, capacity=2)
            batch = batch.dequeue()
            
        #convert it back to the list in list format which allows us to easily use later on
        batch= tf_utils.reshape_list(batch, batch_shape)
        return batch
    def __disp_image(self, img, shape_data, classes, bboxes):
        scores =np.full(classes.shape, 1.0)
        visualization.plt_bboxes(img, classes, scores, bboxes,title='Ground Truth')
        return
    def __disp_matched_anchors(self,img, target_labels_data, target_localizations_data, target_scores_data):
        found_matched = False
        all_anchors = g_ssd_model.get_all_anchors()
        for i, target_score_data in enumerate(target_scores_data):

            num_pos = (target_score_data > 0.5).sum()
            if (num_pos == 0):
                continue
            print('Found  {} matched default boxes in layer {}'.format(num_pos,g_ssd_model.feat_layers[i]))
            pos_sample_inds = (target_score_data > 0.5).nonzero()
            pos_sample_inds = [pos_sample_inds[0],pos_sample_inds[1],pos_sample_inds[2]]

            classes = target_labels_data[i][pos_sample_inds]
            scores = target_scores_data[i][pos_sample_inds]
            bboxes_default= g_ssd_model.get_all_anchors(minmaxformat=True)[i][pos_sample_inds]
            
            
            
            bboxes_gt = g_ssd_model.decode_bboxes_layer(target_localizations_data[i][pos_sample_inds], 
                                       all_anchors[i][pos_sample_inds])
            
            print("default box minimum, {} gt box minimum, {}".format(bboxes_default.min(), bboxes_gt.min()))
            
            marks_default = np.full(classes.shape, True)
            marks_gt = np.full(classes.shape, False)
            scores_gt = np.full(scores.shape, 1.0)
            
            bboxes = bboxes_default
            neg_marks = marks_default
            add_gt = True
            if add_gt :
                bboxes = np.vstack([bboxes_default,bboxes_gt])
                neg_marks = np.hstack([marks_default,marks_gt])
                classes = np.tile(classes, 2)
                scores = np.hstack([scores, scores_gt])
            
            title = "Default boxes: Layer {}".format(g_ssd_model.feat_layers[i])
            visualization.plt_bboxes(img, classes, scores, bboxes,neg_marks=neg_marks,title=title)
            found_matched = True  
            
        return found_matched
    def get_voc_2007_train_data(self):
        self.dataset_name = 'pascalvoc_2007'
        self.dataset_split_name = 'train'
        self.dataset_dir = '../data/voc/tfrecords/'
        
        return self.__get_images_labels_bboxes()
    
    def get_voc_2012_train_data(self):
        self.dataset_name = 'pascalvoc_2012'
        self.dataset_split_name = 'train'
        self.dataset_dir = '../data/voc/tfrecords/'
        
        return self.__get_images_labels_bboxes()
    def get_voc_2007_test_data(self):
        self.dataset_name = 'pascalvoc_2007'
        self.dataset_split_name = 'test'
        self.dataset_dir = '../data/voc/tfrecords/'
        self.is_training_data = False
        
        return self.__get_images_labels_bboxes()
    
        
    def iterate_file_name(self):
        with tf.Graph().as_default():
            tensors_to_run = self.get_voc_2012_train_data()
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):
                    for i in range(100):
                        #test evaluation
#                         image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores = sess.run(list(tensors_to_run))
                        #test training
                        image, filename, gclasses, glocalisations, gscores = sess.run(list(tensors_to_run))
                        print(filename)
        return
    def run(self):
#         return self.iterate_file_name()
        
        with tf.Graph().as_default():
            batch_voc_2007_train = self.get_voc_2007_train_data()
            batch_voc_2007_test = self.get_voc_2007_test_data()
            batch_voc_2012_train = self.get_voc_2012_train_data()
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):  
                    while True:
                        for current_data in [batch_voc_2012_train]:
                             
                            if len(current_data) == 8:
                                #for evalusyion data,we take a bit more data for each batch
                                image, filename,glabels,gbboxes,gdifficults,gclasses, glocalisations, gscores = sess.run(list(current_data))
                            else:
                                image, filename, gclasses, glocalisations, gscores = sess.run(list(current_data))
                            print(filename)
                             
                             
                            #selet the first image in the batch
                            target_labels_data = [item[0] for item in gclasses]
                            target_localizations_data = [item[0] for item in glocalisations]
                            target_scores_data = [item[0] for item in gscores]
                            image_data = image[0]
     
                            image_data = np_image_unwhitened(image_data)
#                             self.__disp_image(image_data, shape_data, glabels_data, gbboxes_data)
                            found_matched = self.__disp_matched_anchors(image_data,target_labels_data, target_localizations_data, target_scores_data)
                            plt.show()
                        #exit the batch data testing right after a successful match have been found
                        if found_matched:
                                break
                            
                        
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()