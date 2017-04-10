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
        
        self.num_readers = 4
        self.batch_size = 32
        self.labels_offset = 0
        
        self.model_name = None #'The name of the architecture to train.'
        self.weight_decay = 0.00004 # 'The weight decay on the model weights.'
        
        self.preprocessing_name = None
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
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=self.is_training_data,
                    num_readers=self.num_readers,
                    common_queue_capacity=20 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        
        # Get for SSD network: image, labels, bboxes.
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])
        glabels -= self.labels_offset
        
        
        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = self.__preprocess_data(image, glabels, gbboxes)
        
#         network_fn = nets_factory.get_network_fn(
#                 self.model_name,
#                 num_classes=(dataset.num_classes - self.labels_offset),
#                 weight_decay=self.weight_decay,
#                 is_training=True)

        # Assign groundtruth information for all default/anchor boxes
        gclasses, glocalisations, gscores = g_ssd_model.tf_ssd_bboxes_encode(glabels, gbboxes)
        
        #tf.train.batch accepts only list of tensors, this batch shape can used to
        #flatten the list in list, and later on convet it back to list in list.
        batch_shape = [1] + [len(gclasses), len(glocalisations), len(gscores)]
        #Batch the samples
        batch = tf.train.batch(
                tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
                batch_size=self.batch_size,
                num_threads=self.num_preprocessing_threads,
                capacity=5 * self.batch_size)

        batch_queue = slim.prefetch_queue.prefetch_queue(
                batch, capacity=2)
        batch_queue_dequed = batch_queue.dequeue()
        
        #convert it back to the list in list format which allows us to easily use later on
        batch_queue_dequed= tf_utils.reshape_list(batch_queue_dequed, batch_shape)
#         
#         self.network_fn = network_fn
#         self.dataset = dataset
        
        #set up the network
        
        return batch_queue_dequed
    def __disp_image(self, img, shape_data, classes, bboxes):
        scores =np.full(classes.shape, 1.0)
        visualization.plt_bboxes(img, classes, scores, bboxes,title='Ground Truth')
        return
    def __disp_matched_anchors(self,img, target_labels_data, target_localizations_data, target_scores_data):
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
            
            
            
            bboxes_gt = g_ssd_model.ssd_bboxes_decode(target_localizations_data[i][pos_sample_inds], 
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
                
            
            
        return
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
        
    
    def run(self):
       
        #fine tune the new parameters
        self.train_dir = './logs/'
        
        
        self.model_name = 'ssd'
        
        
        
        with tf.Graph().as_default():
            batch_voc_2007_train = self.get_voc_2007_train_data()
            batch_voc_2007_test = self.get_voc_2007_test_data()
            batch_voc_2012_train = self.get_voc_2012_train_data()
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):
                    for i in range(1):
                        for current_data in [batch_voc_2007_test]:
                       
                            image_data, target_labels_data, target_localizations_data, target_scores_data = sess.run(list(current_data))
                            
                            
                            #selet the first image in the batch
                            target_labels_data = [item[0] for item in target_labels_data]
                            target_localizations_data = [item[0] for item in target_localizations_data]
                            target_scores_data = [item[0] for item in target_scores_data]
                            image_data = image_data[0]
    
                            image_data = np_image_unwhitened(image_data)
#                             self.__disp_image(image_data, shape_data, glabels_data, gbboxes_data)
                            self.__disp_matched_anchors(image_data,target_labels_data, target_localizations_data, target_scores_data)
                            plt.show()
                        
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()