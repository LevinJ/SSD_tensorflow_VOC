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



class PrepareData():
    def __init__(self):
        self.dataset_name = 'flowers'
        self.dataset_split_name = 'train'
        self.dataset_dir = '/home/levin/workspace/detection/data/flower'
        
        self.num_readers = 4
        self.batch_size = 32
        self.labels_offset = 0
        self.train_image_size = None
        self.model_name = 'inception_v3' #'The name of the architecture to train.'
        self.weight_decay = 0.00004 # 'The weight decay on the model weights.'
        
        self.preprocessing_name = None
        self.num_preprocessing_threads = 4
        
        
        return
    def __get_images_labels_bboxes(self):
        dataset = dataset_factory.get_dataset(
                self.dataset_name, self.dataset_split_name, self.dataset_dir)
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=True,
                    num_readers=self.num_readers,
                    common_queue_capacity=20 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        
        # Get for SSD network: image, labels, bboxes.
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])
        glabels -= self.labels_offset
        
        # Select the preprocessing function.
        preprocessing_name = g_ssd_model.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)
        # Pre-processing image, labels and bboxes.
        image, glabels, gbboxes = \
            image_preprocessing_fn(image, glabels, gbboxes, g_ssd_model.img_shape)
        
#         network_fn = nets_factory.get_network_fn(
#                 self.model_name,
#                 num_classes=(dataset.num_classes - self.labels_offset),
#                 weight_decay=self.weight_decay,
#                 is_training=True)

        # Assign groundtruth information for all default/anchor boxes
        target_labels, target_localizations, target_scores = g_ssd_model.tf_ssd_bboxes_encode(glabels, gbboxes)
#         batch_shape = [1] + [len(ssd_anchors)] * 3  
#         images, labels = tf.train.batch(
#                 [image, label],
#                 batch_size=self.batch_size,
#                 num_threads=self.num_preprocessing_threads,
#                 capacity=5 * self.batch_size)
#         labels = slim.one_hot_encoding(
#                 labels, dataset.num_classes - self.labels_offset)
#         batch_queue = slim.prefetch_queue.prefetch_queue(
#                 [images, labels], capacity=2)
#         images, labels = batch_queue.dequeue()
#         
#         self.network_fn = network_fn
#         self.dataset = dataset
        
        #set up the network
        
        return image, shape, glabels, gbboxes,target_labels, target_localizations, target_scores
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
            
            bboxes = bboxes_default
            neg_marks = marks_default
            add_gt = True
            if add_gt :
                bboxes = np.vstack([bboxes_default,bboxes_gt])
                neg_marks = np.hstack([marks_default,marks_gt])
                classes = np.tile(classes, 2)
                scores = np.tile(scores,2)
            
            title = "Default boxes: Layer {}".format(g_ssd_model.feat_layers[i])
            visualization.plt_bboxes(img, classes, scores, bboxes,neg_marks=neg_marks,title=title)
                
            
            
        return
    
    def run(self):
       
        #fine tune the new parameters
        self.train_dir = './logs/'
        
        self.dataset_name = 'pascalvoc_2007'
        self.dataset_split_name = 'train'
        self.dataset_dir = '../data/voc/tfrecords/'
        self.model_name = 'ssd'
        
        
        
        with tf.Graph().as_default():
            image, shape, glabels, gbboxes,target_labels, target_localizations, target_scores = self.__get_images_labels_bboxes()
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):
                
                    for i in range(3):
                       
                        image_data, shape_data, glabels_data, gbboxes_data,target_labels_data, target_localizations_data, target_scores_data = sess.run([image, shape, glabels, gbboxes,target_labels, target_localizations, target_scores])

                        if i !=2 :
                            continue
                        image_data = np_image_unwhitened(image_data)
                        self.__disp_image(image_data, shape_data, glabels_data, gbboxes_data)
                        self.__disp_matched_anchors(image_data,target_labels_data, target_localizations_data, target_scores_data)
                        
        
        plt.show()
        
        return
    
    


if __name__ == "__main__":   
    obj= PrepareData()
    obj.run()