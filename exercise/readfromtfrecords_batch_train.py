from datasets import dataset_factory
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np


class ReadRecordsBatchTrain(object):
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
    def disp_image(self,image, label):
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.title(str(label))
        plt.show()
        return
    def __get_images_labels(self):
        dataset = dataset_factory.get_dataset(
                self.dataset_name, self.dataset_split_name, self.dataset_dir)
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=self.num_readers,
                    common_queue_capacity=20 * self.batch_size,
                    common_queue_min=10 * self.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= self.labels_offset
        
#         network_fn = nets_factory.get_network_fn(
#                 self.model_name,
#                 num_classes=(dataset.num_classes - self.labels_offset),
#                 weight_decay=self.weight_decay,
#                 is_training=True)
# 
#         train_image_size = self.train_image_size or network_fn.default_image_size
#         
#         preprocessing_name = self.preprocessing_name or self.model_name
#         image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#                 preprocessing_name,
#                 is_training=True)
# 
#         image = image_preprocessing_fn(image, train_image_size, train_image_size)

#         images, labels = tf.train.batch(
#                 [image, label],
#                 batch_size=self.batch_size,
#                 num_threads=self.num_preprocessing_threads,
#                 capacity=5 * self.batch_size)
#         labels = slim.one_hot_encoding(
#                 labels, dataset.num_classes - FLAGS.labels_offset)
#         batch_queue = slim.prefetch_queue.prefetch_queue(
#                 [images, labels], capacity=2 * deploy_config.num_clones)
        
        return image, label
    
    def run_1(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        with tf.Graph().as_default():
            print("create symbolic op")
            images, labels = self.__get_images_labels()
            
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
#                 with slim.queues.QueueRunners(sess):
                coord = tf.train.Coordinator()

                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
                for _ in range(2):
                    
                    images_data, labels_data = sess.run([images, labels]) 
                    self.disp_image(images_data, labels_data)
                coord.request_stop()
                coord.join(threads)
        return
    
    def run(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        with tf.Graph().as_default():
            print("create symbolic op")
            images, labels = self.__get_images_labels()
            
            with tf.Session('') as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                with slim.queues.QueueRunners(sess):
                    for _ in range(2):
                        
                        images_data, labels_data = sess.run([images, labels]) 
                        self.disp_image(images_data, labels_data)
        return
    
    


if __name__ == "__main__":   
    obj= ReadRecordsBatchTrain()
    obj.run()