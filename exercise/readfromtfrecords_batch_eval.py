from datasets import dataset_factory
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np


class ReadRecordsBatchEval(object):
    def __init__(self):
        self.dataset_name = 'flowers'
        self.dataset_split_name = 'validation'
        self.dataset_dir = '/home/levin/workspace/detection/data/flower'
        
        self.batch_size = 32
        self.labels_offset = 0
        self.eval_image_size = None
        self.preprocessing_name = None
        self.model_name = 'inception_v3'
        
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
                shuffle=False,
                common_queue_capacity=2 * self.batch_size,
                common_queue_min=self.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= self.labels_offset
        
        network_fn = nets_factory.get_network_fn(
                self.model_name,
                num_classes=(dataset.num_classes - self.labels_offset),
                is_training=False)
        
        preprocessing_name = self.preprocessing_name or self.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=False)

        eval_image_size = self.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
                [image, label],
                batch_size=self.batch_size,
                num_threads=self.num_preprocessing_threads,
                capacity=5 * self.batch_size)
        
        return images, labels
    
    
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
                        images_data = images_data[0]
                        labels_data = labels_data[0]
                        images_data = ((images_data/2 + 0.5)*255).astype(np.uint8)

                        self.disp_image(images_data, labels_data)
        return
    
    


if __name__ == "__main__":   
    obj= ReadRecordsBatchEval()
    obj.run()