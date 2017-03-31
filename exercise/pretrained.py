import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import imagenet
from nets import inception
from nets import vgg
from preprocessing import inception_preprocessing
from preprocessing import vgg_preprocessing

import tensorflow.contrib.slim as slim
from utility.dumpload import DumpLoad
from datasets import flowers
import math


class PreTrained():
    def __init__(self):
        return
      
    def use_inceptionv4(self):
        image_size = inception.inception_v4.default_image_size
        img_path = "../../data/misec_images/EnglishCockerSpaniel_simon.jpg"
        checkpoint_path = "../../data/trained_models/inception_v4/inception_v4.ckpt"

        with tf.Graph().as_default():
           
            image_string = tf.read_file(img_path)
            image = tf.image.decode_jpeg(image_string, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_images  = tf.expand_dims(processed_image, 0)
            
            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v4_arg_scope()):
                logits, _ = inception.inception_v4(processed_images, num_classes=1001, is_training=False)
            probabilities = tf.nn.softmax(logits)
            
            init_fn = slim.assign_from_checkpoint_fn(
                checkpoint_path,
                slim.get_model_variables('InceptionV4'))
            
            with tf.Session() as sess:
                init_fn(sess)
                np_image, probabilities = sess.run([image, probabilities])
                probabilities = probabilities[0, 0:]
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
                self.disp_names(sorted_inds,probabilities)
                
            plt.figure()
            plt.imshow(np_image.astype(np.uint8))
            plt.axis('off')
            plt.title(img_path)
            plt.show()
            
            
        
        return
    def disp_names(self, sorted_inds,probabilities, include_background=True):
        dump_load = DumpLoad("../../data/imagenet/imagenet_labels_dict.pickle")
        if dump_load.isExisiting():
            names = dump_load.load()
        else:
            names = imagenet.create_readable_names_for_imagenet_labels()
            dump_load.dump(names)
            
        for i in range(5):
            index = sorted_inds[i]
            if include_background:
                print('Probability %0.2f%% => [%s]' % (probabilities[index], names[index]))
            else:
                print('Probability %0.2f%% => [%s]' % (probabilities[index], names[index+1]))
        return
    def use_vgg16(self):
        
        with tf.Graph().as_default():
            image_size = vgg.vgg_16.default_image_size
            img_path = "../../data/misec_images/First_Student_IC_school_bus_202076.jpg"
            checkpoint_path = "../../data/trained_models/vgg16/vgg_16.ckpt"
            
            image_string = tf.read_file(img_path)
            image = tf.image.decode_jpeg(image_string, channels=3)
            processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_images  = tf.expand_dims(processed_image, 0)
            
            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(vgg.vgg_arg_scope()):
                # 1000 classes instead of 1001.
                logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
                probabilities = tf.nn.softmax(logits)
                
                init_fn = slim.assign_from_checkpoint_fn(
                    checkpoint_path,
                    slim.get_model_variables('vgg_16'))
                
                with tf.Session() as sess:
                    init_fn(sess)
                    np_image, probabilities = sess.run([image, probabilities])
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
                    self.disp_names(sorted_inds,probabilities,include_background=False)
                    
                plt.figure()
                plt.imshow(np_image.astype(np.uint8))
                plt.axis('off')
                plt.title(img_path)
                plt.show()
        return
    def get_init_fn(self, checkpoint_path):
        """Returns a function run by the chief worker to warm-start the training."""
        checkpoint_exclude_scopes=["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
    
        return slim.assign_from_checkpoint_fn(
          checkpoint_path,
          variables_to_restore)
    def load_batch(self, dataset, batch_size=32, height=299, width=299, is_training=False):
        """Loads a single batch of data.
        
        Args:
          dataset: The dataset to load.
          batch_size: The number of images in the batch.
          height: The size of each image after preprocessing.
          width: The size of each image after preprocessing.
          is_training: Whether or not we're currently training or evaluating.
        
        Returns:
          images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
          images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
          labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
        """
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32,
            common_queue_min=8)
        image_raw, label = data_provider.get(['image', 'label'])
        
        # Preprocess image for usage by Inception.
        image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
        
        # Preprocess the image for display purposes.
        image_raw = tf.expand_dims(image_raw, 0)
        image_raw = tf.image.resize_images(image_raw, [height, width])
        image_raw = tf.squeeze(image_raw)
    
        # Batch it up.
        images, images_raw, labels = tf.train.batch(
              [image, image_raw, label],
              batch_size=batch_size,
              num_threads=1,
              capacity=2 * batch_size)
        
        return images, images_raw, labels
    def fine_tune_inception(self):
        train_dir = '/tmp/inception_finetuned/'
        image_size = inception.inception_v4.default_image_size
        checkpoint_path = "../../data/trained_models/inception_v4/inception_v4.ckpt"
        flowers_data_dir = "../../data/flower"
        
        
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            
            dataset = flowers.get_split('train', flowers_data_dir)
            images, _, labels = self.load_batch(dataset, height=image_size, width=image_size)
            
            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v4_arg_scope()):
                logits, _ = inception.inception_v4(images, num_classes=dataset.num_classes, is_training=True)
                
            # Specify the loss function:
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
            total_loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
#             total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
#             total_loss = slim.losses.get_total_loss()
        
            # Create some summaries to visualize the training process:
            tf.summary.scalar('losses/Total_Loss', total_loss)
          
            # Specify the optimizer and create the train op:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = slim.learning.create_train_op(total_loss, optimizer)
            
            # Run the training:
            number_of_steps = math.ceil(dataset.num_samples/32) * 1
            final_loss = slim.learning.train(
                train_op,
                logdir=train_dir,
                init_fn=self.get_init_fn(checkpoint_path),
                number_of_steps=number_of_steps)
        
  
            print('Finished training. Last batch loss %f' % final_loss)
        return
    
    def use_fined_model(self):
        image_size = inception.inception_v4.default_image_size
        batch_size = 3
        flowers_data_dir = "../../data/flower"
        train_dir = '/tmp/inception_finetuned/'
        
        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            
            dataset = flowers.get_split('train', flowers_data_dir)
            images, images_raw, labels = self.load_batch(dataset, height=image_size, width=image_size)
            
            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v4_arg_scope()):
                logits, _ = inception.inception_v4(images, num_classes=dataset.num_classes, is_training=True)
        
            probabilities = tf.nn.softmax(logits)
            
            checkpoint_path = tf.train.latest_checkpoint(train_dir)
            init_fn = slim.assign_from_checkpoint_fn(
              checkpoint_path,
              slim.get_variables_to_restore())
            
            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    sess.run(tf.initialize_local_variables())
                    init_fn(sess)
                    np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])
            
                    for i in range(batch_size): 
                        image = np_images_raw[i, :, :, :]
                        true_label = np_labels[i]
                        predicted_label = np.argmax(np_probabilities[i, :])
                        predicted_name = dataset.labels_to_names[predicted_label]
                        true_name = dataset.labels_to_names[true_label]
                        
                        plt.figure()
                        plt.imshow(image.astype(np.uint8))
                        plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                        plt.axis('off')
                        plt.show()
                return
        
    def run(self):
#         self.use_inceptionv4()
#         self.use_vgg16()
#         self.fine_tune_inception()
        self.use_fined_model()
        
        return
    
    



    


if __name__ == "__main__":   
    obj= PreTrained()
    obj.run()