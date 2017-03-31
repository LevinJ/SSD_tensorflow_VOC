import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

import tensorflow.contrib.slim as slim
from utility.dumpload import DumpLoad


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
            plt.show()
            
            
        
        return
    def disp_names(self, sorted_inds,probabilities):
        dump_load = DumpLoad("../../data/imagenet/imagenet_labels_dict.pickle")
        if dump_load.isExisiting():
            names = dump_load.load()
        else:
            names = imagenet.create_readable_names_for_imagenet_labels()
            dump_load.dump(names)
            
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index], names[index]))
        return
    def run(self):
        self.use_inceptionv4()
        
        return
    
    



    


if __name__ == "__main__":   
    obj= PreTrained()
    obj.run()