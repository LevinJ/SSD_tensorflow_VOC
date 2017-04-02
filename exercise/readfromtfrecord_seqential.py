import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

filename = '/home/levin/workspace/detection/data/flower/flowers_train_00000-of-00005.tfrecord'

def read_image(sess, image_data):
        
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        image = sess.run(decode_jpeg,
                                         feed_dict={decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
  
  

for serialized_example in tf.python_io.tf_record_iterator(filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # traverse the Example format to get data
    image_data = example.features.feature['image/encoded'].bytes_list.value[0]
    image_format = example.features.feature['image/format'].bytes_list.value[0]
    image_format = image_format.decode("utf-8")
    height = example.features.feature['image/height'].int64_list.value[0]
    width = example.features.feature['image/width'].int64_list.value[0]
    label = example.features.feature['image/class/label'].int64_list.value[0]
    with tf.Graph().as_default():
        with tf.Session('') as sess:
            image = read_image(sess, image_data)
    
    plt.figure()
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    plt.title(str(label))
    plt.show()
    