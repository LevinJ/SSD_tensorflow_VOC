import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

filenames = ['/home/levin/workspace/detection/data/flower/flowers_train_00000-of-00005.tfrecord',
             '/home/levin/workspace/detection/data/flower/flowers_train_00001-of-00005.tfrecord',
             '/home/levin/workspace/detection/data/flower/flowers_train_00002-of-00005.tfrecord']

def read_and_decode_single_example(filename_queue):

    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        })
    # now return the converted data
    label = features['image/class/label']
    image = features['image/encoded']

#     image = tf.image.decode_jpeg(image, channels=3)
    image_format = features['image/format']
    
   
    
    return label, image, image_format

def input_pipeline(filenames, batch_size=32, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    label, image, image_format = read_and_decode_single_example(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    label_batch, image_batch, image_format_batch = tf.train.shuffle_batch(
        [label, image, image_format], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return label_batch, image_batch, image_format_batch

def input_pipeline_2(filenames, batch_size=32, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_and_decode_single_example(filename_queue)
                  for _ in range(2)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    label_batch, image_batch, image_format_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

    return label_batch, image_batch, image_format_batch


def read_image(sess, image_data):
        
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        image = sess.run(decode_jpeg,
                                         feed_dict={decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def disp_image(image, label):
    plt.figure()
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    plt.title(str(label))
    plt.show()
    return
with tf.Graph().as_default():
    with tf.Session('') as sess:
        labels_batch, images_batch, image_format_batch = input_pipeline_2(filenames)
        init = tf.initialize_all_variables()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        
        for _ in range(5):
            label_data, image_data,image_format_data = sess.run([labels_batch, images_batch,image_format_batch])
            print(label_data)
#             image_data = read_image(sess, image_data)
#             disp_image(image_data, label_data)
            
            
            
     