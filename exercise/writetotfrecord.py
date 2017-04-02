import tensorflow as tf
import os



def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
        values: A scalar or list of values.

    Returns:
        a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(image_format),
            'image/class/label': int64_feature(class_id),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
    }))
    
def read_image_dims(sess, image_data):
        
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        image = sess.run(decode_jpeg,
                                         feed_dict={decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image.shape[0], image.shape[1]
  
  
with tf.Graph().as_default():
    with tf.Session('') as sess:
        output_filename = '/tmp/flowers_test/flowers.tfrecord'
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        
            #get numpy array
            filename = '/home/levin/workspace/detection/data/flower/raw/flower_photos/daisy/5547758_eea9edfd54_n.jpg'
            image_data = tf.gfile.FastGFile(filename, 'r').read()
            
            
            
            height, width = read_image_dims(sess, image_data)
    
            class_name = os.path.basename(os.path.dirname(filename))
            class_id = 0
            
            example = image_to_tfexample(
                                    image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())
        
        
        
        
        
        