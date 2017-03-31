import tensorflow as tf
from datasets import dataset_utils
from datasets import  flowers
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from datasets import download_and_convert_flowers
import numpy as np
from preprocessing import inception_preprocessing

flowers_data_dir = '../../data/flower'
train_dir = '/tmp/tfslim_model/'
print('Will save model to %s' % train_dir)

def display_data():
    
    with tf.Graph().as_default(): 
        dataset = flowers.get_split('train', flowers_data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)
        image, label = data_provider.get(['image', 'label'])
        
        with tf.Session() as sess:    
            with slim.queues.QueueRunners(sess):
                for i in range(4):
                    np_image, np_label = sess.run([image, label])
                    height, width, _ = np_image.shape
                    class_name = name = dataset.labels_to_names[np_label]
                    
                    plt.figure()
                    plt.imshow(np_image)
                    plt.title('%s, %d x %d' % (name, height, width))
                    plt.axis('off')
                    plt.show()
    return

def download_convert():
    dataset_dir = flowers_data_dir
    download_and_convert_flowers.run(dataset_dir)
    return

def disp_data():
    with tf.Graph().as_default(): 
        dataset = flowers.get_split('train', flowers_data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)
        image, label,format = data_provider.get(['image', 'label', 'format'])
        
        with tf.Session() as sess:    
            with slim.queues.QueueRunners(sess):
                for i in range(4):
                    np_image, np_label,np_format = sess.run([image, label,format])
                    height, width, _ = np_image.shape
                    class_name = name = dataset.labels_to_names[np_label]
                    
                    plt.figure()
                    plt.imshow(np_image)
                    plt.title('%s, %d x %d' % (name, height, width))
                    plt.axis('off')
                    plt.show()
                
    return

def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)       
    return net

def apply_random_image():
    with tf.Graph().as_default():
        # The model can handle any input size because the first layer is convolutional.
        # The size of the model is determined when image_node is first passed into the my_cnn function.
        # Once the variables are initialized, the size of all the weight matrices is fixed.
        # Because of the fully connected layers, this means that all subsequent images must have the same
        # input size as the first image.
        batch_size, height, width, channels = 3, 28, 28, 3
        images = tf.random_uniform([batch_size, height, width, channels], maxval=1)
        
        # Create the model.
        num_classes = 10
        logits = my_cnn(images, num_classes, is_training=True)
        probabilities = tf.nn.softmax(logits)
      
        # Initialize all the variables (including parameters) randomly.
        init_op = tf.global_variables_initializer()
      
        with tf.Session() as sess:
            # Run the init_op, evaluate the model outputs and print the results:
            sess.run(init_op)
            probabilities = sess.run(probabilities)
            
    print('Probabilities Shape:')
    print(probabilities.shape)  # batch_size x num_classes 
    
    print('\nProbabilities:')
    print(probabilities)
    
    print('\nSumming across all classes (Should equal 1):')
    print(np.sum(probabilities, 1)) # Each row sums to 1
    return

def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
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


def train_save_model():
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
    
        dataset = flowers.get_split('train', flowers_data_dir)
        images, _, labels = load_batch(dataset)
      
        # Create the model:
        logits = my_cnn(images, num_classes=dataset.num_classes, is_training=True)
     
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()
    
        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', total_loss)
      
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
    
        # Run the training:
        final_loss = slim.learning.train(
          train_op,
          logdir=train_dir,
          number_of_steps=1, # For speed, we just do 1 epoch
          save_summaries_secs=1)
      
        print('Finished training. Final batch loss %d' % final_loss)
    return


def evaluate_model():
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)
        
        dataset = flowers.get_split('train', flowers_data_dir)
        images, _, labels = load_batch(dataset)
        
        logits = my_cnn(images, num_classes=dataset.num_classes, is_training=False)
        predictions = tf.argmax(logits, 1)
        
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'eval/Recall@5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
        })
    
        print('Running evaluation Loop...')
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        metric_values = slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=train_dir,
            eval_op=names_to_updates.values(),
            final_op=names_to_values.values())
    
        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))

    return

    
    
def main():
#     download_convert()
#     disp_data()
#     apply_random_image()
#     train_save_model()
    evaluate_model()
    return


main()



