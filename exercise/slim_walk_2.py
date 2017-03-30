import tensorflow as tf
from datasets import dataset_utils
from datasets import  flowers
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from datasets import download_and_convert_flowers

flowers_data_dir = '../../data/flower'

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

def main():
    download_convert()
    return


main()



