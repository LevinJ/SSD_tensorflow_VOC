
from exercise import train_classifier_mgr


import tensorflow as tf

 
class FineTuneInceptionV3Flower(object):
    def __init__(self):
        return
    
    def run(self):
        FLAGS = tf.app.flags.FLAGS
        ## Fine-tune only the new layers for 1000 steps.
        FLAGS.train_dir = '/tmp/flowers-models/inception_v3'
        
        
        FLAGS.dataset_name = 'flowers'
        FLAGS.dataset_split_name = 'train'
        FLAGS.dataset_dir = '/home/levin/workspace/detection/data/flower'
        
        
        FLAGS.model_name = 'inception_v3'
        
        
        FLAGS.checkpoint_path = "/home/levin/workspace/detection/data/trained_models/inception_v3/inception_v3.ckpt"
        FLAGS.checkpoint_exclude_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'
        
        
        FLAGS.trainable_scopes = 'InceptionV3/Logits,InceptionV3/AuxLogits'
        
        FLAGS.max_number_of_steps = 1000
        FLAGS.batch_size = 32
        FLAGS.learning_rate = 0.01
        FLAGS.learning_rate_decay_type = 'fixed'
        FLAGS.save_interval_secs = 60
        FLAGS.save_summaries_secs = 60
        FLAGS.log_every_n_steps = 100
        FLAGS.optimizer = 'rmsprop'
        FLAGS.weight_decay = 0.00004
                
#         train_classifier_mgr.main(None)
        
        
        #Further fine tune
        FLAGS.train_dir = '/tmp/flowers-models/inception_v3/all'
        FLAGS.checkpoint_path = '/tmp/flowers-models/inception_v3'
        FLAGS.max_number_of_steps = 500
        FLAGS.learning_rate = 0.0001
        FLAGS.log_every_n_steps = 10
        
        train_classifier_mgr.main(None)

        
      
        
        
        return
    


if __name__ == '__main__':
    obj = FineTuneInceptionV3Flower()
    obj.run()


