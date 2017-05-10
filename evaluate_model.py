import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import numpy as np
import math
from preparedata import PrepareData
from nets.ssd import g_ssd_model
import tf_extended as tfe
import time
from postprocessingdata import g_post_processing_data


class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        
        
        self.batch_size = 32
        self.labels_offset = 0
        self.eval_image_size = None
        self.preprocessing_name = None
        self.model_name = 'inception_v3'
        
        self.num_preprocessing_threads = 4
        
        self.checkpoint_path =  '/tmp/tfmodel/'
        self.eval_dir = '/tmp/tfmodel/'
        
        
        return
    
    
    def __setup_eval(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf_global_step = slim.get_or_create_global_step()
        
        if not self.eval_during_training:
            image, filename, glabels,gbboxes,gdifficults, gclasses, glocalisations, gscores = self.get_voc_2007_train_data(is_training_data=False)
            self.eval_dir = './logs/evals/train_data'
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
            image, filename, glabels,gbboxes,gdifficults, gclasses, glocalisations, gscores = self.get_voc_2007_test_data()
            self.eval_dir = './logs/evals/test_data'
        
       
        
        #get model outputs
        predictions, localisations, logits, end_points = g_ssd_model.get_model(image)
        
        
            
#         print_mAP_07_op, print_mAP_12_op = g_post_processing_data.get_mAP_tf_current_batch(predictions, localisations, glabels, gbboxes, gdifficults)
            
        names_to_updates = g_post_processing_data.get_mAP_tf_accumulative(predictions, localisations, glabels, gbboxes, gdifficults)
#         print_filename_op = tf.Print(filename, [filename], "input images: ")
        
        variables_to_restore = slim.get_variables_to_restore()
        
        num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
        
        
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        
        
        if not self.eval_during_training:
            # Standard evaluation loop.
            print("one time evaluate...")
            if tf.gfile.IsDirectory(self.checkpoint_path):
                checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_path)
            else:
                checkpoint_file = self.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_file)
            start = time.time()
            slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_file,
                logdir=self.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()) ,
                session_config=config,
                variables_to_restore=variables_to_restore)
            # Log time spent.
            elapsed = time.time()
            elapsed = elapsed - start
            print('Time spent : %.3f seconds.' % elapsed)
            print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))
        else:
            print("evaluate during training...")
            # Waiting loop.
            slim.evaluation.evaluation_loop(
                master='',
                checkpoint_dir=self.checkpoint_path,
                logdir=self.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore,
                eval_interval_secs=60*60,
                session_config=config,
                timeout=None)
        
        

        
        
        
        
        return
    
    
    def run(self):
        
        
        self.checkpoint_path = './logs/'
        self.fine_tune_vgg16 = True
        
        if self.fine_tune_vgg16: 
            self.checkpoint_path = './logs/finetune'
        
        
        self.eval_dir = './logs/evals/'
        self.eval_during_training = True;
        
        if self.eval_during_training:
            self.batch_size = 1
            #To evaluate while trainin going on
            with tf.device('/device:CPU:0'):      
                self.__setup_eval()
        else:
            self.__setup_eval()
                    
        
        
        return
    
    


if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()