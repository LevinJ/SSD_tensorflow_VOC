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
        
        
        self.batch_size = 2
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
        
        image, filename, glabels,gbboxes,gdifficults, gclasses, glocalisations, gscores = self.get_voc_2007_test_data()
        
        #get model outputs
        predictions, localisations, logits, end_points = g_ssd_model.get_model(image)
        
        
            
        print_mAP_07_op, print_mAP_12_op = g_post_processing_data.get_mAP_tf_current_batch(predictions, localisations, glabels, gbboxes, gdifficults)
            
        names_to_updates = g_post_processing_data.get_mAP_tf_accumulative(predictions, localisations, glabels, gbboxes, gdifficults)
        print_filename_op = tf.Print(filename, [filename], "input images: ")
        
        variables_to_restore = slim.get_variables_to_restore()
        
        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_path)
        
        num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
        
        num_batches = 3
        
        
       
        # Standard evaluation loop.
        start = time.time()
        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=self.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()) + [print_mAP_07_op, print_mAP_12_op,print_filename_op] ,
            variables_to_restore=variables_to_restore)
        # Log time spent.
        elapsed = time.time()
        elapsed = elapsed - start
        print('Time spent : %.3f seconds.' % elapsed)
        print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))
        
        

        
        
        
        
        return
    
    
    def run(self):
        
        
        self.checkpoint_path = './logs/'
        
        
        self.eval_dir = './logs/'
        
        
      
        self.__setup_eval()
        
        
        return
    
    


if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()