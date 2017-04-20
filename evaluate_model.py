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
        
        # Performing post-processing on CPU: loop-intensive, usually more efficient.
        with tf.device('/device:CPU:0'):
        
            # Detected objects from SSD output.
            localisations = g_ssd_model.decode_bboxes_all_ayers_tf(localisations)
            
            rscores, rbboxes = g_ssd_model.detected_bboxes(predictions, localisations)
            
            # Compute TP and FP statistics.
            num_gbboxes, tp, fp, rscores = \
                tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          glabels, gbboxes, gdifficults)
            
        variables_to_restore = slim.get_variables_to_restore()
        m_AP_tf = g_post_processing_data.get_mAP_tf(predictions, localisations, glabels, gbboxes, gdifficults)
        
        dict_metrics = {}
        with tf.device('/device:CPU:0'):
            # First add all losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            # Extra losses as well.
            for loss in tf.get_collection('EXTRA_LOSSES'):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
    
            # Add metrics to summaries and Print on screen.
            for name, metric in dict_metrics.items():
                # summary_name = 'eval/%s' % name
                summary_name = name
                op = tf.summary.scalar(summary_name, metric[0], collections=[])
                # op = tf.Print(op, [metric[0]], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
            # FP and TP metrics.
            tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
            for c in tp_fp_metric[0].keys():
                dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                                    tp_fp_metric[1][c])
                
            # Add to summaries precision/recall values.
            aps_voc07 = {}
            aps_voc12 = {}
            for c in tp_fp_metric[0].keys():
                # Precison and recall values.
                prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])
    
                # Average precision VOC07.
                v = tfe.average_precision_voc07(prec, rec)
                summary_name = 'AP_VOC07/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc07[c] = v
    
                # Average precision VOC12.
                v = tfe.average_precision_voc12(prec, rec)
                summary_name = 'AP_VOC12/%s' % c
                op = tf.summary.scalar(summary_name, v, collections=[])
                # op = tf.Print(op, [v], summary_name)
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                aps_voc12[c] = v
    
            # Mean average precision VOC07.
            summary_name = 'AP_VOC07/mAP'
            mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
            # Mean average precision VOC12.
            summary_name = 'AP_VOC12/mAP'
            mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
            op = tf.summary.scalar(summary_name, mAP, collections=[])
            op = tf.Print(op, [mAP], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            
            current_step = tf.Print(tf_global_step, [tf_global_step], 'current_step')
            tf.summary.scalar('current_step_summary', current_step)
            
        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)
        
        if tf.gfile.IsDirectory(self.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_path)
        
        num_batches = math.ceil(self.dataset.num_samples / float(self.batch_size))
        
        num_batches = 5
        
        

        # Standard evaluation loop.
        start = time.time()
        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=self.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()) +[current_step],
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