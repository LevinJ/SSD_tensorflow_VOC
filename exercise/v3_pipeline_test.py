import tensorflow as tf
from exercise.slim_train_test import SlimTrainMgr
from exercise.slim_eval_test import SlimEvalMgr

class V3PipelineTest(object):
    def __init__(self):
       
        
        return
    
    
    
    
    
    def run(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        train_mgr = SlimTrainMgr()
        eval_mgr = SlimEvalMgr()
        train_mgr.run()
        eval_mgr.run()
        
      
        
        
        return
    
    


if __name__ == "__main__":   
    obj= V3PipelineTest()
    obj.run()