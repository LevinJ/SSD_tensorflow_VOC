
import numpy as np
import os
from nets.ssd import g_ssd_model


class EvalVOC:
    def __init__(self):
      
        
        return
    def __convert2np(self,net_outputs):
        temp = []
        for i in range(len(net_outputs)):
            net_output = net_outputs[i]
            
        return
    def eval_voc(self,image, filename,glabels,gbboxes,gdifficults,predictions, localizations):
        localizations = g_ssd_model.decode_bboxes_all_layers(localizations)
        localizations = localizations.reshape((localizations.shape[0], -1, localizations.shape[-1]))
        return
   
    def run(self):
        
        return
   
    
g_eval_voc = EvalVOC()
    
if __name__ == "__main__":   
    obj= EvalVOC()
    obj.run()