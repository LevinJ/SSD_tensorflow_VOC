import os
import re
import argparse

class RunAllCheckpoints(object):
    def __init__(self):
        
        
        return
    
    
    
    def get_all_checkpoints(self,checkpoint_path):
        
        with open(self.checkpoint_path + "checkpoint") as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        checkpoints = []
        for line in content:
            m = re.search('all_model_checkpoint_paths: "model.ckpt-(.*)"', line)
            if m:
                num = m.group(1)
                checkpoints.append(num)
        min_step = 100
        step = 100
        last_step = min_step
        sel_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint = int(checkpoint)
            if checkpoint < min_step:
                continue
            if checkpoint == int(checkpoints[-1]):
                #the last checkpoint always get selected
                sel_checkpoints.append(checkpoint)
                continue
            if checkpoint >= last_step:
                sel_checkpoints.append(checkpoint)
                last_step = last_step + step
        if self.check_only_latest:
            #if we only want to evluate the latest checkpoints
            sel_checkpoints = [sel_checkpoints[-1]]
        return sel_checkpoints
    def parse_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--finetune',  help='whether use checkpoints under finetune folder',  action='store_true')
        parser.add_argument('-l', '--latest',  help='evaluate only the latest checkpoints',  action='store_true')
        args = parser.parse_args()
        self.checkpoint_path = './logs/'
        if args.finetune:
            self.checkpoint_path = './logs/finetune'
        self.check_only_latest = args.latest
            
        return
    def run_all_checkpoints(self):
        self.parse_param()
        
        sel_checkpoints = self.get_all_checkpoints(self.checkpoint_path)
                
        self.eval_during_training = False;
        self.eval_one_time = True
        
        for checkpoint in sel_checkpoints:
            for eval_train in [True, False]:
                
                checkpoint_file = self.checkpoint_path + "model.ckpt-" + str(checkpoint)
                if eval_train:
                    data = "train"
                else:
                    data = "test"
                print("checkpoint {}, {} data".format(checkpoint_file, data))
                
                cmd_str = "python ./evaluate_model.py "
                if eval_train:
                    cmd_str = cmd_str + " -t "
#                 cmd_str = cmd_str + " -c " + self.checkpoint_path
                cmd_str = '{} -c "{}"'.format(cmd_str, checkpoint_file)
                os.system(cmd_str)
            
        return
    
    
    def run(self):
        self.parse_param()
        self.run_all_checkpoints()
        
        
        
        return
    
    


if __name__ == "__main__":   
    obj= RunAllCheckpoints()
    obj.run()