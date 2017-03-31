import pickle
import numpy as np
import os


class DumpLoad:
    def __init__(self, pickle_filepath):
        self.pickle_filepath = pickle_filepath
        dir_path = os.path.dirname(pickle_filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        return
    def load(self):
        with open(self.pickle_filepath, 'rb') as handle:
            dataset = pickle.load(handle)
        return dataset
    def isExisiting(self):
        return os.path.exists(self.pickle_filepath) 
    def run(self):
        self.dump((np.arange(10), np.arange(20)))
        print(self.load())
        return
    
    def dump(self, dataset, protocoal = pickle.HIGHEST_PROTOCOL):
        with open(self.pickle_filepath, 'wb') as f:
            pickle.dump(dataset, f, protocoal)
        return
    

    
if __name__ == "__main__":   
    obj= DumpLoad('./data/in/in/myfile.pickle')
    obj.run()