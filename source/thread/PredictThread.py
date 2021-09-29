import os
from threading import Thread 
from source.frcnn import nn_arch_vgg16 as nn
from natsort import natsorted
class RuningAI(Thread):
    
    def __init__(self, AIOBJ):
        Thread.__init__(self)
        self.AIOBJ = AIOBJ
        self.test_images_list = natsorted(os.listdir("temp/"))
        self.array = []
        for i in self.test_images_list:
            self.array.append("temp/"+str(i))
        print(self.array)
        
    def run(self):
        # self.AIOBJ.CONFIG_SESSION()
        self.AIOBJ.PREDECT_PHOTO(network_arch = nn ,test_images_list = self.array)




                
    

            
            # def PREDECT_PHOTO(self,test_images_list = ['source/DATASET/Testing_Dataset/images/crack17.jpg'],  
            #     network_arch = 'vgg',
            #     config_filename="source/model/config.pickle", 
            #     preprocessing_function = None,
            #     num_rois=32,
            #     final_classification_threshold = 0.8):