from threading import Thread 
class RuningAI(Thread):
    
    def __init__(self, AIOBJ):
        Thread.__init__(self)
        self.AIOBJ = AIOBJ
        
    def run(self):
        self.AIOBJ.CONFIG_SESSION()
        self.AIOBJ.PREDECT_PHOTO()




                
    

            