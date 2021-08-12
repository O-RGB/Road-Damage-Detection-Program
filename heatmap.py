
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

class heatmap(FigureCanvasQTAgg):
    def __init__(self, parent, width=3.6, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)
        self.axes.set_facecolor((0, 0, 0.49))
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        plt.show()

    def SetArrayPlotUpdate(self,array):
        self.array = array
        if not any(self.array):
            x = []
            y = []
        else:
            x, y = np.array(spliArray(self.array,5)).T
        self.axes.cla()  # Clear the canvas.
        
        img, extent = myplot(x, y, 64)
        self.axes.imshow(img,  origin='lower', cmap=cm.jet)
        self.axes.invert_yaxis()

        self.draw()
       
 
    def plot(self):

        x, y = np.array(spliArray(self.array,5)).T
        
        img, extent = myplot(x, y, 64)
        self.axes.imshow(img,  origin='lower', cmap=cm.jet)
        self.axes.invert_yaxis()

        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        plt.show()


def spliArray(arrayList,indexFps):
    array = []  
    ix,jx = 0,0

    for i in arrayList:
        # if(len(i) == 0):
        #     i = [[0, 0, 0, 0]]
        for j in i:
            array.append([ix,int((j[0]+j[2])/2)])
        ix = ix + indexFps 
    return array  

def myplot(x, y, s, bins=[1920,1000]):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent
            
    
        