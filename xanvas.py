import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent, width=3.5, height=0.88, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)
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

        self.axes.scatter(x,y)
        self.axes.set_ylim(0, 1920)

        self.axes.invert_yaxis()
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        self.draw()

    def plot(self):

        x, y = np.array(spliArray(self.array,5)).T
        
        self.axes.scatter(x,y)
        self.axes.set_ylim(0, 1920)

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