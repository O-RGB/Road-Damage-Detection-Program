import re
from threading import Thread 
import os
import cv2

class Counter(Thread):

    def __init__(self, DIR,fps, GUI):
        Thread.__init__(self)
        self.DIR = DIR
        self.GUI = GUI
        self.fps = fps
        self.fileGPS = []
        with open('C:/Users/okoza/Desktop/gps.txt') as f:
            lines = [line.rstrip() for line in f]
        for i in lines:
            self.fileGPS.append(i.split(", "))

    def run(self):
        dirname = "temp"
        dirname2 = "tempVideo"
        dirname3 = "detectOut"
        try:
            os.makedirs(dirname)
            os.makedirs(dirname2)
            os.makedirs(dirname3) 
        except OSError:
            if os.path.exists(dirname) or os.path.exists(dirname2)  or os.path.exists(dirname3):
                pass
            else:
                raise
        
        i = 0
        j = 0
        cap = cv2.VideoCapture(self.DIR,0) 
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for_lool = int(frame_count/self.fps)
        WriteFileTemp(0,self.fileGPS[0][0],self.fileGPS[0][1],0)
        km_h = 0
        # self.GUI.LOADING(0)
        # self.GUI.RuningFalse(False)
        while True:
            if (j != 0) and (j < len(self.fileGPS)-1):
                lat1,long1,lat2,long2 = self.fileGPS[j-1][0],self.fileGPS[j-1][1],self.fileGPS[j][0],self.fileGPS[j][1]
                km_h = haversine(lat1,long1,lat2,long2)
                if km_h*3600 > 35:
                    newArrayLatLong = interpData(lat1,long1,lat2,long2)
                    km_hx = haversine(lat1,long1,newArrayLatLong[0][0],newArrayLatLong[0][1])
                    WriteFileTemp(j,newArrayLatLong[0][0],newArrayLatLong[0][1],km_hx,"temp/0{}.0.jpg")

                    cap.set(1,i - 15); 
                    # print("---",i- 15)
                    _, img = cap.read()
                    cv2.imwrite("temp/0{}.0.jpg".format(j),img)

                WriteFileTemp(j,self.fileGPS[j][0],self.fileGPS[j][1],km_h)
            
            cap.set(1,i); 
            eat, img = cap.read()
            cv2.imwrite("temp/0{}.jpg".format(j),img)
            # print(i)
            i = i+self.fps
            j = j + 1
            # self.GUI.LOADING(int(j*100/for_lool))
            if i+self.fps > frame_count:
                break
        # self.GUI.RuningFalse(True)
            

def WriteFileTemp(loop,gpsLat,gpslong,km_h,filename = "temp/0{}.jpg"):
    f = open("temp.txt", "a")
    strd = str(loop)+","+str(gpsLat)+","+str(gpslong)+","+str(int(km_h*3600))+","+str("{:.2f}".format(km_h*1000))+","+str(filename.format(loop))+"\n"
    f.write(strd)
    f.close()

from math import radians, cos, sin, asin, sqrt
def haversine(lat1 ,lon1 ,lat2 ,lon2):
    lat1 ,lon1 ,lat2 ,lon2 = coverStrtoFloat(lat1 ,lon1 ,lat2 ,lon2)

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

import numpy as np
def interpData(lat1 ,lon1 ,lat2 ,lon2):
    lat1 ,lon1 ,lat2 ,lon2 = coverStrtoFloat(lat1 ,lon1 ,lat2 ,lon2)
    
    times = [0, 2]
    lat = [lat1, lat2]
    lon = [lon1, lon2]
    t = np.arange(0, times[len(times)-1]+1)

    latint = np.interp(t, times, lat)
    lonint = np.interp(t, times, lon)

    array = []
    for i in range(len(latint)):
        if i != 0 and i != len(latint)-1:
            array.append([latint[i],lonint[i]])
    return array

def coverStrtoFloat(lat1 ,lon1 ,lat2 ,lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    return lat1 ,lon1 ,lat2 ,lon2

