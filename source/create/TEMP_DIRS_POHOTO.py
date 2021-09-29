import datetime
import re
from threading import Thread 
import os
import cv2
from math import radians, cos, sin, asin, sqrt
import numpy as np

class Counter(Thread):

    def __init__(self, DIR,fps, GUI):
        Thread.__init__(self)
        self.DIR = DIR
        self.GUI = GUI
        self.fps = fps
        self.fileGPS = []


    def run(self):
        
        NewFolder()
        GPS = ReadFileGpsPath(self.GUI._file_GPS)

        InterLoop = 23
        sumall = int(self.fps/(InterLoop+1))
        IndexFrame, j, km_h, count  = 0, 0, 0, 0 
        VideoCapture = cv2.VideoCapture(self.DIR,0) 
        frame_count = int(VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)) # for_lool = int(frame_count/self.fps) # self.GUI.LOADING(0) # self.GUI.RuningFalse(False)
        
        while True:
            if (j != 0) and (j <= len(GPS)):
                lat1,long1,lat2,long2 = GPS[j-1][0],GPS[j-1][1],GPS[j][0],GPS[j][1]
                km_h = haversine(lat1,long1,lat2,long2)

                km_h_if = int(km_h*3600)
                if km_h_if > 50: InterLoop = 2
                elif km_h_if > 60: InterLoop = 3
                elif km_h_if > 70: InterLoop = 4
               
                InterGPS = interpData(lat1,long1,lat2,long2,InterLoop)
                km_hx = (km_h/(InterLoop+1))
                km_h = (km_h-(km_hx*InterLoop))

                for i in range(InterLoop):
                    IndexInter = (IndexFrame-self.fps)+(sumall*(i+1))
                    WriteFileTemp(count, InterGPS[i+1][0],InterGPS[i+1][1],km_hx,IndexInter/self.fps)
                    capVideo(VideoCapture,IndexInter,j,i+1,"temp/0{}.{}.jpg")
                    count = count + 1

            WriteFileTemp(count,GPS[j][0],GPS[j][1],km_h,IndexFrame/self.fps)
            capVideo(VideoCapture,IndexFrame,j)
            IndexFrame = IndexFrame + self.fps
            InterLoop = 23
            
            j = j + 1 # self.GUI.LOADING(int(j*100/for_lool))
            count = count + 1

            if IndexFrame+self.fps > frame_count or j >= len(GPS):
                break 
        self.GUI.RuningFalse(True)
        self.GUI.START_FOR_AI()
            
def capVideo(cap,i,j,jtemp="",filename = "temp/0{}.jpg"):
    cap.set(1,i); 
    eat, img = cap.read()
    cv2.imwrite(filename.format(j,jtemp),img)

def WriteFileTemp(count,gpsLat,gpslong,km_h,time):
    f = open("temp.txt", "a")
    seconds = time*60
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    time =  ("%02d:%02d:%02d"%(hours,minutes,seconds))
    strd = str(count)+","+str(gpsLat)+","+str(gpslong)+","+str(int(km_h*3600))+","+str("{:.2f}".format(km_h*1000))+","+str(time)+"\n"
    f.write(strd)
    f.close()

def str_td(td):
    s = str(td).split(", ", 1)
    a = s[-1]
    if a[1] == ':':
        a = "0" + a
    s2 = s[:-1] + [a]
    return ", ".join(s2)

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

def interpData(lat1 ,lon1 ,lat2 ,lon2, InterLoop):
    lat1 ,lon1 ,lat2 ,lon2 = coverStrtoFloat(lat1 ,lon1 ,lat2 ,lon2)
    
    times = [0, InterLoop+1]
    lat = [lat1, lat2]
    lon = [lon1, lon2]
    t = np.arange(0, times[len(times)-1]+1)

    latint = np.interp(t, times, lat)
    lonint = np.interp(t, times, lon)

    array = []
    for i in range(len(latint)):
        # if i != 0 or i != len(latint):
        #     array.append([latint[i],lonint[i]])
        array.append([latint[i],lonint[i]])
    return array

def coverStrtoFloat(lat1 ,lon1 ,lat2 ,lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    return lat1 ,lon1 ,lat2 ,lon2

def NewFolder():
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

def ReadFileGpsPath(Path):
    GPS = []
    with open(Path) as f:
        lines = [line.rstrip() for line in f]
    for i in lines:
        GPS.append(i.split(", "))
    return GPS
