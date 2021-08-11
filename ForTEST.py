
# from keras.backend import print_tensor
# import matplotlib.pyplot as plt
# import math

# videoSize = [1920,768]
# fps = 30

# arraytest = [
#     [[]],
#     [[1094, 403, 1267, 489]],
#     [[1065, 489, 1411, 604]],
#     [[863, 633, 1641, 863], [950, 604, 1180, 835]],
#     [[]],
#     [[]]
# ]

# arraytestx = [[], [], [], [], [], [], [], [], [], [[1094, 403, 1267, 489]], [[1065, 489, 1411, 604]], [[863, 633, 1641, 863], [950, 604, 1180, 835]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 
# [], [], [], [], [], [], [[748, 662, 1065, 835]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [[1641, 604, 1785, 691]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [[1151, 431, 1324, 518], [1353, 431, 1526, 518]], [[1209, 518, 1382, 604]], [[1497, 604, 1814, 777], [1267, 604, 1497, 748]], [[1353, 748, 1699, 921]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [[431, 777, 604, 863]], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [[604, 662, 979, 835]], [], [], 
# [], [], [], [], [], [], [], [], [], [], [], []]

# arraytestx=[[], [], [], [], [], [], [], [], [],[[1641, 604, 1785, 691]]]
# array = []
# ix,jx = 0,0
# for i in arraytestx:
#     for j in i:
#         array.append([ix,int((j[0]+j[2])/2)])
#     ix = ix + 5

# data =  np.array(array)
# x, y = data.T

# plt.ion() ## Note this correction

# plt.ylim(0, 1920)

# ax = plt.gca()
# ax.invert_yaxis()



# for i in range(50):
#     x = np.append(x, i+1000)
#     y = np.append(y, i+500)
#     plt.scatter(x,y)
#     plt.show()
#     plt.pause(0.0001) #Note this correction
    





# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from scipy.ndimage.filters import gaussian_filter


# def myplot(x, y, s, bins=[1920,400]):
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     heatmap = gaussian_filter(heatmap, sigma=s)

#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     return heatmap.T, extent



# fig = plt.figure()
# axs = fig.add_subplot(111)

# x = []
# y = []

# img, extent = myplot(x, y, 64)
# axs.imshow(img,  origin='lower', cmap=cm.jet)
# axs.invert_yaxis()


# plt.show()


# arrayList = [[]]

# array = []  
# ix = 0


# for i in arrayList:
#     print(len(i))
#     if(len(i) == 0):
#         i = [[0, 0, 0, 0]]
#     for j in i:
#         array.append([ix,int((j[0]+j[2])/2)])

# print(array)


# from math import radians, cos, sin, asin, sqrt
# def haversine(lat1 ,lon1 ,lat2 ,lon2):
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a)) 
#     # Radius of earth in kilometers is 6371
#     km = 6371* c
#     return km*3600

# print(int(haversine(103.2483844289111,16.245199907982073,103.24866871371364,16.245024789616384)))

# import numpy as np

# times = [0, 2]
# lat = [16.24513778298438, 16.245176409825287]
# lon = [103.24850172682649, 103.24836493409047]
# t = np.arange(0, times[len(times)-1]+1)

# latint = np.interp(t, times, lat)
# lonint = np.interp(t, times, lon)

# for i in range(len(latint)):
#     print(latint[i],lonint[i])
#     print()

    

# array =[]
# with open('C:/Users/okoza/Desktop/gps.txt') as f:
#     lines = [line.rstrip() for line in f]

# for i in lines:
#     array.append(i.split(", "))

# for i in array:
#     print(i)

# fileGPS  = [] 
# with open('temp.txt') as f:
#     lines = [line.rstrip() for line in f]
# for i in lines:
#     fileGPS.append(i.split(","))

# for i in range(len(fileGPS)):
#     print(fileGPS[i])
    

import cv2
import numpy as np
blank_image = np.zeros((20,50,3), np.uint8)
blank_image.fill(255) # or img[:] = 255


cv2.imshow("ff",blank_image)
# cv2.putText(image,"Hello World!!!", (x,y), 0, 2, 255)