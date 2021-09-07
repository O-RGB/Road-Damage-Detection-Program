#########################################################
from xml.etree import ElementTree
import os
import csv
import cv2
import os.path
#########################################################
path = "annotations/"
files = os.listdir(path)

#train = int(len(files)*80)/100 #train 80%
#test = len(files)-train #test 20%

listDataset = []
listPart = []
#########################################################
for i in range(len(files)):
    tree = ElementTree.parse(path+'/'+files[i])
    root = tree.getroot()
    filenameDataset = root.findall(".//filename")[0].text
    countObj = root.findall(".//object")
    
    for j in range(len(countObj)):
        listIndexDataset =     [countObj[j].findall(".//xmin")[0].text,
                                countObj[j].findall(".//ymin")[0].text,
                                countObj[j].findall(".//xmax")[0].text,
                                countObj[j].findall(".//ymax")[0].text]
        classDataset =          countObj[j].findall(".//name")[0].text
        part = "images/"+filenameDataset
        
        listDataset.append(part
              +","+listIndexDataset[0]
              +","+listIndexDataset[1]
              +","+listIndexDataset[2]
              +","+listIndexDataset[3]
              +","+classDataset)
        listPart.append(part)
#########################################################

with open('annotate.txt', 'w') as f:
    for line in listDataset:
        path = line.split(",")
        if os.path.exists(path[0]):
            f.write(line)
            f.write('\n')
#########################################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

image_path, xmin, ymin, xmax, ymax, clas= listDataset[0].split(",")
X_MIN, Y_MIN = int(xmin), int(ymin)
X_MAX, Y_MAX = int(xmax), int(ymax)
WIDTH = X_MAX - X_MIN
HEIGHT = Y_MAX - Y_MIN

im = Image.open(image_path)
fig, ax = plt.subplots()
ax.imshow(im)
rect = patches.Rectangle((int(X_MIN), int(Y_MIN)), WIDTH, HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
