
import os
from natsort import natsorted



test_images_list = natsorted(os.listdir("Road-Damage-Detection-Program/temp/"))
array = []
for i in test_images_list:
    array.append("temp/"+str(i))

print(array)