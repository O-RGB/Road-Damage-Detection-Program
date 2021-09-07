
import cv2
from fpdf import FPDF
from matplotlib import pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm

from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import os
import base64
import shutil

class CreadPDF:
    def __init__(self,real_Position,Plothole,Crack,Repair,savepath,distanceAll):

        self.distanceAll = distanceAll
        self.real_Position = real_Position
        self.plot(self.real_Position,"real","orange")
        self.Plothole = Plothole
        self.plot(self.Plothole,"plothole","red")
        self.Crack = Crack
        self.plot(self.Crack,"crack","blue")

        self.savepath = savepath

        self.Repair = Repair
        self.plot(self.Repair,"repair","green")
        self.plotHeadMap(self.real_Position,"HeadMap")

        self.countPlothole = self.countData(self.Plothole)
        self.countCrack = self.countData(self.Crack)
        self.countRepair = self.countData(self.Repair)
        self.bar([self.countPlothole,self.countCrack,self.countRepair])

        self.FileForWeb = ""
        
        self.cread()


    def cread(self):
        print("cread")
        GPS = self.ReadFileGpsPath()

        self.pdf = FPDF()
        self.pdf.add_page()
        self.conter = "                                                                      "
        self.tab = "          "
        self.tabBold = "       "
        
        self.setfont(20,"ระบบตรวจจับความเสียหายถนน",10, Bold = True)
        self.setfont(16,self.tab + "ตรวจจับตั้งแต่ตำแหน่งที่ "+GPS[0][0]+","+GPS[0][1]+" ถึง "+GPS[len(GPS)-1][0]+","+GPS[len(GPS)-1][1]+" ",7)
        self.setfont(16,"ด้วยเวลา "+GPS[len(GPS)-1][5]+" นาที ระยะทาง "+str(int(self.distanceAll))+" เมตร โดยพบหลุม "+str(self.countPlothole)+" ครั้ง ถนนแตก "+str(self.countCrack)+" ครั้ง",7)
        self.setfont(16,"ถนนซ่อมปะ "+str(self.countRepair)+" ครั้ง",10)

        self.pdf.image("real.jpg",     x=2, y=57,  w=200, h=50)
        self.pdf.image("plothole.jpg", x=2, y=112, w=200, h=50)
        self.pdf.image("crack.jpg",    x=2, y=167, w=200, h=50)
        self.pdf.image("repair.jpg",   x=2, y=222, w=200, h=50)

        self.setfont(20,"กราฟความเสียหาย",8, Bold = True) 
        self.setfont(20, self.tabBold + "ตำแหน่งจุดที่พบความเสียหายทั้งหมด",49, Bold = True) 
        self.setfont(14,self.conter + "ความเสียหายทั้งหมดที่ตวรจพบ",7) 

        self.setfont(20, self.tabBold + "ตำแหน่งหลุมบนถนน",47, Bold = True) 
        self.setfont(14,self.conter + "         หลุมที่ตวรจพบ",7) 

        self.setfont(20, self.tabBold + "ตำแหน่งถนนแตก",48, Bold = True) 
        self.setfont(14,self.conter + "      ถนนแตกที่ตวรจพบ",7) 

        self.setfont(20, self.tabBold + "ตำแหน่งถนนซ่อมปะ",48, Bold = True) 
        self.setfont(14,self.conter + "      ถนนซ่อมป่ะที่ตวรจพบ",10) 

        self.setfont(20, self.tabBold + "แสดงความเสียหายด้วย Heat Map",60, Bold = True) 
        

        self.pdf.image("HeadMap.jpg",   x=2, y=18, w=200, h=50)
        self.pdf.image("bar.jpg",   x=2, y=78, w=200, h=50)

        self.setfont(20, self.tabBold + "จำนวนความเสียหาย",60, Bold = True)

        self.setfont(20, self.tabBold + "ตำแหน่งและภาพ",10, Bold = True)

        self.pdf.add_font('THSarabun_0', '', 'source/font/THSarabun_0.ttf', uni=True)
        self.pdf.set_font('THSarabun_0', '', 16) 

        epw = self.pdf.w - 2*self.pdf.l_margin
        self.col_width = epw/2
        self.th = self.pdf.font_size

        arrayAllfolder,arrayAllPhoto = self.ReadDetectOut()
        

        self.creadNameTabel()

        y = 153
        e = 40
        for i in range(len(arrayAllfolder)):
            if self.pdf.get_y() > 255:
                y = 23.5
                e = 39.5
            latlng = GPS[int(arrayAllfolder[i][0])][1]+", "+GPS[int(arrayAllfolder[i][0])][2]
            sc = arrayAllfolder[i][2]
            time = GPS[int(arrayAllfolder[i][0])][5]
            self.FileForWeb += latlng+", "+sc
            self.creadTable(latlng,sc,arrayAllPhoto[i],y,time)
            y=y+e
        
    
        self.pdf.output(self.savepath) 
        file = open((self.savepath[:-3]+"txt"), "w")
        print(self.savepath[:-3]+"txt")
        file.write(self.FileForWeb)
        file.close 


        os.remove("crack.jpg")
        os.remove("plothole.jpg")
        os.remove("repair.jpg")
        os.remove("real.jpg")
        os.remove("HeadMap.jpg") 
        os.remove("bar.jpg") 
        os.remove("temp.txt") 

        shutil.rmtree('tempVideo')
        shutil.rmtree('temp')
        shutil.rmtree('detectOut')


    def setfont(self,size,text,newline,Bold=False):
        if Bold == True:
            self.pdf.add_font('THSarabun Bold_0', '', 'source/font/THSarabun Bold_0.ttf', uni=True)
            self.pdf.set_font('THSarabun Bold_0', '', size) 
        else:
            self.pdf.add_font('THSarabun_0', '', 'source/font/THSarabun_0.ttf', uni=True)
            self.pdf.set_font('THSarabun_0', '', size) 
        
        self.pdf.cell(20, 10, u''+text)      
        self.pdf.ln(newline)

    def spliArray(self,arrayList,indexFps):
            array = []  
            ix,jx = 0,0

            for i in arrayList:
                for j in i:
                    array.append([ix,int((j[0]+j[2])/2)])
                ix = ix + indexFps 
            return array 

    def plot(self,array,name,color = "blue"):
        
        fig = plt.figure()    
        fig.set_size_inches(10,2.5)
        axes = plt.gca()
        axes.set_ylim([0,1920])
        axes.set_xlim([0,len(array)*5])

        x, y = np.array(self.spliArray(array,5)).T
        plt.scatter(x, y ,color=color)
        plt.gca().invert_yaxis()

        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)

        fig.savefig(name+'.jpg')
        return name

    def plotHeadMap(self,array,name):

        fig = plt.figure()    
        fig.set_size_inches(10,2.5)
        axes = plt.gca()
        x, y = np.array(self.spliArray(array,5)).T
        img, extent = self.myplot(x, y, 64)
        axes.imshow(img,  origin='lower', cmap=cm.jet)
        axes.invert_yaxis()

        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        fig.savefig(name+'.jpg')
    
    def myplot(self,x, y, s, bins=[2500,600]):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    def countData(self,array):
        count = 0
        for i in array:
            if len(i) != 0:
                for j in i:
                    count = count + 1
        return count

    def bar(self,performance):
        objects = ('Plothole', 'Crack', 'Repair')
        y_pos = np.arange(len(objects))
        fig = plt.figure()    
        fig.set_size_inches(10,2.5)

        plt.barh(y_pos, performance, align='center', alpha=1)
        plt.yticks(y_pos, objects)
        
        plt.rcParams.update({'font.size': 15})
        for index, value in enumerate(performance):
            plt.text(value, index, " "+str(value))

        fig.savefig('bar.jpg')

    def ReadDetectOut(self):
        arrayAllfolder = []
        arrayAllPhoto = []
        for idx,img_name in enumerate(natsorted(os.listdir("detectOut"))):
            temp = str(img_name).split(".")
            arrayAllPhoto.append(img_name)
            arrayAllfolder.append(temp)
        return arrayAllfolder,arrayAllPhoto

    def ReadFileGpsPath(self):
        fileGPS  = [] 
        with open('temp.txt') as f:
            lines = [line.rstrip() for line in f]
        for i in lines:
            fileGPS.append(i.split(","))
        return fileGPS
    
    def creadTable(self,pos,cs,Photo,y,time):
        if cs == "pothole": cs = u"หลุม"
        elif cs == "crack": cs = u""
        elif cs == "repai": cs = u"ถนนซ่อมปะ"

        img = cv2.imread("detectOut/"+Photo)
        img = cv2.resize(img, (130,130), interpolation = cv2.INTER_AREA)
        jpg_img = cv2.imencode('.jpg', img)
        self.FileForWeb += ", "+ base64.b64encode(jpg_img[1]).decode('utf-8')
        self.FileForWeb += "\n"

        temp = ["",""]
        spa = [temp,temp,["","ตำแหน่ง: "+pos],["","หมวด: "+cs],["","เวลา: "+time],temp,temp]
        i = 0
        
        if (self.pdf.get_y()) > 255:
            self.pdf.add_page()
            self.creadNameTabel()
        self.pdf.image("detectOut/"+Photo,   x=40, y=y, w=35, h=35)
        for row in spa:
            for datum in row:
                if i == (len(spa)*len(row))-1 or i == (len(spa)*len(row))-2:
                    self.pdf.cell(self.col_width, 1*self.th, str(datum), 'L,B,R',0,'L')
                else:
                    self.pdf.cell(self.col_width, 1*self.th, str(datum),'L,R', 0, 'L')
                i = i + 1
            self.pdf.ln(1*self.th)
            
    def creadNameTabel(self):
        data = [[u'รูปภาพ',u'รายละเอียด']]
        for row in data:
            for datum in row:
                self.pdf.cell(self.col_width, 2*self.th, str(datum), border=1)
            self.pdf.ln(2*self.th)
            break

