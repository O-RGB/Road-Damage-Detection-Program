import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog, QLabel, QPushButton

import cv2,glob
from natsort.natsort import natsorted
import numpy as np
from source.graph import heatmap,xanvas
from source.create import TEMP_DIRS_POHOTO as READFILE
from source.frcnn import PredictFRCNN
from source.frcnn import RoiPoolingConv
from source.thread import PredictThread
from source.frcnn.config  import Config
from source.create import  creadPDF

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
            self.AIOBJ = PredictFRCNN.AI(self)
            self.FileImg = []   
            self.ImgShowNow = 0
            self.distanceAll = 0.0
            self.FileTextIndex = 0
            self.real_Position = []
            self.Plothole = []
            self.Crack = []
            self.Repair = []
            self.ready = False
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(1091, 581)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.frame = QtWidgets.QFrame(self.centralwidget)
            self.frame.setGeometry(QtCore.QRect(0, 0, 1101, 591))
            self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.frame.setObjectName("frame")
            self.widget = QtWidgets.QWidget(self.frame)
            self.widget.setGeometry(QtCore.QRect(0, 0, 271, 601))
            self.widget.setMouseTracking(False)
            self.widget.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.widget.setAutoFillBackground(False)
            self.widget.setStyleSheet("background-color: rgb(100, 100, 100);")
            self.widget.setObjectName("widget")
            self.pushButton = QtWidgets.QPushButton(self.widget)
            self.pushButton.setGeometry(QtCore.QRect(230, 100, 31, 21))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton.setFont(font)
            self.pushButton.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton.setLayoutDirection(QtCore.Qt.LeftToRight)
            self.pushButton.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton.setObjectName("pushButton")
            self.pushButton_2 = QtWidgets.QPushButton(self.widget)
            self.pushButton_2.setGeometry(QtCore.QRect(0, 230, 271, 41))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton_2.setFont(font)
            self.pushButton_2.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton_2.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton_2.setObjectName("pushButton_2")
            self.label_2 = QtWidgets.QLabel(self.widget)
            self.label_2.setGeometry(QtCore.QRect(30, 20, 51, 51))
            self.label_2.setAutoFillBackground(False)
            self.label_2.setStyleSheet("border-radius: 25px")
            self.label_2.setText("")
            self.label_2.setObjectName("label_2")
            self.label_3 = QtWidgets.QLabel(self.widget)
            self.label_3.setGeometry(QtCore.QRect(100, 30, 131, 16))
            self.label_3.setStyleSheet("color: rgb(255, 255, 255);")
            font = QtGui.QFont()
            font.setPointSize(14)
            self.label_3.setFont(font)
            self.label_3.setObjectName("label_3")
            self.label_4 = QtWidgets.QLabel(self.widget)
            self.label_4.setGeometry(QtCore.QRect(100, 50, 131, 16))
            self.label_4.setStyleSheet("color: rgb(255, 255, 255);")
            font = QtGui.QFont()
            font.setPointSize(8)
            self.label_4.setFont(font)
            self.label_4.setObjectName("label_4")
            self.pushButton_3 = QtWidgets.QPushButton(self.widget)
            self.pushButton_3.setGeometry(QtCore.QRect(0, 190, 271, 41))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton_3.setFont(font)
            self.pushButton_3.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton_3.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton_3.setObjectName("pushButton_3")
            self.pushButton_4 = QtWidgets.QPushButton(self.widget)
            self.pushButton_4.setGeometry(QtCore.QRect(230, 150, 31, 21))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton_4.setFont(font)
            self.pushButton_4.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton_4.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton_4.setObjectName("pushButton_4")
            self.lineEdit = QtWidgets.QLineEdit(self.widget)
            self.lineEdit.setGeometry(QtCore.QRect(10, 100, 211, 21))
            self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.lineEdit.setObjectName("lineEdit")
            self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
            self.lineEdit_2.setGeometry(QtCore.QRect(10, 150, 211, 21))
            self.lineEdit_2.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.lineEdit_2.setObjectName("lineEdit_2")
            self.label_8 = QtWidgets.QLabel(self.widget)
            self.label_8.setGeometry(QtCore.QRect(10, 80, 51, 16))
            self.label_8.setStyleSheet("color: rgb(255, 255, 255);")
            self.label_8.setObjectName("label_8")
            self.label_9 = QtWidgets.QLabel(self.widget)
            self.label_9.setGeometry(QtCore.QRect(10, 130, 51, 16))
            self.label_9.setStyleSheet("color: rgb(255, 255, 255);")
            self.label_9.setObjectName("label_9")
            self.pushButton.raise_()
            self.pushButton_2.raise_()
            self.label_3.raise_()
            self.label_4.raise_()
            self.pushButton_3.raise_()
            self.pushButton_4.raise_()
            self.lineEdit.raise_()
            self.label_2.raise_()
            self.lineEdit_2.raise_()
            self.label_8.raise_()
            self.label_9.raise_()
            self.frame_3 = QtWidgets.QFrame(self.frame)
            self.frame_3.setGeometry(QtCore.QRect(270, -10, 831, 631))
            self.frame_3.setStyleSheet("background-color: rgb(255, 255, 255);\n""border-left-color: rgb(225, 225, 225);")
            self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
            self.frame_3.setObjectName("frame_3")
            self.groupBox = QtWidgets.QGroupBox(self.frame_3)
            self.groupBox.setGeometry(QtCore.QRect(20, 20, 391, 331))
            self.groupBox.setObjectName("groupBox")
            self.label = QtWidgets.QLabel(self.groupBox)
            self.label.setGeometry(QtCore.QRect(10, 20, 371, 301))
            self.label.setStyleSheet("background-color: rgb(225, 225, 225);")
            self.label.setText("")
            self.label.setObjectName("label")
            self.groupBox_2 = QtWidgets.QGroupBox(self.frame_3)
            self.groupBox_2.setGeometry(QtCore.QRect(430, 360, 371, 221))
            self.groupBox_2.setObjectName("groupBox_2")
            self.label_5 = QtWidgets.QLabel(self.groupBox_2)
            self.label_5.setGeometry(QtCore.QRect(10, 30, 351, 81))
            self.label_5.setStyleSheet("\n""background-color: rgb(225, 225, 225);")
            self.label_5.setText("")
            self.label_5.setObjectName("label_5")
            self.label_6 = QtWidgets.QLabel(self.groupBox_2)
            self.label_6.setGeometry(QtCore.QRect(10, 120, 351, 81))
            self.label_6.setStyleSheet("\n""background-color: rgb(225, 225, 225);")
            self.label_6.setText("")
            self.label_6.setObjectName("label_6")
            self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
            self.comboBox.setGeometry(QtCore.QRect(290, 0, 71, 22))
            self.comboBox.setAcceptDrops(False)
            self.comboBox.setEditable(True)
            self.comboBox.setObjectName("comboBox")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.comboBox.addItem("")
            self.groupBox_3 = QtWidgets.QGroupBox(self.frame_3)
            self.groupBox_3.setGeometry(QtCore.QRect(430, 20, 371, 331))
            self.groupBox_3.setObjectName("groupBox_3")
            self.label_7 = QtWidgets.QLabel(self.groupBox_3)
            self.label_7.setGeometry(QtCore.QRect(10, 20, 351, 301))
            self.label_7.setStyleSheet("background-color: rgb(225, 225, 225);")
            self.label_7.setText("")
            self.label_7.setObjectName("label_7")
            self.groupBox_4 = QtWidgets.QGroupBox(self.frame_3)
            self.groupBox_4.setGeometry(QtCore.QRect(20, 360, 391, 221))
            self.groupBox_4.setObjectName("groupBox_4")
            self.label_13 = QtWidgets.QLabel(self.groupBox_4)
            self.label_13.setGeometry(QtCore.QRect(20, 60, 21, 16))
            self.label_13.setObjectName("label_13")
            self.lineEdit_4 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_4.setGeometry(QtCore.QRect(60, 60, 291, 20))
            self.lineEdit_4.setText("")
            self.lineEdit_4.setObjectName("lineEdit_4")
            self.label_11 = QtWidgets.QLabel(self.groupBox_4)
            self.label_11.setGeometry(QtCore.QRect(20, 84, 31, 16))
            self.label_11.setObjectName("label_11")
            self.lineEdit_3 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_3.setGeometry(QtCore.QRect(60, 84, 291, 20))
            self.lineEdit_3.setText("")
            self.lineEdit_3.setObjectName("lineEdit_3")
            self.label_15 = QtWidgets.QLabel(self.groupBox_4)
            self.label_15.setGeometry(QtCore.QRect(170, 123, 51, 17))
            self.label_15.setAlignment(QtCore.Qt.AlignCenter)
            self.label_15.setObjectName("label_15")
            self.lineEdit_5 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_5.setGeometry(QtCore.QRect(60, 184, 291, 20))
            self.lineEdit_5.setText("")
            self.lineEdit_5.setObjectName("lineEdit_5")
            self.label_16 = QtWidgets.QLabel(self.groupBox_4)
            self.label_16.setGeometry(QtCore.QRect(20, 160, 31, 16))
            self.label_16.setObjectName("label_16")
            self.line_4 = QtWidgets.QFrame(self.groupBox_4)
            self.line_4.setGeometry(QtCore.QRect(20, 126, 331, 16))
            self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
            self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line_4.setObjectName("line_4")
            self.lineEdit_6 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_6.setGeometry(QtCore.QRect(60, 160, 291, 20))
            self.lineEdit_6.setText("")
            self.lineEdit_6.setObjectName("lineEdit_6")
            self.label_12 = QtWidgets.QLabel(self.groupBox_4)
            self.label_12.setGeometry(QtCore.QRect(20, 184, 31, 16))
            self.label_12.setObjectName("label_12")
            self.line_5 = QtWidgets.QFrame(self.groupBox_4)
            self.line_5.setGeometry(QtCore.QRect(20, 30, 331, 16))
            self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
            self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line_5.setObjectName("line_5")
            self.label_17 = QtWidgets.QLabel(self.groupBox_4)
            self.label_17.setGeometry(QtCore.QRect(180, 26, 31, 20))
            self.label_17.setAlignment(QtCore.Qt.AlignCenter)
            self.label_17.setObjectName("label_17")
            self.line_4.raise_()
            self.label_13.raise_()
            self.lineEdit_4.raise_()
            self.label_11.raise_()
            self.lineEdit_3.raise_()
            self.label_15.raise_()
            self.label_16.raise_()
            self.lineEdit_6.raise_()
            self.label_12.raise_()
            self.lineEdit_5.raise_()
            self.line_5.raise_()
            self.label_17.raise_()

                

            

            MainWindow.setCentralWidget(self.centralwidget)

            

            self.canvas = xanvas.Canvas(self.label_5)
            self.canvas_1 = heatmap.heatmap(self.label_6)

            

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Road Detector"))
        self.pushButton.setText(_translate("MainWindow", "..."))
        self.pushButton_2.setText(_translate("MainWindow", "PDF SAVE"))
        self.label_3.setText(_translate("MainWindow", "Road Detector"))
        self.label_4.setText(_translate("MainWindow", "ver 1.0.5"))
        self.pushButton_3.setText(_translate("MainWindow", "START"))
        self.pushButton_4.setText(_translate("MainWindow", "..."))
        self.label_8.setText(_translate("MainWindow", "Video File"))
        self.label_9.setText(_translate("MainWindow", "GPS File"))
        self.groupBox.setTitle(_translate("MainWindow", "Image"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Data"))
        self.comboBox.setItemText(0, _translate("MainWindow", "ALL"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Plothole"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Crack"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Repair"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Detect"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Deteil"))
        self.label_13.setText(_translate("MainWindow", "Lat:"))
        self.label_11.setText(_translate("MainWindow", "Long:"))
        self.label_15.setText(_translate("MainWindow", "Distance"))
        self.label_16.setText(_translate("MainWindow", "length:"))
        self.label_12.setText(_translate("MainWindow", "Km/h"))
        self.label_17.setText(_translate("MainWindow", "GPS"))


        self.setImg_label(cv2.imread("source/gui/iconB.png"),self.label_2)

        self.pushButton.clicked.connect(self.DIR_FILE_PATH)
        self.pushButton_4.clicked.connect(self.DIR_FILE_PATH_GPS)
        self.pushButton_3.clicked.connect(self.START)
        self.pushButton_2.clicked.connect(self.PDF_SAVE)
        
        
        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.comboBox.setEnabled(False)


        self.comboBox.activated[str].connect(self.onChanged) 




    def DIR_FILE_PATH(self):
        try:
            resopnse = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video File (*.mp4 *.avi )")
            if resopnse:
                self._file_path = resopnse[0]
                self.lineEdit.setText(resopnse[0])
                self.pushButton_4.setEnabled(True)
                self.SET_IMG_OR("first")
        except:
            print()

    def DIR_FILE_PATH_GPS(self):
        try:
            resopnse = QFileDialog.getOpenFileName(None, "Select Text file", "", "Text file (*.txt)")
            if resopnse:
                self._file_GPS = resopnse[0]
                self.lineEdit_2.setText(resopnse[0])
                self.RuningFalse(True)
            
        except:
            print()
       
    def START(self):
        thr1 = READFILE.Counter(self._file_path,30,self)
        thr1.start()
        self.RuningFalse(False)

    def START_FOR_AI(self):
        self.SET_IMG_OR("getfile")
        threadForRun = PredictThread.RuningAI(self.AIOBJ)
        threadForRun.start()
        self.RuningFalse(False)
        self.ready = True

    def PDF_SAVE(self):
        # try:
            response = QFileDialog.getSaveFileName(None, "Save PDF file", "Report.pdf", "Adobe PDF Files (*.pdf)")
            if response:
                self.RuningFalse(True)
                pdfOBJ = creadPDF.CreadPDF(self.real_Position,self.Plothole,self.Crack,self.Repair,response[0],self.distanceAll)
                
        # except:
        #     print()
       


    # //////////////////////////////////////
    def onChanged(self,text):
        if self.ready == True:
            if text == "ALL":
                self.canvas.SetArrayPlotUpdate(self.real_Position)
            elif text == "Plothole":
                self.canvas.SetArrayPlotUpdate(self.Plothole)
            elif text == "Crack":
                self.canvas.SetArrayPlotUpdate(self.Crack)
            elif text == "Repair":
                self.canvas.SetArrayPlotUpdate(self.Repair)
                
    def READ_FILE_TEMP_TEXT(self):
        self.fileGPS  = [] 
        with open('temp.txt') as f:
            lines = [line.rstrip() for line in f]
        for i in lines:
            self.fileGPS.append(i.split(","))
    
    def NEXT_FILE_TEMP(self):
        self.GPS_NEXT_LIST(self.fileGPS[self.FileTextIndex][1],self.fileGPS[self.FileTextIndex][2])
        self.GPS_DISTANCE(self.fileGPS[self.FileTextIndex][4],self.fileGPS[self.FileTextIndex][3])
        self.FileTextIndex = self.FileTextIndex + 1 

    def GPS_DISTANCE(self,distance,km_h):
        self.distanceAll = float(self.distanceAll) + float(distance) 
        self.create_word("{:.2f}".format(self.distanceAll),self.lineEdit_6)
        self.create_word("{:.2f}".format(float(km_h)),self.lineEdit_5)

    def GPS_NEXT_LIST(self,lat,long):
        self.create_word(lat,self.lineEdit_4)
        self.create_word(long,self.lineEdit_3)

    def create_word(self,str,label):
        blank_image = np.zeros((30,400,3), np.uint8)
        blank_image.fill(255) 
        cv2.putText(blank_image, str, (17,17), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 0))
        self.setImg_label(blank_image,label)

    def setImg_label(self,img,label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        label.setPixmap(pixmap_image)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setScaledContents(True)
        label.setMinimumSize(1,1)
        label.show()

    def RuningFalse(self,bool):
        self.pushButton.setEnabled(bool)
        self.pushButton_2.setEnabled(bool)
        self.pushButton_3.setEnabled(bool)
        self.pushButton_4.setEnabled(bool)
        self.comboBox.setEnabled(bool)
        
    def SET_IMG_DETECT(self,img):
        self.setImg_label(img,self.label_7)

    def SET_IMG_OR(self,mode="READALL"):
        if mode == "first":
            cap = cv2.VideoCapture(self._file_path,0) 
            cap.set(1,0)
            eat, img = cap.read()
        elif mode == "getfile":
            for img in natsorted(glob.glob("temp/*.jpg")):
                self.FileImg.append(img)
            img = cv2.imread(self.FileImg[self.ImgShowNow-1])
        else:
            self.ImgShowNow = self.ImgShowNow + 1
            img = cv2.imread(self.FileImg[self.ImgShowNow-1])
            self.NEXT_FILE_TEMP()
        self.setImg_label(img,self.label)

    def Set_Array_report(self,real_Position,Plothole,Crack,Repair):
        self.real_Position = real_Position
        self.Plothole = Plothole
        self.Crack = Crack
        self.Repair = Repair

    
        
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app_icon = QtGui.QIcon()
    app_icon.addFile('source/gui/iconW.png', QtCore.QSize(24,24))
    app.setWindowIcon(app_icon)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
