from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog

import cv2,glob

from natsort.natsort import natsorted
import heatmap,xanvas
import TEMP_DIRS_POHOTO as READFILE
import PredictFRCNN
import PredictThread
from config import Config

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
            self.AIOBJ = PredictFRCNN.AI(self)
            self.FileImg = []
            self.ImgShowNow = 0
            self.distanceAll = 0.0
            self.FileTextIndex = 0
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
            self.pushButton.setGeometry(QtCore.QRect(0, 90, 271, 41))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton.setFont(font)
            self.pushButton.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton.setLayoutDirection(QtCore.Qt.LeftToRight)
            self.pushButton.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton.setObjectName("pushButton")
            self.pushButton_2 = QtWidgets.QPushButton(self.widget)
            self.pushButton_2.setGeometry(QtCore.QRect(0, 130, 271, 41))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton_2.setFont(font)
            self.pushButton_2.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton_2.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton_2.setObjectName("pushButton_2")
            self.progressBar = QtWidgets.QProgressBar(self.widget)
            self.progressBar.setGeometry(QtCore.QRect(10, 550, 251, 23))
            font = QtGui.QFont()
            font.setPointSize(10)
            self.progressBar.setFont(font)
            self.progressBar.setStyleSheet("color: rgb(255, 255, 255);")
            self.progressBar.setLocale(QtCore.QLocale(QtCore.QLocale.Afar, QtCore.QLocale.Ethiopia))
            self.progressBar.setProperty("value", 0)
            self.progressBar.setObjectName("progressBar")
            self.label_2 = QtWidgets.QLabel(self.widget)
            self.label_2.setGeometry(QtCore.QRect(30, 20, 51, 51))
            self.label_2.setAutoFillBackground(False)
            self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);border-radius: 25px")
            self.label_2.setText("")
            self.label_2.setObjectName("label_2")
            self.label_3 = QtWidgets.QLabel(self.widget)
            self.label_3.setGeometry(QtCore.QRect(100, 30, 131, 16))
            font = QtGui.QFont()
            font.setPointSize(14)
            self.label_3.setFont(font)
            self.label_3.setObjectName("label_3")
            self.label_4 = QtWidgets.QLabel(self.widget)
            self.label_4.setGeometry(QtCore.QRect(100, 50, 131, 16))
            font = QtGui.QFont()
            font.setPointSize(8)
            self.label_4.setFont(font)
            self.label_4.setObjectName("label_4")
            self.pushButton_3 = QtWidgets.QPushButton(self.widget)
            self.pushButton_3.setGeometry(QtCore.QRect(0, 170, 271, 41))
            font = QtGui.QFont()
            font.setPointSize(12)
            self.pushButton_3.setFont(font)
            self.pushButton_3.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton_3.setStyleSheet("background-color: rgb(255, 171, 37);")
            self.pushButton_3.setObjectName("pushButton_3")
            self.pushButton_4 = QtWidgets.QPushButton(self.widget)
            self.pushButton_4.setObjectName(u"pushButton_4")
            self.pushButton_4.setGeometry(QtCore.QRect(0, 210, 271, 41))
            self.pushButton_4.setFont(font)
            self.pushButton_4.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
            self.pushButton_4.setStyleSheet(u"background-color: rgb(255, 171, 37);")
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
            self.label_13.setGeometry(QtCore.QRect(20, 56, 21, 16))
            self.label_13.setObjectName("label_13")
            self.lineEdit_4 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_4.setGeometry(QtCore.QRect(60, 56, 121, 20))
            self.lineEdit_4.setText("N/A")
            self.lineEdit_4.setObjectName("lineEdit_4")
            self.line_3 = QtWidgets.QFrame(self.groupBox_4)
            self.line_3.setGeometry(QtCore.QRect(20, 36, 161, 16))
            self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
            self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line_3.setObjectName("line_3")
            self.label_11 = QtWidgets.QLabel(self.groupBox_4)
            self.label_11.setGeometry(QtCore.QRect(20, 80, 31, 16))
            self.label_11.setObjectName("label_11")
            self.lineEdit_3 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_3.setGeometry(QtCore.QRect(60, 80, 121, 20))
            self.lineEdit_3.setText("N/A")
            self.lineEdit_3.setObjectName("lineEdit_3")
            self.label_15 = QtWidgets.QLabel(self.groupBox_4)
            self.label_15.setGeometry(QtCore.QRect(75, 124, 51, 17))
            self.label_15.setAlignment(QtCore.Qt.AlignCenter)
            self.label_15.setObjectName("label_15")
            self.lineEdit_5 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_5.setGeometry(QtCore.QRect(60, 174, 121, 20))
            self.lineEdit_5.setText("N/A")
            self.lineEdit_5.setObjectName("lineEdit_5")
            self.label_16 = QtWidgets.QLabel(self.groupBox_4)
            self.label_16.setGeometry(QtCore.QRect(20, 150, 31, 16))
            self.label_16.setObjectName("label_16")
            self.line_4 = QtWidgets.QFrame(self.groupBox_4)
            self.line_4.setGeometry(QtCore.QRect(20, 126, 161, 16))
            self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
            self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line_4.setObjectName("line_4")
            self.lineEdit_6 = QtWidgets.QLabel(self.groupBox_4)
            self.lineEdit_6.setGeometry(QtCore.QRect(60, 150, 121, 20))
            self.lineEdit_6.setText("N/A")
            self.lineEdit_6.setObjectName("lineEdit_6")
            self.label_12 = QtWidgets.QLabel(self.groupBox_4)
            self.label_12.setGeometry(QtCore.QRect(20, 174, 31, 16))
            self.label_12.setObjectName("label_12")
            self.line_5 = QtWidgets.QFrame(self.groupBox_4)
            self.line_5.setGeometry(QtCore.QRect(20, 30, 161, 16))
            self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
            self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line_5.setObjectName("line_5")
            self.label_17 = QtWidgets.QLabel(self.groupBox_4)
            self.label_17.setGeometry(QtCore.QRect(85, 26, 31, 20))
            self.label_17.setAlignment(QtCore.Qt.AlignCenter)
            self.label_17.setObjectName("label_17")
            self.line_4.raise_()
            self.label_13.raise_()
            # self.lineEdit_4.raise_()
            self.label_11.raise_()
            # self.lineEdit_3.raise_()
            self.label_15.raise_()
            self.label_16.raise_()
            # self.lineEdit_6.raise_()
            self.label_12.raise_()
            # self.lineEdit_5.raise_()
            self.line_5.raise_()
            self.label_17.raise_()
            MainWindow.setCentralWidget(self.centralwidget)

            self.canvas = xanvas.Canvas(self.label_5)
            self.canvas_1 = heatmap.heatmap(self.label_6)

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "File"))
        self.pushButton_2.setText(_translate("MainWindow", "Save"))
        self.label_3.setText(_translate("MainWindow", "Road Detector"))
        self.label_4.setText(_translate("MainWindow", "ver 1.0"))
        self.pushButton_3.setText(_translate("MainWindow", "Detect"))
        self.groupBox.setTitle(_translate("MainWindow", "Image"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Data"))
        self.comboBox.setItemText(0, _translate("MainWindow", "P"))
        self.comboBox.setItemText(1, _translate("MainWindow", "C"))
        self.comboBox.setItemText(2, _translate("MainWindow", "R"))
        self.comboBox.setItemText(3, _translate("MainWindow", "ALL"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Detect"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Deteil"))
        self.label_13.setText(_translate("MainWindow", "Lat:"))
        self.label_11.setText(_translate("MainWindow", "Long:"))
        self.pushButton_4.setText(_translate("MainWindow", "GPS File"))
        self.label_13.setText(_translate("MainWindow", "Lat:"))
        self.label_11.setText(_translate("MainWindow", "Long:"))
        self.label_15.setText(_translate("MainWindow", "Distance"))
        self.label_16.setText(_translate("MainWindow", "length:"))
        self.label_12.setText(_translate("MainWindow", "Km/h"))
        self.label_17.setText(_translate("MainWindow", "GPS"))

        self.pushButton.clicked.connect(self.DIR_FILE_PATH)
        self.pushButton_2.clicked.connect(self.DIR_FILE_SAVE)
        self.pushButton_3.clicked.connect(self.fuck)
        self.pushButton_4.clicked.connect(self.DIR_FILE_PATH_GPS)

 
    def LOADING(self,intData):
        self.progressBar.setProperty("value",intData)


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
        self.lineEdit_6.setText("{:.2f}".format(self.distanceAll))
        self.lineEdit_5.setText("{:.2f}".format(float(km_h)))
        self.lineEdit_6.show()
        self.lineEdit_5.show()

    def GPS_NEXT_LIST(self,lat,long):
        self.lineEdit_4.setText(lat)
        self.lineEdit_3.setText(long)
        self.lineEdit_4.show()
        self.lineEdit_3.show()
  
    def DIR_FILE_PATH_GPS(self):
        resopnse = QFileDialog.getOpenFileName()
        self._file_path = resopnse[0]
        self.lineEdit_4.setText(resopnse[0])
        self.lineEdit_3.setText(resopnse[0])
        
        

    def DIR_FILE_PATH(self):
        resopnse = QFileDialog.getOpenFileName()
        self._file_path = resopnse[0]
        self.SET_IMG_OR("first")

    def DIR_FILE_SAVE(self):
        thr1 = READFILE.Counter(self._file_path,30,self)
        thr1.start()

    def fuck(self):
        self.SET_IMG_OR("getfile")
        threadForRun = PredictThread.RuningAI(self.AIOBJ)
        threadForRun.start()
        self.RuningFalse(False)

    def RuningFalse(self,bool):
        self.pushButton.setEnabled(bool)
        self.pushButton_2.setEnabled(bool)
        self.pushButton_3.setEnabled(bool)
        self.pushButton_4.setEnabled(bool)
        self.comboBox.setEnabled(bool)

        
        
    def SET_IMG_DETECT(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.label_7.setPixmap(pixmap_image)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setScaledContents(True)
        self.label_7.setMinimumSize(1,1)
        self.label_7.show()

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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.label.setPixmap(pixmap_image)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setMinimumSize(1,1)
        self.label.show()

        



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
