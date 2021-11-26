import sys
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLabel, QSpinBox, QMessageBox
import fun1, fun2
import librosa 
from PyQt5.QtGui import QImage,QPixmap, QPalette, QColor
import cv2

Ui_MainWindow, QtBaseClass = uic.loadUiType("MainWindow.ui")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    audioData = None
    samplingRate = 0
    imgData = None
    FRAME_SIZE = int(1024)
    HOP_SIZE = int(FRAME_SIZE / 4)
    stftMod = None
    stftPhase = None
    fileDurationMs = 0
    startFrameTime = 0
    startFrameFreq = 0
    durationFrameTime = 0
    durationFrameFreq = 0
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
    
    def printError(self, textToDisplay):
        messageBox = QMessageBox(text = textToDisplay)
        messageBox.exec_()
    
    def fillAudioLabels(self, durationMsec,sampling):
        samplingText = str("Próbkowanie: ") + str(sampling) + " Hz"
        minutes = int(np.floor(durationMsec / 60000))
        durationMsec -= (minutes * 60000)
        sec = int(np.floor(durationMsec / 1000 ))
        durationMsec -= (sec * 1000)
        durationMsec = int(durationMsec)
        durationText = str("Czas: "  + str(minutes) + "m " + str(sec) + "s " + str(durationMsec) + "ms")
        
        durationLabel = self.findChild(QLabel,"audioDurationLabel")
        durationLabel.setText(durationText)
        samplingLabel = self.findChild(QLabel,"samplingLabel")
        samplingLabel.setText(samplingText)

    def fillImageLabels(self, fileName, width , height, minLum, maxLum):
        widthLabel = self.findChild(QLabel,"heightLabel")
        heightLabel = self.findChild(QLabel,"widthLabel")
        minLumLabel = self.findChild(QLabel,"minLumLabel")
        maxLumLabel = self.findChild(QLabel,"maxLumLabel")
        nameLabel = self.findChild(QLabel,"imgFileNameLabel")
        widthLabel.setText("W = " + str(width))
        heightLabel.setText("H = " + str(height))
        minLumLabel.setText("min. lum. = " + str(minLum))
        maxLumLabel.setText("max. lum. = " + str(maxLum))
        nameLabel.setText(fileName)

    def chooseAudioFileButtonPressed(self):
        fileDialog = QFileDialog(self,filter = "*.wav")
        fileDialog.exec_()
        selectedFiles = fileDialog.selectedFiles()
        if( len(selectedFiles) == 1):
            label = self.findChild(QLabel, "audioFileNameLabel")
            label.setText(selectedFiles[0])      
            audioDataTmp, durationTMP, samplingRate = fun1.readAudioFile(selectedFiles[0])
            self.fileDurationMs = durationTMP
            self.audioData = audioDataTmp
            self.fillAudioLabels(self.fileDurationMs, samplingRate)
            self.SAMPLING_RATE = samplingRate
            
    def chooseImageFileButtonPressed(self):
        fileDialog = QFileDialog(self,filter = "*.png")
        fileDialog.exec_()
        selectedFiles = fileDialog.selectedFiles()
        if( len(selectedFiles) == 1):
            absPathToFile = selectedFiles[0]
            self.imgData ,width , height, minLum, maxLum = fun1.readImg(absPathToFile)
            self.fillImageLabels(absPathToFile, width , height, minLum, maxLum)
    
    def setProperColorOfGuiTexts(self, axis):
        if(axis == "time"):
            label1 = self.findChild(QLabel,"calculatedFrameToStart_T")
            label2 = self.findChild(QLabel,"calculatedFramesToUse_T")
            areValuesCorrect = self.durationFrameTime + self.startFrameTime < self.stftModul.shape[1]
        elif(axis == "freq"):
            label1 = self.findChild(QLabel,"calculatedFrameToStart_F")
            label2 = self.findChild(QLabel,"calculatedFramesToUse_F")
            areValuesCorrect = self.durationFrameFreq + self.startFrameFreq < self.stftModul.shape[0]
        
        color = QColor(0x000000) if areValuesCorrect else QColor(0xff0000)
        palette1 = label1.palette()
        palette2 = label2.palette()
        palette1.setColor(label1.foregroundRole(), color)
        palette2.setColor(label2.foregroundRole(), color)
        label1.setPalette(palette1)
        label2.setPalette(palette2)
    
    def startTimeChanged(self):
        spinBox = self.findChild(QSpinBox,"startImgSpinBox_T")
        valueInMs = int(spinBox.value()) * 100
        MsPerFrame = self.fileDurationMs / self.stftModul.shape[1]
        startFrame = int(np.floor(valueInMs / MsPerFrame))
        self.startFrameTime = startFrame
        
        label1 = self.findChild(QLabel,"calculatedFrameToStart_T")
        label1.setText(" = " + str(startFrame) + " klatka czasu")
        self.setProperColorOfGuiTexts("time")
        
        
    def startFrequencyChanged(self):
        spinBox = self.findChild(QSpinBox,"startImgSpinBox_F")
        startValueInHz = int(spinBox.value()) * 100
        MAX_FREQ = 20000
        hzPerFrame = MAX_FREQ / self.stftModul.shape[0]
        startFrame = int(np.floor(startValueInHz / hzPerFrame))
        self.startFrameFreq = startFrame
        label = self.findChild(QLabel,"calculatedFrameToStart_F")
        label.setText(" = " + str(startFrame) + " klatka częstotliwosci")
        self.setProperColorOfGuiTexts("freq")
        
    def durationTimeChanged(self):
        spinBox = self.findChild(QSpinBox,"imgDurationSpinBox_T")
        valueInMs = int(spinBox.value()) * 100
        MsPerFrame = self.fileDurationMs / self.stftModul.shape[1]
        durationFrames = int(np.floor(valueInMs / MsPerFrame))
        self.durationFrameTime = durationFrames
        label = self.findChild(QLabel,"calculatedFramesToUse_T")
        label.setText(" = " + str(durationFrames) + " klatek czasu")
        self.setProperColorOfGuiTexts("time")
        
    def frequencyDurationChanged(self):
        spinBox = self.findChild(QSpinBox,"imgDurationSpinBox_F")
        startValueInHz = int(spinBox.value()) * 100
        MAX_FREQ = 20000
        hzPerFrame = MAX_FREQ / self.stftModul.shape[0]
        durationFrames = int(np.floor(startValueInHz / hzPerFrame))
        self.durationFrameFreq = durationFrames
        label = self.findChild(QLabel,"calculatedFramesToUse_F")
        label.setText(" = " + str(durationFrames) + " klatek częstotliwosci")
        self.setProperColorOfGuiTexts("freq")
        
    def paintSpectogramToLabel(self):
        labelToPaint = self.findChild(QLabel,"spectogramLabel")
        img = cv2.imread("./spectogram.png")
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixMap = QPixmap(qImg)
        pixMap = pixMap.scaledToWidth(labelToPaint.width())
        labelToPaint.setPixmap(pixMap)

    def turnOnGui(self):
        self.findChild(QSpinBox ,"startImgSpinBox_T").setEnabled(True)
        self.findChild(QSpinBox ,"startImgSpinBox_F").setEnabled(True)
        self.findChild(QSpinBox ,"imgDurationSpinBox_T").setEnabled(True)
        self.findChild(QSpinBox ,"imgDurationSpinBox_F").setEnabled(True)
        
        
    def calculateSTFT(self):
        if(len(self.audioData) == 0):
            self.printError("some error")
            return
        stft = librosa.stft(self.audioData, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
        self.stftModul, self.stftPhase = fun1.splitCompNum(stft)
        timeFramesLabel = self.findChild(QLabel,"totalTimeFramesLabel")
        freqFramesLabel = self.findChild(QLabel,"totalFreqFramesLabel")
        timeFramesLabel.setText("klatek częst. = " + str(self.stftModul.shape[0]))
        freqFramesLabel.setText("klatek czasowych = " + str(self.stftModul.shape[1]))
        
        turnOnGui = self.stftModul.shape[0] > 0 and self.stftModul.shape[1] > 0 and len(self.imgData) > 0
        if(turnOnGui):
            self.turnOnGui()
        
        fun1.paintSpectogram(stft, self.SAMPLING_RATE, self.HOP_SIZE)
        self.paintSpectogramToLabel()
        
    def calculateRecomendedValues():
        print("calculateRecomendedValues")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  
    window = MainWindow()
    window.show()
    window.fillAudioLabels(0,0)
    window.fillImageLabels(fileName = "Obraz: ", width=0 , height=0, minLum=0, maxLum=0)
    sys.exit(app.exec_())