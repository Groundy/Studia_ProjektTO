import sys
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLabel, QSpinBox, QCheckBox, QPushButton, QDoubleSpinBox
import fun1
import librosa 
from PyQt5.QtGui import QImage,QPixmap, QColor

Ui_MainWindow, QtBaseClass = uic.loadUiType("MainWindow.ui")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    audioData = None
    samplingRate = 0
    fileDurationMs = 0
    imgData = None
    FRAME_SIZE = int(1024)
    HOP_SIZE = int(FRAME_SIZE / 4)
    stftModulOrg = None
    stftPhaseOrg = None
    stftModulModified = None
    startFrameTime = 0
    startFrameFreq = 0
    durationFrameTime = 0
    durationFrameFreq = 0
    miliSecsForTimeFrame = 0
    hzPerFreqFrame = 0
    amplification = float(0)
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
    def aplifierSpinBoxChanged(self):
        value = float(self.findChild(QDoubleSpinBox ,"amplifierSpinBox").value())
        if(value >= -40 and value <= -10):
            self.amplification = value
    
    def fillAudioLabels(self):
        durationMsec = self.fileDurationMs
        
        samplingText = str("Próbkowanie: ") + str(self.samplingRate) + " Hz"
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

    def resetLabelToUnmodifiedSpectrogram(self):
        figure = fun1.paintSpectogram(self.stftModulOrg, self.samplingRate, self.HOP_SIZE)
        self.paintSpectogramToLabel(figure)
        
    def addImgToTmpSpectrogram(self):
        self.amplification = self.findChild(QDoubleSpinBox ,"amplifierSpinBox").value()
        self.stftModulModified = fun1.mapImgToSTFT(
            self,
            self.startFrameTime,
            self.startFrameFreq,
            self.imgData,
            self.stftModulOrg,
            self.durationFrameTime,
            self.durationFrameFreq,
            self.amplification)
        figure = fun1.paintSpectogram(self.stftModulModified, self.samplingRate, self.HOP_SIZE)
        self.paintSpectogramToLabel(figure)
        
    def saveResultsToFiles(self):
        fun1.saveFiles(self)
        
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
            self.audioData, self.fileDurationMs, self.samplingRate = fun1.readAudioFile(selectedFiles[0])
            self.fillAudioLabels()
            
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
            label1 = self.findChild(QLabel,"calculatedTimeToStart")
            label2 = self.findChild(QLabel,"calculatedTimeToUse")
            areValuesCorrect = self.durationFrameTime + self.startFrameTime < self.stftModulOrg.shape[1]
        elif(axis == "freq"):
            label1 = self.findChild(QLabel,"calculatedFreqToStart")
            label2 = self.findChild(QLabel,"calculatedFreqToUse")
            areValuesCorrect = self.durationFrameFreq + self.startFrameFreq < self.stftModulOrg.shape[0]
        
        color = QColor(0x000000) if areValuesCorrect else QColor(0xff0000)
        palette1 = label1.palette()
        palette2 = label2.palette()
        palette1.setColor(label1.foregroundRole(), color)
        palette2.setColor(label2.foregroundRole(), color)
        label1.setPalette(palette1)
        label2.setPalette(palette2)
    
    def startTimeChanged(self):
        spinBox = self.findChild(QSpinBox,"startImgSpinBox_T")
        startTimeInFrames = int(spinBox.value())
        startTimeInMsc = int(startTimeInFrames * self.miliSecsForTimeFrame)
        self.startFrameTime = startTimeInFrames
        
        label1 = self.findChild(QLabel,"calculatedTimeToStart")
        label1.setText("= " + str(startTimeInMsc) + " ms")
        self.setProperColorOfGuiTexts("time")
        
    def allowScalingChechBoxChanged(self):
        isChecked = self.findChild(QCheckBox,"scalingChekbox").isChecked()
        spinBox1 = self.findChild(QSpinBox ,"imgDurationSpinBox_T")
        spinBox2 = self.findChild(QSpinBox ,"imgDurationSpinBox_F")
        if(isChecked):
            spinBox1.setEnabled(True)
            spinBox2.setEnabled(True)            
            spinBox1.setValue(1)
            spinBox2.setValue(1)
        else:
            spinBox1.setEnabled(False)
            spinBox2.setEnabled(False)    
            spinBox1.setValue(self.imgData.shape[1])
            spinBox2.setValue(self.imgData.shape[0])
        self.durationTimeChanged()
        self.frequencyDurationChanged()
        
    def printScaleRatioToGui(self):
        h = float(self.durationFrameFreq)
        w = float(self.durationFrameTime)
        properValues =  h > 0 and w > 0 
        if(properValues):
            dimRatio =  w/h
            dimRatioStr = str(round(dimRatio, 2))
            textToSet = "Przeskalowany stosunek W/H : "+ dimRatioStr if properValues else "Złe parametry skalowania"     
            label = self.findChild(QLabel,"WToHRatioLabel_mod")
            label.setText(textToSet)
        
    def startFrequencyChanged(self):
        spinBox = self.findChild(QSpinBox,"startImgSpinBox_F")
        startFreqInFrames = int(spinBox.value())
        startFreqInHz = int(startFreqInFrames * self.hzPerFreqFrame)
        self.startFrameFreq = startFreqInFrames
        label = self.findChild(QLabel,"calculatedFreqToStart")
        label.setText("= " + str(startFreqInHz) + " Hz")
        self.setProperColorOfGuiTexts("freq")
        
    def durationTimeChanged(self):
        spinBox = self.findChild(QSpinBox,"imgDurationSpinBox_T")
        durationTimeInFrames = int(spinBox.value())
        durationTimeInMsc = int(durationTimeInFrames * self.miliSecsForTimeFrame)
        self.durationFrameTime = durationTimeInFrames
        label = self.findChild(QLabel,"calculatedTimeToUse")
        label.setText("= " + str(durationTimeInMsc) + " ms")
        self.setProperColorOfGuiTexts("time")
        self.printScaleRatioToGui()
        
    def frequencyDurationChanged(self):
        spinBox = self.findChild(QSpinBox,"imgDurationSpinBox_F")
        durationFreqInFrames = int(spinBox.value())
        durationFreqInHz = int(durationFreqInFrames * self.hzPerFreqFrame)
        self.durationFrameFreq = durationFreqInFrames
        label = self.findChild(QLabel,"calculatedFreqToUse")
        label.setText("= " + str(durationFreqInHz) + " Hz")
        self.setProperColorOfGuiTexts("freq")
        self.printScaleRatioToGui()
        
    def paintSpectogramToLabel(self, spectrogram):
        labelToPaint = self.findChild(QLabel,"spectogramLabel")
        img = fun1.pyPlotToCv2Img(spectrogram)
        height= img.shape[0]
        width = img.shape[1]
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixMap = QPixmap(qImg)
        pixMap = pixMap.scaledToWidth(labelToPaint.width())
        labelToPaint.setPixmap(pixMap)

    def turnOnGui(self):
        self.findChild(QSpinBox ,"startImgSpinBox_T").setEnabled(True)
        self.findChild(QSpinBox ,"startImgSpinBox_F").setEnabled(True)
        self.findChild(QCheckBox,"scalingChekbox").setEnabled(True)
        self.findChild(QSpinBox ,"imgDurationSpinBox_T").setValue(self.imgData.shape[1])
        self.findChild(QSpinBox ,"imgDurationSpinBox_F").setValue(self.imgData.shape[0])
        
        self.findChild(QPushButton ,"resetSpectLabelButton").setEnabled(True)
        self.findChild(QPushButton ,"setImgInSpectButton").setEnabled(True)
        self.findChild(QPushButton ,"saveResultsButton").setEnabled(True)
        self.findChild(QDoubleSpinBox ,"amplifierSpinBox").setEnabled(True)

        
    def calculateSTFT(self):
        if(len(self.audioData) == 0):
            fun1.Log(self, "[ERROR] Nie wybrano pliku dzwiękowego")
            return
        fun1.Log(self, "Wyliczanie spektrogramu oryginalnego pliku")
        stft = librosa.stft(self.audioData, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE,window=fun1.windowType)
        self.stftModulOrg, self.stftPhaseOrg = fun1.splitCompNum(stft)
        
        timeFramesLabel = self.findChild(QLabel,"totalTimeFramesLabel")
        freqFramesLabel = self.findChild(QLabel,"totalFreqFramesLabel")
        timeFramesLabel.setText("klatek częst. = " + str(self.stftModulOrg.shape[0]))
        freqFramesLabel.setText("klatek czasowych = " + str(self.stftModulOrg.shape[1]))
        widthHightRatioLabelOrg = self.findChild(QLabel,"WToHRatioLabel_org")
        dimRatio = float(self.imgData.shape[0]) / float(self.imgData.shape[1])
        dimRatioStr = str(round(dimRatio, 2))
        widthHightRatioLabelOrg.setText("Oryginalny stosunek W/H : "+dimRatioStr)
        
        maxFreq = (1 + self.FRAME_SIZE / 2) * self.samplingRate / self.FRAME_SIZE
        self.hzPerFreqFrame = maxFreq / stft.shape[0]
        self.miliSecsForTimeFrame = self.fileDurationMs / stft.shape[1]
        turnOnGui = self.stftModulOrg.shape[0] > 0 and self.stftModulOrg.shape[1] > 0 and len(self.imgData) > 0
        if(turnOnGui):
            self.turnOnGui()
        spectrogram = fun1.paintSpectogram(self.stftModulOrg, self.samplingRate, self.HOP_SIZE)
        self.paintSpectogramToLabel(spectrogram)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)  
    window = MainWindow()
    window.show()
    window.fillImageLabels(fileName = "Obraz: ", width=0 , height=0, minLum=0, maxLum=0)
    sys.exit(app.exec_())