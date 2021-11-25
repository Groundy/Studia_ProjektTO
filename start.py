import sys
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLabel, QSpinBox, QMessageBox
import funcs
import librosa 
from PyQt5.QtGui import QImage,QPixmap
import cv2

Ui_MainWindow, QtBaseClass = uic.loadUiType("MainWindow.ui")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    audioData = None
    SAMPLING_RATE = 0
    imgData = None
    FRAME_SIZE = int(1024)
    HOP_SIZE = int(FRAME_SIZE / 4)
    STFT = None
    LEN_FREQ_FRAMES = 0
    LEN_TIME_FRAMES = 0
    fileDurationMs = 0
    START_TIME_FRAME = 0
    START_FREQ_FRAME = 0
    DURATION_TIME_FRAME = 0
    DURATION_FREQ_FRAME = 0
    
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
            audioDataTmp, durationTMP, samplingRate = funcs.readAudioFile(selectedFiles[0])
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
            self.imgData ,width , height, minLum, maxLum = funcs.readImg(absPathToFile)
            self.fillImageLabels(absPathToFile, width , height, minLum, maxLum)
    
    def startTimeChanged(self):
        spinBox = self.findChild(QSpinBox,"startImgSpinBox_T")
        valueInMs = int(spinBox.value()) * 100
        MsPerFrame = self.fileDurationMs / self.LEN_TIME_FRAMES
        startFrame = int(np.floor(valueInMs / MsPerFrame))
        self.START_TIME_FRAME = startFrame
        label = self.findChild(QLabel,"calculatedFrameToStart_T")
        label.setText(" = " + str(startFrame) + " klatka czasu")
        
    def startFrequencyChanged(self):
        spinBox = self.findChild(QSpinBox,"startImgSpinBox_F")
        startValueInHz = int(spinBox.value()) * 100
        MAX_FREQ = 20000
        hzPerFrame = MAX_FREQ / self.LEN_FREQ_FRAMES
        startFrame = int(np.floor(startValueInHz / hzPerFrame))
        self.START_FREQ_FRAME = startFrame
        label = self.findChild(QLabel,"calculatedFrameToStart_F")
        label.setText(" = " + str(startFrame) + " klatka częstotliwosci")
        
    def durationTimeChanged(self):
        spinBox = self.findChild(QSpinBox,"imgDurationSpinBox_T")
        valueInMs = int(spinBox.value()) * 100
        MsPerFrame = self.fileDurationMs / self.LEN_TIME_FRAMES
        durationFrames = int(np.floor(valueInMs / MsPerFrame))
        self.DURATION_TIME_FRAME = durationFrames
        label = self.findChild(QLabel,"calculatedFramesToUse_T")
        label.setText(" = " + str(durationFrames) + " klatek czasu")
        
    def frequencyDurationChanged(self):
        spinBox = self.findChild(QSpinBox,"imgDurationSpinBox_F")
        startValueInHz = int(spinBox.value()) * 100
        MAX_FREQ = 20000
        hzPerFrame = MAX_FREQ / self.LEN_FREQ_FRAMES
        durationFrames = int(np.floor(startValueInHz / hzPerFrame))
        self.DURATION_FREQ_FRAME = durationFrames
        label = self.findChild(QLabel,"calculatedFramesToUse_F")
        label.setText(" = " + str(durationFrames) + " klatek częstotliwosci")
        
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
        if(len(self.audioData) == 0 ):
            #wyswietlic jakis blad
            return
        STFT_TMP = librosa.stft(self.audioData, n_fft=self.FRAME_SIZE, hop_length=self.HOP_SIZE)
        STFT_TMP = abs(STFT_TMP)
        self.STFT = STFT_TMP
        self.LEN_FREQ_FRAMES = len(STFT_TMP[1])
        self.LEN_TIME_FRAMES = len(STFT_TMP[0])
        timeFramesLabel = self.findChild(QLabel,"totalTimeFramesLabel")
        freqFramesLabel = self.findChild(QLabel,"totalFreqFramesLabel")
        timeFramesLabel.setText("klatek częst. = " + str(self.LEN_FREQ_FRAMES))
        freqFramesLabel.setText("klatek czasowych = " + str(self.LEN_TIME_FRAMES))
        
        turnOnGui = self.LEN_FREQ_FRAMES > 0 and self.LEN_TIME_FRAMES > 0 and len(self.imgData) > 0
        if(turnOnGui):
            self.turnOnGui()
        
        funcs.paintSpectogram(STFT_TMP,self.SAMPLING_RATE, self.HOP_SIZE)
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