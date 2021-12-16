# -*- coding: utf-8 -*-
import librosa 
import librosa.display
import cv2
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import cmath
import fun1
import librosa 
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QDateTime, QSettings
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit


Ui_MainWindow, QtBaseClass = uic.loadUiType("MainWindow.ui")
windowType = "nuttall"

def readAudioFile(absPath):
    audioData, samplingRate = librosa.load(absPath, sr=None)
    duration = int(1000*(len(audioData))/samplingRate)
    return audioData, duration, samplingRate

def readImg(absPath):
    img = cv2.imread(absPath, cv2.IMREAD_GRAYSCALE)
    width = img.shape[1]
    height = img.shape[0]
    minLum = np.min(img)
    maxLum = np.max(img)
    return img , width , height, minLum, maxLum

def paintSpectogram(Y, sr, hop_length, y_axis="linear" ):    
    Y = Y / np.max(Y)
    Y = Y ** 2
    Y = librosa.power_to_db(Y)
    figure = plt.figure(figsize=(20, 10))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    figure.tight_layout()
    return figure
    
def mapImgToSTFT(ptr, startX, startY, monoImg, stft, durationX = 0, durationY = 0, amplifierDb = -10):
    Log(ptr,"Rozpoczęto mapowanie obrazu do spektrogramu")
    
    needToScaleImg = durationX != 0 and durationY != 0
    if(needToScaleImg):
        aboveX = (startX + durationY) >= stft.shape[1]
        aboveY = (startY + durationY) >= stft.shape[0]
    else:
        aboveX = (startX + monoImg.shape[1]) >= stft.shape[1]
        aboveY = (startY + monoImg.shape[0]) >= stft.shape[0]
    dimensionError = startX < 0 or startY < 0 or aboveX or aboveY
    if(dimensionError):
        Log(ptr, "[ERROR] Źle wprowadzono punkt startu i wymiary obrazu do mapowania")
        return
    
    minVal = np.min(stft)
    maxVal = np.max(stft)
    if(maxVal <= minVal):
        Log(ptr, "[ERROR] błąd skalowania jasnosci obrazu")
    valPerPix = (maxVal - minVal) / 255
    
    
    imgCpy = monoImg.copy()
    stftCpy = stft.copy()
    if(needToScaleImg):
        shape = (durationX, durationY)
        factorX = durationX / monoImg.shape[1]
        factorY = durationY / monoImg.shape[0]
        imgCpy = cv2.resize(imgCpy, shape, fx = factorX, fy = factorY)
        
    imgW = imgCpy.shape[1]
    imgH = imgCpy.shape[0]
    for x in range(0, imgW):
        for y in range(0, imgH):   
            yPosSpect = y + startY   
            xPosSpect = x + startX   
            valOfLumInPic = imgCpy[imgH - y - 1][x]
            valToSet = minVal + (valPerPix * valOfLumInPic)
            valToSet *= 10**(amplifierDb/10)
            stftCpy[yPosSpect][xPosSpect] = valToSet
            
    return stftCpy

def stftToWavFile(stft, fileName, frame_size, hop_size , samplingRate = 44100):
    audioData = librosa.istft(stft, hop_size, frame_size, window=windowType)
    sf.write(fileName, audioData, samplingRate, 'PCM_24')
    
def mergeCompNum(modulArr, phaseArr):
    xShape = modulArr.shape[1]
    yShape = modulArr.shape[0]
    complexArr = np.zeros_like(phaseArr, dtype=np.complex64)
    for x in range(xShape):
            for y in range(yShape):
                complexArr[y][x] = cmath.rect(modulArr[y][x], phaseArr[y][x])
    return complexArr

def splitCompNum(complexNumberArr):
    xShape = complexNumberArr.shape[1]
    yShape = complexNumberArr.shape[0]
    modulArr = np.zeros_like(complexNumberArr, dtype=np.float32)
    phaseArr = np.zeros_like(complexNumberArr, dtype=np.float32)
    for x in range(xShape):
            for y in range(yShape):
                modulArr[y][x], phaseArr[y][x] = cmath.polar(complexNumberArr[y][x])
    return modulArr, phaseArr

def pyPlotToCv2Img(fig): 
    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_image = np.array(fig.canvas.get_renderer()._renderer)
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
    return graph_image

def Log(ptr, text):
    if(ptr != None):
        logView = ptr.findChild(QPlainTextEdit,"plainTextEdit")
        logView.appendPlainText(text)

def saveFiles(ptr):
    Log(ptr,"Zapisywanie plików na dysk")
    folderPath = str(QFileDialog.getExistingDirectory(ptr, 'Select Folder'))
    if(len(folderPath) == 0):
        Log(ptr,"Nie wybrano folderu do zapisu")
        
    currentTimeStr = QDateTime.currentDateTime().toString("hh_mm_ss_")
    baseFileName = folderPath + "/" + currentTimeStr
    audio_FileName = baseFileName + "audio.wav"
    key_FileName =  baseFileName + "key.ini"
    spectogramOrg_FileName = baseFileName + "spect_oryginalny.png"
    spectogramModified_FileName = baseFileName + "spect_modified.png"
    spectogramReadFromFile_FileName = baseFileName + "spect_read_from_wav_file.png"
    imgFromSpectogram_FileName = baseFileName + "imgFromSpectogram.png"
    imgFromWav_FileName = baseFileName + "imgFromWav.png"
    
    mergedStft = mergeCompNum(ptr.stftModulModified, ptr.stftPhaseOrg)
    stftToWavFile(mergedStft, audio_FileName, ptr.FRAME_SIZE, ptr.HOP_SIZE, ptr.samplingRate)
    
    orgSpectogram = paintSpectogram(ptr.stftModulOrg, ptr.samplingRate, ptr.HOP_SIZE)
    orgSpectogram.savefig(spectogramOrg_FileName)
    
    modifiedSpectogram = paintSpectogram(ptr.stftModulModified, ptr.samplingRate, ptr.HOP_SIZE)
    modifiedSpectogram.savefig(spectogramModified_FileName)
    
    settingsFile = QSettings(key_FileName, QSettings.IniFormat)
    settingsFile.setValue("startT", ptr.startFrameTime)
    settingsFile.setValue("startF", ptr.startFrameFreq)
    settingsFile.setValue("durationT", ptr.durationFrameTime)
    settingsFile.setValue("durationF", ptr.durationFrameFreq)
    settingsFile.setValue("amplification", ptr.amplification)
    settingsFile.setValue("frame_size", ptr.FRAME_SIZE)
    settingsFile.setValue("hop_size", ptr.HOP_SIZE)
    settingsFile.setValue("samplingRate", ptr.samplingRate)
    settingsFile.setValue("minVal", int(np.min(ptr.stftModulModified)))
    settingsFile.setValue("maxVal", int(np.max(ptr.stftModulModified)))
    
    #tutaj czytamy uprzednio zapisane pliki
    imgFromMappedSpect = fun1.extractImgFromSTFT(
        stftModul = ptr.stftModulModified,
        startT = ptr.startFrameTime,
        startF = ptr.startFrameFreq,
        durationT = ptr.durationFrameTime,
        durationF = ptr.durationFrameFreq,
        amplification = ptr.amplification
        ) 
    

    Log(ptr,"Odczyt plików z dysku")
    startT_Read, startF_Read, durationT_Read, durationF_Read, amplification_Read, frame_size_Read, hop_size_Read, samplingRate_Read, minVal_Read, maxVal_Read = readParamsFromIniFile(key_FileName)
    audioDataRead, duration, samplingRateRead = readAudioFile(audio_FileName)
    
    Log(ptr,"Analiza spektrogramu ze zmodyfikowanego pliku wav")
    stftRead = librosa.stft(audioDataRead, n_fft=frame_size_Read, hop_length=hop_size_Read, window=windowType)
    stftModulRead, stftPhaseRead = splitCompNum(stftRead)
    spectogramRead = paintSpectogram(stftModulRead, samplingRate_Read, hop_size_Read)
    spectogramRead.savefig(spectogramReadFromFile_FileName)
    ptr.paintSpectogramToLabel(spectogramRead)
    
    imgReadFromWavFile = fun1.extractImgFromSTFT(
        stftModul = stftModulRead,
        startT = startT_Read,
        startF = startF_Read,
        durationT = durationT_Read,
        durationF = durationF_Read,
        amplification = amplification_Read
        )
    
    cv2.imwrite(imgFromSpectogram_FileName, imgFromMappedSpect)
    cv2.imwrite(imgFromWav_FileName, imgReadFromWavFile)
    calcErrorRates(ptr,imgFromMappedSpect,imgReadFromWavFile)

def extractImgFromSTFT(stftModul, startT, startF, durationT, durationF, amplification = -10):
    maxValue = np.max(stftModul)
    f_start = startF
    f_end = durationF + startF
    t_start = startT
    t_end = durationT + startT
    img = stftModul[f_start : f_end , t_start : t_end]
    img *= 10**(-amplification/10)
    img = (img / np.max(img)) * maxValue
    img = img.astype(int)
    img = np.flipud(img)
    return img

def readParamsFromIniFile(pathToIniFile):
    settingsFile = QSettings(pathToIniFile, QSettings.IniFormat)
    startT_Read = settingsFile.value("startT", 0, int)
    startF_Read = settingsFile.value("startF", 0, int)
    durationT_Read = settingsFile.value("durationT", 0, int)
    durationF_Read = settingsFile.value("durationF", 0, int)
    amplification_Read = settingsFile.value("amplification", -10, float)
    frame_size_Read = settingsFile.value("frame_size", 1024, int)
    hop_size_Read = settingsFile.value("hop_size", 256, int)
    samplingRate_Read = settingsFile.value("samplingRate", 44100, int)
    minVal_Read = settingsFile.value("minVal", 0, int)
    maxVal_Read = settingsFile.value("maxVal", 150, int)                             
    return startT_Read, startF_Read, durationT_Read, durationF_Read, amplification_Read, frame_size_Read, hop_size_Read, samplingRate_Read, minVal_Read, maxVal_Read

def calcErrorRates(ptr, orgImg, readFromWavImg):
    diff = readFromWavImg.copy() - orgImg.copy()
    pixelsAmount = orgImg.shape[0] * orgImg.shape[1]
    Log(ptr, "Wyliczanie miar błędów")
    mad = np.sum(abs(diff)) / pixelsAmount
    mse = np.sum(diff ** 2) / pixelsAmount
    
    mianownik = np.sum(diff ** 2)
    
    snr = np.sum(orgImg.copy() ** 2) / mianownik
    snr = 10 * np.log10(snr)
    
    maxPixVal = np.max(orgImg)
    psnr = np.sum(pixelsAmount * (maxPixVal ** 2)) / mianownik
    psnr = 10 * np.log10(psnr)

    psnrStr = str(round(psnr,2))
    madStr = str(round(mad,2)) 
    mseStr = str(round(mse,2))
    Log(ptr,"MSE: " + mseStr)
    Log(ptr,"MAD: " + madStr)
    Log(ptr,"PSNR: " + psnrStr + " dB")
    return mse, mad, snr, psnr
