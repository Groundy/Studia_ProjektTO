# -*- coding: utf-8 -*-
import librosa 
import librosa.display
import cv2
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import cmath
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from PyQt5 import QtCore
from PyQt5.QtCore import QDateTime, QSettings

windowType = "nuttall"
scale = 0.1#parametr ktory ogranicza wartosc maksymalna obrazu tak aby ta moc "nie wylewala sie" na inne czestotliwosci

def readAudioFile(absPath):
    audioData, samplingRate = librosa.load(absPath,sr=None)
    duration = int(1000*(len(audioData))/samplingRate)
    return audioData, duration, samplingRate

def readImg(absPath):
    img = cv2.imread(absPath, cv2.IMREAD_GRAYSCALE)
    width = img.shape[1]
    height = img.shape[0]
    minLum = np.min(img)
    maxLum = np.max(img)
    return img ,width , height, minLum, maxLum

def paintSpectogram(Y, sr, hop_length, y_axis="linear" ):    
    Y = Y / np.max(Y)
    Y = Y ** 2
    Y = librosa.power_to_db(Y)
    figure = plt.figure(figsize=(20, 10))
    librosa.display.specshow(Y, sr=sr,hop_length=hop_length, x_axis="time",y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    figure.tight_layout()
    #plt.savefig("spectogram.png",bbox_inches='tight')
    return figure
    
def mapImgToSTFT(startX, startY, monoImg, stft, durationX = 0, durationY = 0, amplifierDb = 0):
    #wzmocnienie nie powinno byc wieksze niz 0
    needToScaleImg = durationX != monoImg.shape[1] and durationY != monoImg.shape[0]
    if(needToScaleImg):
        aboveX = (startX + durationY) >= stft.shape[1]
        aboveY = (startY + durationY) >= stft.shape[0]
    else:
        aboveX = (startX + monoImg.shape[1]) >= stft.shape[1]
        aboveY = (startY + monoImg.shape[0]) >= stft.shape[0]
    dimensionError = startX < 0 or startY < 0 or aboveX or aboveY
    if(dimensionError):
        print("dimErr!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return
    
    minVal = np.min(stft)
    maxVal = np.max(stft) * scale
    if(maxVal <= minVal):
        print("Error scaling pix lum")
    valPerPix = (maxVal - minVal) / 255
    
    
    imgCpy = monoImg.copy()
    stftCpy = stft.copy()
    if(needToScaleImg):
        shape = (durationX,durationY)
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
            if(amplifierDb != 0):
                valToSet*= 10**(amplifierDb/10)
            stftCpy[yPosSpect][xPosSpect] = valToSet
    return stftCpy

def stftToWavFile(stft, fileName, frame_size, hop_size ,samplingRate = 44100):
    audioData = librosa.istft(stft, hop_size, frame_size,window=windowType)
    sf.write(fileName, audioData, samplingRate, 'PCM_24')
    
def mergeCompNum(modulArr, phaseArr):
    xShape = modulArr.shape[1]
    yShape = modulArr.shape[0]
    complexArr = np.zeros_like(phaseArr,dtype=np.complex64)
    for x in range(xShape):
            for y in range(yShape):
                complexArr[y][x] = cmath.rect(modulArr[y][x], phaseArr[y][x])
    return complexArr

def splitCompNum(complexNumberArr):
    xShape = complexNumberArr.shape[1]
    yShape = complexNumberArr.shape[0]
    modulArr = np.zeros_like(complexNumberArr,dtype=np.float32)
    phaseArr = np.zeros_like(complexNumberArr,dtype=np.float32)
    for x in range(xShape):
            for y in range(yShape):
                modulArr[y][x],phaseArr[y][x] = cmath.polar(complexNumberArr[y][x])
    return modulArr, phaseArr

def pyPlotToCv2Img(fig): 
    # create a figure
    #fig = plt.figure()
    canvas = FigureCanvas(fig)
    canvas.draw()

    # convert canvas to image
    graph_image = np.array(fig.canvas.get_renderer()._renderer)

    # it still is rgb, convert to opencv's default bgr
    graph_image = cv2.cvtColor(graph_image,cv2.COLOR_RGB2BGR)
    return graph_image


def saveFiles(stft_modul, stft_phase, pathToFolder, startT, startF, durationT,
              durationF, amplification, frame_size, hop_size, samplingRate):
    
    currentTimeStr = QDateTime.currentDateTime().toString("hh_mm_ss_")
    baseFileName = pathToFolder + "/" + currentTimeStr
    audio_FileName = baseFileName + "audio.wav"
    key_FileName =  baseFileName + "key.ini"
    spectogramModified_FileName = baseFileName + "spect1.png"
    spectogramRead_FileName = baseFileName + "spect2.png"
    imgCalculated_FileName = baseFileName + "imgCalculated.png"
    imgRead_FileName = baseFileName + "imgRead.png"
    
    mergedStft = mergeCompNum(stft_modul, stft_phase)
    stftToWavFile(mergedStft,audio_FileName,frame_size,hop_size,samplingRate)
    
    figure = paintSpectogram(stft_modul,samplingRate,hop_size)
    figure.savefig(spectogramModified_FileName)
    
    #oczyt zapisanego dzwieku i ponowna zmiana do spectrogramu
    audioDataRead, duration, samplingRateRead = readAudioFile(audio_FileName)
    stftRead = librosa.stft(audioDataRead, n_fft=frame_size, hop_length=hop_size,window=windowType)
    stftModulRead, stftPhaseRead = splitCompNum(stftRead)
    spectogramRead = paintSpectogram(stftModulRead,samplingRateRead,hop_size)
    spectogramRead.savefig(spectogramRead_FileName)
    imgRead, imgCalculated, mad = compareTwoModArrays(stftModulRead, stft_modul, startT, startF, durationT, durationF, pathToFolder + "/")
    
    cv2.imwrite(imgCalculated_FileName,imgCalculated)
    cv2.imwrite(imgRead_FileName,imgRead)
    
    settingsFile = QSettings(key_FileName, QSettings.IniFormat)
    settingsFile.setValue("startT",startT)
    settingsFile.setValue("startF",startF)
    settingsFile.setValue("durationT",durationT)
    settingsFile.setValue("durationF",durationF)
    settingsFile.setValue("amplification",amplification)
    settingsFile.setValue("frame_size",frame_size)
    settingsFile.setValue("hop_size",hop_size)
    settingsFile.setValue("samplingRate",samplingRate)
    settingsFile.setValue("MAD",str(mad))
    
    return spectogramRead

def compareTwoModArrays(stftModulRead, stftModulCalculated, startT, startF, durationT, durationF, pathToFolder = "./"):
    f_start = startF
    f_end = durationF + startF
    t_start = startT
    t_end = durationT + startT
    imgCalculated = stftModulCalculated[f_start : f_end , t_start : t_end]
    imgRead = stftModulRead[f_start : f_end , t_start : t_end]
    #odwracania macierzy do gory nogami
    imgCalculated = np.flipud(imgCalculated)
    imgRead = np.flipud(imgRead)
    
    imgCalculated = 255 * (imgCalculated / np.max(imgCalculated))
    imgRead = 255 * (imgRead / np.max(imgRead))              
    
    imgCalculated = imgCalculated.astype(int)
    imgRead = imgRead.astype(int)
    
    #cv2.imwrite(pathToFolder + "imgCalculated.png",imgCalculated)
    #cv2.imwrite(pathToFolder + "imgRead.png",imgRead)
    
    diffSum = np.sum(abs(imgRead - imgCalculated))
    mad = diffSum/(durationT * durationF)
    mad = round(mad,2)
    return imgRead, imgCalculated, mad
    