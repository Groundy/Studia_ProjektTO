# -*- coding: utf-8 -*-
import librosa 
import librosa.display
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(Y, sr=sr,hop_length=hop_length, x_axis="time",y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.savefig("spectogram.png",bbox_inches='tight')
    
def mapImgToSTFT(startX, startY, monoImg, stft, durationX = 0, durationY = 0, amplifierDb = 0):
    #startX, startY odnosza sie do lewego gornego rogu spektrogramu
    #wzmocnienie nie powinno byc wieksze niz 0
    needToScaleImg = durationX != 0 and durationY != 0
    lessThanZeroZero = startX < 0 or startY < 0
    dimensionError = False
    if(needToScaleImg):
        aboveX = (startX + durationY) >= stft.shape[1]
        aboveY = (startY + durationY) >= stft.shape[0]
        dimensionError = lessThanZeroZero or aboveX or aboveY
    else:
        aboveX = (startX + monoImg.shape[1]) >= stft.shape[1]
        aboveY = (startY + monoImg.shape[0]) >= stft.shape[0]
        dimensionError = lessThanZeroZero or aboveX or aboveY
    if(dimensionError):
        print("dimErr!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return
    
    minVal = np.min(stft)
    maxVal = np.max(stft)
    valPerPix = (maxVal - minVal) / 255
    
    if(needToScaleImg):
        shape = (durationX,durationY)
        factorX = durationX / monoImg.shape[1]
        factorY = durationY / monoImg.shape[0]
        monoImg = cv2.resize(monoImg, shape, fx = factorX, fy = factorY)
    for x in range(0, monoImg.shape[1]):
        for y in range(0, monoImg.shape[0]):   
            yPosSpect = stft.shape[0] - (y + startY + 1)   
            xPosSpect = x + startX   
            valOfLumInPic = monoImg[y][x]
            valToSet = minVal + (valPerPix * valOfLumInPic)
            if(amplifierDb != 0):
                valToSet*= 10**(amplifierDb/10)
            stft[yPosSpect][xPosSpect] = valToSet
    return stft

def stftToWavFile(stft, fileName, frame_size, hop_size ,samplingRate = 44100):
    audioData = librosa.istft(stft, hop_size, frame_size)
    sf.write(fileName, audioData, samplingRate, 'PCM_24')