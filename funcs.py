# -*- coding: utf-8 -*-
import librosa 
import librosa.display
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QDateTime 


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