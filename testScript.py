# -*- coding: utf-8 -*-
import fun1
import cv2
import numpy as np
import librosa
pathToImg = "./testFiles/qr.png"
pathToAudio = "./testFiles/file.wav"
pathToResultFolder = "./res/"
orgImg = cv2.imread(pathToImg, cv2.IMREAD_GRAYSCALE)
FRAME_SIZE = int(1024)
HOP_SIZE = int(FRAME_SIZE / 4)
windowType = "nuttall"


audioData, samplingRate = librosa.load(pathToAudio, sr=None)
duration = int(1000*(len(audioData))/samplingRate)

orgStft = librosa.stft(audioData, FRAME_SIZE, hop_length=HOP_SIZE, window=fun1.windowType)
orgStftR, orgStftPhase = fun1.splitCompNum(orgStft)

def doSingleCalc(posX, posY, amp):
    modifiedStftR = fun1.mapImgToSTFT(
        ptr = None,
        startX = posX,
        startY = posY,
        monoImg = orgImg,
        stft = orgStftR,
        amplifierDb = amp
        )
    fileNameForWavFile = pathToResultFolder + "file.wav"
    stft = fun1.mergeCompNum(modifiedStftR,orgStftPhase)
    fun1.stftToWavFile(stft, fileNameForWavFile, FRAME_SIZE, HOP_SIZE , samplingRate)

    audioDataRead = librosa.load(fileNameForWavFile, samplingRate)
    stftRead = librosa.stft(audioDataRead[0], FRAME_SIZE, hop_length=HOP_SIZE, window=fun1.windowType)
    stftReadR, stftReadPhase = fun1.splitCompNum(stftRead)
    imgRead = fun1.extractImgFromSTFT(
        stftModul = stftReadR,
        startT = posX,
        startF = posY,
        durationT = orgImg.shape[1],
        durationF = orgImg.shape[0],
        amplification = amp)
    
    fileName = pathToResultFolder + str(-amp) + ".png"
    cv2.imwrite(fileName ,imgRead)
    
    psnr, mad, mse = fun1.calcErrorRates(None, orgImg, imgRead)
    return psnr, mad, mse
    


amplificationLevels = np.arange(-0, -62, -2)

for amp in amplificationLevels:
    mse, mad, snr, psnr = doSingleCalc(280, 222, amp)
    print(str(amp))
    print(str(round(mse,3)))     
    print(str(round(mad,3)))  
    print(str(round(snr,3))) 
    print(str(round(psnr,3))) 
    
