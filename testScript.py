# -*- coding: utf-8 -*-
import fun1
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
pathToImg = "./testFiles/qr.png"
pathToAudio = "./testFiles/file.wav"
pathToResultFolder = "./res/"
FRAME_SIZE = int(1024)
HOP_SIZE = int(FRAME_SIZE / 4)
windowType = "nuttall"


audioData, samplingRate = librosa.load(pathToAudio, sr=None)
duration = int(1000*(len(audioData))/samplingRate)
orgStft = librosa.stft(audioData, FRAME_SIZE, hop_length=HOP_SIZE, window=fun1.windowType)
orgStftR, orgStftPhase = fun1.splitCompNum(orgStft)

def doSingleCalc(posX, posY, amp, orgImg, nameToSave = ""):
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
    
    if(len(nameToSave) > 0):
        fileName = pathToResultFolder + nameToSave
        fileName += "(AMP_" + str(-amp) + ")"
        fileName += ".png"
        cv2.imwrite(fileName ,imgRead)
    
    mse, mad, snr, psnr = fun1.calcErrorRates(None, orgImg, imgRead)
    return mse, mad, snr, psnr 
 
def thresholdQR(QRImg):
    imgCpy = QRImg.copy()
    maxX = QRImg.shape[1]
    maxY = QRImg.shape[0]
    maxLum = np.max(QRImg)
    minLum = np.min(QRImg)
    threshold = minLum + ((maxLum - minLum) / 2)
    for y in range(maxY): 
        threshold = np.mean(imgCpy[y][:])
        for x in range(maxX):
            aboveThreshold = imgCpy[y][x] > threshold
            toSet = 0xff if aboveThreshold else 0
            imgCpy[y][x] = toSet
    return imgCpy

def thresholdQR(QRImg, size):
    imgCpy = QRImg.copy()
    maxX = QRImg.shape[1]
    maxY = QRImg.shape[0]
    maxLum = np.max(QRImg)
    minLum = np.min(QRImg)
    xBlocks = int(maxY / size)
    yBlocks = int(maxY / size)
    blocksMeans = np.zeros((yBlocks, xBlocks),dtype = float)
    for x in range(xBlocks):
        for y in range(yBlocks):
            block = imgCpy[y*size : y*size + size,x*size : x*size + size].copy()
            blocksMeans[y][x] = np.mean(block)
            
    blocksMeans = (blocksMeans / np.max(blocksMeans)) * 255
    for x in range(xBlocks):
        for y in range(yBlocks):
            val = blocksMeans[y][x]
            aboveThreshold = val > 85
            toSet = 255 if aboveThreshold else 0
            imgCpy[y*size : y*size + size, x*size : x*size + size] = toSet
    return imgCpy

def thresholdAdv(img, bitsPerPixel):
    imgCpy = img.copy()
    maxX = img.shape[1]
    maxY = img.shape[0]
    maxLum = np.max(img)
    minLum = np.min(img)
    levels = 2 ** bitsPerPixel
    diffPerLevel = (maxLum - minLum) / (levels - 1)
    endValuePerLevel = 255 / (levels - 1)
    for x in range(maxX):
        for y in range(maxY):
            pixVal = imgCpy[y][x]
            levelNumber = round(pixVal / diffPerLevel)
     
            imgCpy[y][x] = endValuePerLevel * levelNumber
    return imgCpy

def testAmpInfluence():
    amplificationLevels = np.arange(-40, -10+3, 3)
    length = len(amplificationLevels)
    lenaLOW = np.zeros(length, dtype=np.float32)
    qrLOW = np.zeros(length, dtype=np.float32)
    lenaHIGH = np.zeros(length, dtype=np.float32)
    qrHIGH = np.zeros(length, dtype=np.float32)
    lenaImg = cv2.imread("./testFiles/lena.png", cv2.IMREAD_GRAYSCALE)
    qrImg = cv2.imread("./testFiles/qr.png", cv2.IMREAD_GRAYSCALE)
    i = 0
    for amp in amplificationLevels:
        progress = (100 * i)/len(amplificationLevels)
        print(int(progress))
        mse, mad, snr, psnrQRLOW = doSingleCalc(5, 222, amp, qrImg)
        mse, mad, snr, psnrLenaLOW = doSingleCalc(5, 222, amp, lenaImg)
        mse, mad, snr, psnrQRHIGH = doSingleCalc(350, 222, amp, qrImg)
        mse, mad, snr, psnrLenaHIGH = doSingleCalc(350, 222, amp, lenaImg)
        lenaLOW[i] = psnrLenaLOW
        qrLOW[i] = psnrQRLOW
        lenaHIGH[i] = psnrLenaHIGH
        qrHIGH[i] = psnrQRHIGH
        #qrImg = thresholdQR(qrImg)
        #fileName = pathToResultFolder + "QR" + "(AMP_" + str(-amp) + ")" + ".png"
        #cv2.imwrite(fileName ,qrImg)
        i+=1
        

    fig = plt.figure()
    plt.title("Jakosć obrazu w zależnosci od wzmocnienia")
    plt.xlabel("wzmocnienie [dB]")
    plt.ylabel("PSNR [dB]")
    plt.plot(amplificationLevels, qrLOW, "-.", label="qr nisko")
    plt.plot(amplificationLevels, lenaLOW, "-.", label="lena nisko")
    plt.plot(amplificationLevels, qrHIGH, "-.", label="qr wysoko")
    plt.plot(amplificationLevels, lenaHIGH, "-.", label="lena wysoko")
    plt.legend()
    
def testThresholding():
    lenaImg = cv2.imread("./testFiles/lenaDestroyed.png", cv2.IMREAD_GRAYSCALE)
    imgsVect = np.arange(2,9)
    w = lenaImg.shape[1]
    h = lenaImg.shape[0]
    imgs = np.zeros((len(imgsVect),w,h),dtype = np.uint8)
    
    for i in imgsVect:
        img = thresholdAdv(lenaImg,i)
        cv2.imwrite("./res/lena_treshold_" + str(i) + ".png",img)
        imgs[i - 2] = img
        
    wholeImg = np.zeros((2*h,7 * w),dtype = np.uint8)
    for N in range(len(imgs)):
        for x in range(w):
            for y in range(h):
                wholeImg[y + h][N*w + x] = imgs[N][y][x]
                wholeImg[y][N*w + x] = lenaImg[y][x]
    cv2.imwrite("./res/whole.png",wholeImg)
    
    qrImg = cv2.imread("./testFiles/qrDestroyed.png", cv2.IMREAD_GRAYSCALE)
    qrImg2 = thresholdQR(qrImg,5)
    cv2.imwrite("./res/qrRepaired.png",qrImg2)




testThresholding()