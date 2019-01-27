'''
@author: ahguedes
'''
import numpy as np
import cv2
import utils

class Format():
    BGR = 0,
    RGB = 1

class MultiColorSpace(object):
    '''
    Skin Segmentation using a multi color space thresholding proposed in 10.1109/ICCOINS.2016.7783247
    '''

    def __init__(self, img, format = Format.RGB):
        '''
        Constructor
        '''
        if format != Format.BGR:
            self.inpuImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self.inpuImg = img
        
    def apply(self):
        kernel = np.ones((5,5),np.uint8)
        
        rgbNormalized = self.normalizeRGB(self.inpuImg)
        rgbSkinMask = self.skinRGBEval(rgbNormalized)
        #rgbSkinMask = cv2.dilate(rgbSkinMask.astype(np.float32), kernel, 10)
        #rgbSkinMask = cv2.erode(rgbSkinMask.astype(np.float32), kernel, 10)
        hsvSkinMask = self.skinHSVEval(rgbNormalized)
        #hsvSkinMask = cv2.erode(hsvSkinMask.astype(np.float32), kernel, 2)
        #hsvSkinMask = cv2.dilate(hsvSkinMask.astype(np.float32), kernel, 2)
        
        ycrcbSkinMask = self.skinYCrCbEval(self.inpuImg)
        #ycrcbSkinMask = cv2.dilate(ycrcbSkinMask.astype(np.float32), kernel, 10)
        #ycrcbSkinMask = cv2.erode(ycrcbSkinMask.astype(np.float32), kernel, 10)
      
        #utils.showImages([rgbSkinMask, hsvSkinMask, ycrcbSkinMask], ["1", "1", "1"], 1, 3)
      
        multiChannelMask = np.zeros(rgbSkinMask.shape, np.uint)  
        multiChannelMask[np.where(rgbSkinMask.astype(np.uint) & hsvSkinMask.astype(np.uint) & ycrcbSkinMask.astype(np.uint))] = 1
        multiChannelMask = np.repeat(multiChannelMask[:,:, np.newaxis], 3, axis=2)
        
        self.resultImg = np.copy(self.inpuImg)
        self.resultImg *= multiChannelMask
        
    def getResult(self):
        return self.resultImg
    
    def normalizeRGB(self, img):
        sumValue = img.sum(axis=2)
        normalizedImg = img.astype("float")
        sumValue = np.expand_dims(sumValue, axis=2)
        sumValue = np.repeat(sumValue, 3, 2)
        
        normalizedImg /= sumValue
        
        nanIdx = np.where(np.isnan(normalizedImg))
        normalizedImg[nanIdx[0], nanIdx[1], nanIdx[2]] = 0    
    
        return normalizedImg
    
    #Generate HSV from a float64 image
    def hsvFromNormalizedRgb(self, normalizedRGB):
        hsvImg = np.zeros(normalizedRGB.shape[0]*normalizedRGB.shape[1]*normalizedRGB.shape[2],
                        dtype=np.float64)
        hsvImg = hsvImg.reshape((normalizedRGB.shape[0],normalizedRGB.shape[1], normalizedRGB.shape[2]))
        
        #Value Channel
        hsvImg[:, :, 2] = np.amax(normalizedRGB, axis = 2)
        
        #Saturation Channel
        idx = np.where(hsvImg[:, :, 2] != 0)
        hsvImg[idx[0], idx[1], 1] = (
            hsvImg[idx[0], idx[1], 2] - np.amin(normalizedRGB, axis = 2)[idx[0], idx[1]]
            ) / hsvImg[idx[0], idx[1], 2]
        
        #Hue Channel
        idxVeqR = np.where(hsvImg[:, :, 2] == normalizedRGB[:,:,0])
        idxVeqG = np.where(hsvImg[:, :, 2] == normalizedRGB[:,:,1])
        idxVeqB = np.where(hsvImg[:, :, 2] == normalizedRGB[:,:,2]  )
        minValues = np.amin(normalizedRGB, axis = 2)
        
        #if V = R
        hsvImg[idxVeqR[0],idxVeqR[1],0] = (
            60*(normalizedRGB[idxVeqR[0], idxVeqR[1], 1] - normalizedRGB[idxVeqR[0], idxVeqR[1], 2])
            ) / (hsvImg[idxVeqR[0], idxVeqR[1], 2] - minValues[idxVeqR[0], idxVeqR[1]])
        
        #if V = G
        hsvImg[idxVeqG[0],idxVeqG[1],0] = 2 + (
            60*(normalizedRGB[idxVeqG[0], idxVeqG[1], 2] - normalizedRGB[idxVeqG[0], idxVeqG[1], 0])
            ) / (hsvImg[idxVeqG[0], idxVeqG[1], 2] - minValues[idxVeqG[0], idxVeqG[1]])
        
        #if V = B
        hsvImg[idxVeqB[0],idxVeqB[1],0] = 4 + (
            60*(normalizedRGB[idxVeqB[0], idxVeqB[1], 0] - normalizedRGB[idxVeqB[0], idxVeqB[1], 1])
            ) / (hsvImg[idxVeqB[0], idxVeqB[1], 2] - minValues[idxVeqB[0], idxVeqB[1]])
        
        negativeHueIdx = np.where(hsvImg[:,:,0] < 0)
        hsvImg[negativeHueIdx[0], negativeHueIdx[1]] += 360 
        
        return hsvImg
    
    def skinRGBEval(self, normalizedImg):
        eval = np.zeros(normalizedImg.shape[0]*normalizedImg.shape[1], dtype=np.float64)
        eval = eval.reshape((normalizedImg.shape[0], normalizedImg.shape[1]))
        eval2 = np.zeros(normalizedImg.shape[0]*normalizedImg.shape[1], dtype=np.float64)
        eval2 = eval2.reshape((normalizedImg.shape[0], normalizedImg.shape[1]))
        eval3 = np.zeros(normalizedImg.shape[0]*normalizedImg.shape[1], dtype=np.float64)
        eval3 = eval3.reshape((normalizedImg.shape[0], normalizedImg.shape[1]))
        
        #Eq 4...
        idx = np.where(normalizedImg[:,:,1]!=0)
        eval[idx[0],idx[1]] = normalizedImg[idx[0],idx[1],0] / normalizedImg[idx[0],idx[1],1]
        evalResult = eval > 1.185
        
        x = np.power(normalizedImg.sum(axis=2), 2) #para mi, sin sentido... pero si el paper lo dice
        idx = np.where(x != 0)
        eval2[idx[0], idx[1]] = (
            normalizedImg[idx[0],idx[1],0] * normalizedImg[idx[0],idx[1],2]
            ) / x[idx[0],idx[1]]
        eval2Result = eval2 > 0.107
        
        eval3[idx[0], idx[1]] = (
            normalizedImg[idx[0], idx[1], 0] * normalizedImg[idx[0], idx[1], 1]
            ) / x[idx[0], idx[1]]
        eval3Result = eval3 > 0.112
        
        finalResult = evalResult & eval2Result & eval3Result
        #skinIndex = np.where(finalResult)
        skinIndex = np.where(evalResult)
        
        mask = np.zeros(normalizedImg.shape[0]*normalizedImg.shape[1], dtype=np.uint)
        mask = mask.reshape((normalizedImg.shape[0],normalizedImg.shape[1]))
        mask[skinIndex[0], skinIndex[1]] = 1
        
        mask = mask.astype(np.uint)
    
        return mask
    
    def skinHSVEval(self, normalizedImg):
        hsvImg = self.hsvFromNormalizedRgb(normalizedImg)
        
        evalValChannel = hsvImg[:, :, 2] >= 0.4
        #Eq 12
        evalSatChannel = (hsvImg[:, :, 1] >= 0.2) & (hsvImg[:, :, 1] <= 0.6)
        #Eq 13
        evalHueChannel = (
                ((hsvImg[:, :, 0] >= 0) & (hsvImg[:, :, 0] <= 25)) |
                ((hsvImg[:, :, 0] >= 335) & (hsvImg[:, :, 1] <= 360))
            )
            
        #evalIdx = np.where(evalValChannel & evalSatChannel & evalHueChannel)
        evalIdx = np.where(evalSatChannel & evalHueChannel)
        
        mask = np.zeros(normalizedImg.shape[0]*normalizedImg.shape[1], dtype=np.uint)
        mask = mask.reshape((normalizedImg.shape[0], normalizedImg.shape[1]))
        mask[evalIdx[0], evalIdx[1]] = 1
        
        mask = mask.astype(np.uint)
        
        return mask
    
    def skinYCrCbEval(self, rgbImg):
        yCrCbImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2YCrCb)
        mask = np.zeros(rgbImg.shape[0]*rgbImg.shape[1], dtype=np.uint)
        mask = mask.reshape((rgbImg.shape[0], rgbImg.shape[1]))
        
        #Eq 17...
        crEval = (yCrCbImg[:,:,1] > 133) & (yCrCbImg[:,:,1] < 173)
        #Eq 18...
        cbEval = (yCrCbImg[:,:,2] > 77) & (yCrCbImg[:,:,2] < 127)
        evalIdx = np.where(crEval & cbEval)
        mask[evalIdx[0], evalIdx[1]] = 1
        
        mask = mask.astype(np.uint)
        
        return mask