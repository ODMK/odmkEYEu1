# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((odmkEYEuObj.py))::__
#
# Python ODMK img processing research
# ffmpeg experimental
#
#
# <<<PRIMARY-PARAMETERS>>>
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os, sys
from math import atan2, floor, ceil
import random
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance

rootDir = 'C:/odmkDev/odmkCode/odmkPython/'
audioScrDir = 'C:/odmkDev/odmkCode/odmkPython/audio/wavsrc/'
audioOutDir = 'C:/odmkDev/odmkCode/odmkPython/audio/wavout/'

#sys.path.insert(0, 'C:/odmkDev/odmkCode/odmkPython/util')
sys.path.insert(0, rootDir+'util')
from odmkClear import *

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(1, rootDir+'DSP')
import odmkClocks as clks
import odmkSigGen1 as sigGen

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
#sys.path.insert(2, rootDir+'audio')
#import native_wav as wavio

# temp python debugger - use >>>pdb.set_trace() to set break
import pdb



# // *---------------------------------------------------------------------* //
clear_all()

# // *---------------------------------------------------------------------* //

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# -----------------------------------------------------------------------------
# utility func
# -----------------------------------------------------------------------------


def cyclicZn(n):
    ''' calculates the Zn roots of unity '''
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)        # Define cyclic group Zn points
        cZn[k] = np.cos(((k)*2*np.pi)/n) + np.sin(((k)*2*np.pi)/n)*1j   # Euler's identity

    return cZn


def quarterCos(n):
    ''' returns a quarter period cosine wav of length n '''
    t = np.linspace(0, np.pi/4, n)
    qtrcos = np.zeros((n))
    for j in range(n):
        qtrcos[j] = sp.cos(t[j])
    return qtrcos

def randomIdx(k):
    '''returns a random index in range 0 - k-1'''
    randIdx = round(random.random()*(k-1))
    return randIdx

def randomIdxArray(n, k):
    '''for an array of k elements, returns a list of random elements
       of length n (n integers rangin from 0:k-1)'''
    randIdxArray = []
    for i in range(n):
        randIdxArray.append(round(random.random()*(k-1)))
    return randIdxArray
      
def randomPxlLoc(SzX, SzY):
    '''Returns a random location in a 2D array of SzX by SzY'''
    randPxlLoc = np.zeros(2)
    randPxlLoc[0] = int(round(random.random()*(SzX-1)))
    randPxlLoc[1] = int(round(random.random()*(SzY-1)))
    return randPxlLoc


class odmkWeightRnd:
    def __init__(self, weights):
        self.__max = .0
        self.__weights = []
        for value, weight in weights.items():
            self.__max += weight
            self.__weights.append((self.__max, value))

    def random(self):
        r = random.random() * self.__max
        for wtceil, value in self.__weights:
            if wtceil > r:
                return value


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : object definition
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 

class odmkEYEu:
    ''' # Python ODMK img processing research
        # ffmpeg experimental
        --xLength: length of output video/audio file
        --bpm: tempo of video/audio
        --timeSig: 4/4 or 3/4
        --SzX, SzY: video dimensions
        --framesPerSec: video frames per second
        --fs: audio sample rate
        --rootDir: root directory of Python code
        >>tbEYEu = eyeInst.odmkEYEu(eyeLength, 93, 30.0, 41000, rootDir)
    '''

    def __init__(self, xLength, bpm, timeSig, SzX, SzY, framesPerSec=30.0, fs=48000, cntrlEYE=0, rootDir='None'):

        # *---set primary parameters from inputs---*

        if rootDir != 'None':
            self.rootDir = rootDir
            os.makedirs(rootDir, exist_ok=True)
        else:
            self.rootDir = os.path.dirname(os.path.abspath(__file__))
        self.odmkEYEuOutDir = os.path.join(self.rootDir, "odmkEYEuOutDir/")
        os.makedirs(self.odmkEYEuOutDir, exist_ok=True)
        # myfile_path = os.path.join(self.rootDir, "myfile.txt")

        self.xLength = xLength
        self.fs = fs
        self.bWidth = 24         # audio sample bit width
        # ub313 bpm = 178 , 267
        self.bpm = bpm    #glamourGoat14 = 93        
        self.timeSig = timeSig         # time signature: 4 = 4/4; 3 = 3/4

        self.framesPerSec = framesPerSec
        
        self.cntrlEYE = cntrlEYE

        self.eyeDir = rootDir+'eye/eyeSrc/'
        os.makedirs(self.eyeDir, exist_ok=True)
        
        print('\nAn odmkEYEu object has been instanced with:')
        print('xLength = '+str(self.xLength)+'; bpm = '+str(self.bpm)+' ; timeSig = '+str(self.timeSig)+' ; framesPerSec = '+str(self.framesPerSec)+' ; fs = '+str(self.fs))

        print('\n')
        print('// *------------------------------------------------------* //')
        print('// *---::Set Master Dimensions::---*')
        print('// *------------------------------------------------------* //')
        
        
        self.mstrSzX = SzX
        self.mstrSzY = SzY        
        
        
        print('\nOutput frame width = '+str(self.mstrSzX))
        print('Output frame Heigth = '+str(self.mstrSzY))
        
        # // *-------------------------------------------------------------* //
        
        print('\n')
        print('// *------------------------------------------------------* //')
        print('// *---::Instantiate clocking/sequencing object::---*')
        print('// *------------------------------------------------------* //')
        
        self.eyeClks = clks.odmkClocks(self.xLength, self.fs, self.bpm, self.framesPerSec)
        
        # // *-------------------------------------------------------------* //

    
        # cntrlLoadX - Image loading controls
        # cntrlLoadX = <n>
        # <0> Bypass  => No images loaded
        # <1> Load    => choose raw image dir (a directory of random .jpg images)
        self.cntrlLoadX = 0
        if self.cntrlLoadX == 1:
            # jpgSrcDir: set the directory path used for processing .jpg files
            #jpgSrcDir = eyeDir+'totwnEyesSrc/'
            #jpgSrcDir = eyeDir+'odmkGorgulan56Scale1280x960/'    # pre-scaled or scaled img list
            #jpgSrcDir = eyeDir+'odmkGorgulan56Scale360x240/'
            self.jpgSrcDir = self.eyeDir+'odmkG1Src2/'
            #jpgSrcDir = eyeDir+'odmkGorgulanSrc/'

        self.cntrlScaleAll = 0
        # cntrlScaleAll - Image scale processing
        # <0> Bypass  => No scaling
        # <1> Scale
        #cntrlScaleAll = 0
        if self.cntrlScaleAll == 0:
            self.high = 0    # "head crop" cntrl - <0> => center crop, <1> crop from img top
            # *** Define Dir & File name for Scaling
            # scaleFileName: file name used for output processed .jpg (appended with 000X)
            self.scaleFileName = 'odmkGorgulan56'
            # scaleOutDirName: set the directory path used for output processed .jpg
            self.scaleOutDirName = 'odmkGorgulan56Scale360x240/'

        # defined at input (*****add method to update cntrlEYE??)
        # self.cntrlEYE = 21

        if self.cntrlEYE != 0:
            # eyeOutFileName: file name used for output processed .jpg (appended with 000X)
            #eyeOutFileName = 'eyeEchoGorgulan1v'
            self.eyeOutFileName = 'odmkGSrc1CEnhanceTsc'
            # eyeOutDirName: set the directory path used for output processed .jpg
            #eyeOutDirName = 'eyeEchoGorgulan1vDir/'
            self.eyeOutDirName = 'odmkGSrc1CEnhanceTscDir/'
        
        
        # <<<selects image post-processing function>>>
        # *-------------------------------------------------------------------*
        # 0  => Bypass
        # 1  => Image Random Select Algorithm              ::odmkImgRndSel::
        self.postProcess = 0
        
        if self.postProcess == 1:
            # repeatReverseAllImg:
            self.n = 2                         # number of repetitions (including 1st copy)
            self.srcDir = 'baliMaskYoukaiDir/'       # source image directory 
            self.eyeOutDirName = 'baliMaskYoukaiDirRpt/'    # output image directory 
            self.eyeOutFileName = 'baliMaskYoukai'         # output image name 
            
        elif self.postProcess == 2:
            # repeatReverseAllImg - operates on directory
            self.n = 2                         # number of repetitions (including 1st copy)
            self.srcDir = 'baliMaskYoukaiDir/'       # source image directory
            self.eyeOutDirName = 'baliMaskYoukaiDirRpt/'    # output image directory 
            self.eyeOutFileName = 'baliMaskYoukai'         # output image name
        
        

    # #########################################################################
    # end : object definition
    # #########################################################################

    # #########################################################################
    # begin : member method definition
    # #########################################################################


    # -------------------------------------------------------------------------
    # utility func
    # -------------------------------------------------------------------------

    def importAllJpg(self, srcDir):
        ''' imports all .jpg files into Python mem from source directory.
            srcDir should be Fullpath ending with "/" '''
    
        # generate list all files in a directory
        imgSrcList = []
        imgObjList = []
    
        print('\nLoading source images from the following path:')
        print(srcDir)
    
        for filename in os.listdir(srcDir):
            if (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
                imgSrcList.append(filename)
                imgPath = srcDir+filename
                imgObjTemp = misc.imread(imgPath)
                imgObjList.append(imgObjTemp)
        imgCount = len(imgSrcList)
        print('\nFound '+str(imgCount)+' images in the folder:\n')
        print(imgSrcList[0]+' ... '+imgSrcList[-1])
        # print('\nCreated python lists:')
        # print('<<imgObjList>> (ndimage objects)')
        # print('<<imgSrcList>> (img names)\n')
        # print('// *--------------------------------------------------------* //')
    
        return [imgObjList, imgSrcList]


    # -----------------------------------------------------------------------------
    # img Pre-processing func
    # -----------------------------------------------------------------------------
    
    # def odmkEyePrint(self, img, title, idx):
        # imgWidth = img.shape[1]
        # imgHeight = img.shape[0]
        # figx = plt.figure(num=idx, facecolor='silver', edgecolor='k')
        # plt.imshow(img)
        # plt.xlabel('Src Width = '+str(imgWidth))
        # plt.ylabel('Src Height = '+str(imgHeight))
        # plt.title(title)
    
    
    def odmkEyeRescale(self, img, SzX, SzY):
        # Rescale Image to SzW width and SzY height:
        # *****misc.imresize returns img with [Y-dim, X-dim] format instead of X,Y*****
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        imgRescale_r = misc.imresize(img_r, [SzY, SzX], interp='cubic')
        imgRescale_g = misc.imresize(img_g, [SzY, SzX], interp='cubic')
        imgRescale_b = misc.imresize(img_b, [SzY, SzX], interp='cubic')
        imgRescale = np.dstack((imgRescale_r, imgRescale_g, imgRescale_b))
        return imgRescale
    
    
    def odmkEyeZoom(self, img, Z):
        # Zoom Image by factor Z:
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        imgZoom_r = ndimage.interpolation.zoom(img_r, Z)
        imgZoom_g = ndimage.interpolation.zoom(img_g, Z)
        imgZoom_b = ndimage.interpolation.zoom(img_b, Z)
        imgZoom = np.dstack((imgZoom_r, imgZoom_g, imgZoom_b))
        return imgZoom
    
    
    # *---FIXIT---*
    
    #def odmkCenterPxl(img):
    #    imgWidth = img.shape[1]
    #    imgHeight = img.shape[0]
    #    cPxl_x = floor(imgWidth / 2)
    #    cPxl_y = floor(imgHeight / 2)
    #    centerPxl = [cPxl_x, cPxl_y]
    #    return centerPxl
    #
    #
    #def odmkHelixPxl(vLength, theta):
    #    helixPxlArray = []
    #    # generate a vector of 'center pixels' moving around unit circle
    #    # quantized cyclicZn coordinates!
    #    # theta = angle of rotation per increment
    #    return helixPxlArray
    #
    #
    #def odmkFiboPxl(vLength, theta):
    #    fiboPxlArray = []
    #    # generate a vector of 'center pixels' tracing a fibonnaci spiral
    #    # quantized fibonnaci coordinates!
    #    # theta = angle of rotation per increment
    #    return fiboPxlArray
    
    
    def odmkEyeCrop(self, img, SzX, SzY, high):
        ''' crop img to SzX width x SzY height '''
    
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
    
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
    
        imgCrop_r = np.zeros((SzY, SzX))
        imgCrop_g = np.zeros((SzY, SzX))
        imgCrop_b = np.zeros((SzY, SzX))
    
        wdiff_lhs = 0
        hdiff_ths = 0
    
        if imgWidth > SzX:
            wdiff = floor(imgWidth - SzX)
            if (wdiff % 2) == 1:    # wdiff is odd
                wdiff_lhs = floor(wdiff / 2)
                # wdiff_rhs = ceil(wdiff / 2)
            else:
                wdiff_lhs = wdiff / 2
                # wdiff_rhs = wdiff / 2
    
        if imgHeight > SzY:
            hdiff = floor(imgHeight - SzY)
            if (hdiff % 2) == 1:    # wdiff is odd
                hdiff_ths = floor(hdiff / 2)
                # hdiff_bhs = ceil(wdiff / 2)
            else:
                hdiff_ths = hdiff / 2
                # hdiff_bhs = hdiff / 2
    
        for j in range(SzY):
            for k in range(SzX):
                if high == 1:
                    imgCrop_r[j, k] = img_r[j, int(k + wdiff_lhs)]
                    imgCrop_g[j, k] = img_g[j, int(k + wdiff_lhs)]
                    imgCrop_b[j, k] = img_b[j, int(k + wdiff_lhs)]
                else:
                    # pdb.set_trace()
                    imgCrop_r[j, k] = img_r[int(j + hdiff_ths), int(k + wdiff_lhs)]
                    imgCrop_g[j, k] = img_g[int(j + hdiff_ths), int(k + wdiff_lhs)]
                    imgCrop_b[j, k] = img_b[int(j + hdiff_ths), int(k + wdiff_lhs)]
            # if j == SzY - 1:
                # print('imgItr ++')
        imgCrop = np.dstack((imgCrop_r, imgCrop_g, imgCrop_b))
        return imgCrop
    
    
    def odmkEyeDim(self, img, SzX, SzY, high):
        ''' Zoom and Crop image to match source dimensions '''
    
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
    
        #pdb.set_trace()
    
        if (imgWidth == SzX and imgHeight == SzY):
            # no processing
            imgScaled = img
    
        elif (imgWidth == SzX and imgHeight > SzY):
            # cropping only
            imgScaled = self.odmkEyeCrop(img, SzX, SzY, high)
    
        elif (imgWidth > SzX and imgHeight == SzY):
            # cropping only
            imgScaled = self.odmkEyeCrop(img, SzX, SzY, high)
    
        elif (imgWidth > SzX and imgHeight > SzY):
            # downscaling and cropping
            wRatio = SzX / imgWidth
            hRatio = SzY / imgHeight
            if wRatio >= hRatio:
                zoomFactor = wRatio
            else:
                zoomFactor = hRatio
            imgScaled = self.odmkEyeZoom(img, zoomFactor)
            imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)
    
        elif (imgWidth < SzX and imgHeight == SzY):
            # upscaling and cropping
            wdiff = SzX - imgWidth
            zoomFactor = (imgWidth + wdiff) / imgWidth
            imgScaled = self.odmkEyeZoom(img, zoomFactor)
            imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)
    
        elif (imgWidth == SzX and imgHeight < SzY):
            # upscaling and cropping
            hdiff = SzY - imgHeight
            zoomFactor = (imgHeight + hdiff) / imgHeight
            imgScaled = self.odmkEyeZoom(img, zoomFactor)
            imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)
    
        elif (imgWidth < SzX and imgHeight < SzY):
            # upscaling and cropping (same aspect ratio -> upscaling only)
            wRatio = SzX / imgWidth
            hRatio = SzY / imgHeight
            if wRatio >= hRatio:
                zoomFactor = wRatio
            else:
                zoomFactor = hRatio
            imgScaled = self.odmkEyeZoom(img, zoomFactor)
            imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)
    
        elif (imgWidth > SzX and imgHeight < SzY):
            # upscaling and cropping
            # wdiff = imgWidth - SzX
            hdiff = SzY - imgHeight
            zoomFactor = (imgHeight + hdiff) / imgHeight
            imgScaled = self.odmkEyeZoom(img, zoomFactor)
            imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)
    
        elif (imgWidth < SzX and imgHeight > SzY):
            # upscaling and cropping
            wdiff = SzX - imgWidth
            # hdiff = imgHeight - SzY
            zoomFactor = (imgWidth + wdiff) / imgWidth
            imgScaled = self.odmkEyeZoom(img, zoomFactor)
            imgScaled = self.odmkEyeCrop(imgScaled, SzX, SzY, high)
    
        return imgScaled
    
    
    def odmkScaleAll(self, srcObjList, SzX, SzY, w=0, high=0, outDir='None', outName='None'):
        ''' rescales and normalizes all .jpg files in image object list.
            scales, zooms, & crops images to dimensions SzX, SzY
            Assumes srcObjList of ndimage objects exists in program memory
            Typically importAllJpg is run first.
            if w == 1, write processed images to output directory'''
    
        # check write condition, if w=0 ignore outDir
        if w == 0 and outDir != 'None':
            print('Warning: write = 0, outDir ignored')
        if w == 1:
            if outDir == 'None':
                print('Error: outDir must be specified; processed img will not be saved')
                w = 0
            else:
                # If Dir does not exist, makedir:
                os.makedirs(outDir, exist_ok=True)
        if outName != 'None':
            imgScaledOutNm = outName
        else:
            imgScaledOutNm = 'odmkScaleAllOut'
    
        imgScaledNmArray = []
        imgScaledArray = []
    
        imgCount = len(srcObjList)
        # Find num digits required to represent max index
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for k in range(imgCount):
            imgScaled = self.odmkEyeDim(srcObjList[k], SzX, SzY, high)
            # auto increment output file name
            nextInc += 1
            zr = ''    # reset lead-zero count to zero each itr
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgScaledNm = imgScaledOutNm+strInc+'.jpg'
            if w == 1:
                imgScaledFull = outDir+imgScaledNm
                misc.imsave(imgScaledFull, imgScaled)
            imgScaledNmArray.append(imgScaledNm)
            imgScaledArray.append(imgScaled)
    
        if w == 1:    
            print('Saved Scaled images to the following location:')
            print(imgScaledFull)
    
        print('\nScaled all images in the source directory\n')
        print('Renamed files: "processScaled00X.jpg"')
    
        print('\nCreated numpy arrays:')
        print('<<imgScaledArray>> (img data objects)')
        print('<<imgScaledNmArray>> (img names)\n')
    
        print('\n')
    
        print('// *--------------------------------------------------------------* //')
    
        return [imgScaledArray, imgScaledNmArray]
    
    
    # // *********************************************************************** //
    # // *---::ODMK img Post-processing func::---*
    # // *********************************************************************** //
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - repeat list of files in directory n times::---*
    # // *--------------------------------------------------------------* //
    
    def repeatAllImg(self, srcDir, n, w=0, repeatDir='None', repeatName='None'):
        ''' repeats a list of .jpg images in srcDir n times,
            duplicates and renames all files in the source directory.
            The new file names are formatted for processing with ffmpeg
            if w = 1, write files into output directory.
            if concatDir specified, change output dir name.
            if concatName specified, change output file names'''
    
        # check write condition, if w=0 ignore outDir
        if w == 0 and repeatDir != 'None':
            print('Warning: write = 0, outDir ignored')
        if w == 1:
            if repeatDir == 'None':
                print('Warning: if w=1, repeatDir must be specified; processed img will not be saved')
                w = 0
            else:
                # If Dir does not exist, makedir:
                os.makedirs(repeatDir, exist_ok=True)
    
        [imgObjList, imgSrcList] = self.importAllJpg(srcDir)
    
        if repeatName != 'None':
            reName = repeatName
        else:
            reName = 'imgRepeat'
    
        imgRepeatObjList = []
        imgRepeatSrcList = []
        imgCount = len(imgObjList) * n
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(n):
            for j in range(len(imgObjList)):
                nextInc += 1
                zr = ''
                for h in range(n_digits - len(str(nextInc))):
                    zr += '0'
                strInc = zr+str(nextInc)
                imgRepeatNm = reName+strInc+'.jpg'
                # append new name to imgConcatSrcList
                imgRepeatSrcList.append(imgSrcList[j].replace(imgSrcList[j], imgRepeatNm))
                imgRepeatObjList.append(imgObjList[j])
                if w == 1:
                    imgRepeatNmFull = repeatDir+imgRepeatNm
                    misc.imsave(imgRepeatNmFull, imgRepeatObjList[j])
    
        return [imgRepeatObjList, imgRepeatSrcList]
        
        
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - repeat list of files in directory n times::---*
    # // *--------------------------------------------------------------* //
    
    def repeatReverseAllImg(self, srcDir, n, w=0, repeatReverseDir='None', repeatReverseName='None'):
        ''' repeats a list of .jpg images in srcDir in alternating reverse
            order, n times (n: including 1st copy of source).
            n=1 => copy of src. n=2 => 1st copy + reversed copy
            n>=3 => 1st copy + reversed copy + 1st copy ...
            duplicates and renames all files in the source directory.
            The new file names are formatted for processing with ffmpeg
            if w = 1, write files into output directory.
            if concatDir specified, change output dir name.
            if concatName specified, change output file names'''
    
        # check write condition, if w=0 ignore outDir
        if w == 0 and repeatReverseDir != 'None':
            print('Warning: write = 0, outDir ignored')
        if w == 1:
            if repeatReverseDir == 'None':
                print('Warning: if w=1, RepeatReverseDir must be specified; processed img will not be saved')
                w = 0
            else:
                # If Dir does not exist, makedir:
                os.makedirs(repeatReverseDir, exist_ok=True)
    
        [imgObjList, imgSrcList] = self.importAllJpg(srcDir)
    
        if repeatReverseName != 'None':
            reName = repeatReverseName
        else:
            reName = 'imgRepeat'
    
        imgRepeatReverseObjList = []
        imgRepeatReverseSrcList = []
        imgCount = len(imgObjList) * n
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(n):
            for j in range(len(imgObjList)):
                nextInc += 1
                zr = ''
                for h in range(n_digits - len(str(nextInc))):
                    zr += '0'
                strInc = zr+str(nextInc)
                imgRepeatReverseNm = reName+strInc+'.jpg'
                # append new name to imgConcatSrcList
                if (i % 2) == 1:
                    imgRepeatReverseSrcList.append(imgSrcList[j].replace(imgSrcList[(len(imgObjList)-1)-j], imgRepeatReverseNm))
                    imgRepeatReverseObjList.append(imgObjList[(len(imgObjList)-1)-j])
                else:
                    imgRepeatReverseSrcList.append(imgSrcList[j].replace(imgSrcList[j], imgRepeatReverseNm))
                    imgRepeatReverseObjList.append(imgObjList[j])                
                if w == 1:
                    imgRepeatReverseNmFull = repeatReverseDir+imgRepeatReverseNm
                    misc.imsave(imgRepeatReverseNmFull, imgRepeatReverseObjList[j])
    
        return [imgRepeatReverseObjList, imgRepeatReverseSrcList]    
        
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - concat all files in directory list::---*
    # // *--------------------------------------------------------------* //
    
    
    def concatAllDir(self, dirList, w=0, concatDir='None', concatName='None'):
        ''' Takes a list of directories containing .jpg images,
            renames and concatenates all files into new arrays.
            The new file names are formatted for processing with ffmpeg
            if w = 1, write files into output directory.
            if concatDir specified, change output dir name.
            if concatName specified, change output file names'''
    
        # check write condition, if w=0 ignore outDir
        if w == 0 and concatDir != 'None':
            print('Warning: write = 0, outDir ignored')
        if w == 1:
            if concatDir == 'None':
                print('Warning: if w=1, concatDir must be specified; processed img will not be saved')
                w = 0
            else:
                # If Dir does not exist, makedir:
                os.makedirs(concatDir, exist_ok=True)
        if concatName != 'None':
            reName = concatName
        else:
            reName = 'imgConcat'
    
        imgConcatObjList = []
        imgConcatSrcList = []
        imgCounttmp = 0
        cntOffset = 0
        for j in range(len(dirList)):
            # load jth directory all files
            [imgObjList, imgSrcList] = self.importAllJpg(dirList[j])
            cntOffset += imgCounttmp
            imgCounttmp = len(imgSrcList)
            # replace current names with concat output name
            n_digits = int(ceil(np.log10(imgCounttmp+cntOffset))) + 2
            nextInc = cntOffset
            for k in range(imgCounttmp):
                nextInc += 1
                zr = ''
                for h in range(n_digits - len(str(nextInc))):
                    zr += '0'
                strInc = zr+str(nextInc)
                imgConcatNm = reName+strInc+'.jpg'
                # append new name to imgConcatSrcList
                imgConcatSrcList.append(imgSrcList[k].replace(imgSrcList[k], imgConcatNm))
                imgConcatObjList.append(imgObjList[k])
    
        # normalize image index n_digits to allow for growth
        imgCount = len(imgConcatSrcList)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(imgCount):
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            # print('strInc = '+str(strInc))
            imgNormalizeNm = reName+strInc+'.jpg'
            # print('imgConcatSrcList[i] = '+str(imgConcatSrcList[i]))
            imgConcatSrcList[i] = imgConcatSrcList[i].replace(imgConcatSrcList[i], imgNormalizeNm)
            # print('imgConcatSrcList[i] = '+str(imgConcatSrcList[i]))
            if w == 1:
                imgConcatNmFull = concatDir+imgNormalizeNm
                misc.imsave(imgConcatNmFull, imgConcatObjList[i])
    
        return [imgConcatObjList, imgConcatSrcList]
    
    
    # // *********************************************************************** //
    # // *---::ODMK img Pixel-Banging Algorithms::---*
    # // *********************************************************************** //

    # <<<selects pixel banging algorithm>>>
    # *---------------------------------------------------------------------------*
    # 0  => Bypass
    # 1  => EYE Random Select Algorithm              ::odmkImgRndSel::
    # 2  => EYE Random BPM Algorithm                 ::odmkImgRndSel::
    # 3  => EYE Random Select Telescope Algorithm    ::**nofunc**::
    # 4  => EYE BPM rotating Sequence Algorithm      ::odmkImgRotateSeq::
    # 5  => EYE Rotate & alternate sequence          ::**nofunc**::
    # 6  => EYE CrossFade Sequencer (alpha-blend)    ::odmkImgXfade::
    # 67 => EYE CrossFade Sequencer (solarize)       ::odmkImgSolXfade::
    # 7  => EYE Divide CrossFade Sequencer           ::odmkImgDivXfade::
    # 8  => EYE CrossFade Rotate sequence            ::odmkImgXfadeRot::
    # 9  => EYE telescope sequence                   ::odmkImgTelescope::
    # 10 => EYE CrossFade telescope Sequence         ::odmkImgXfadeTelescope::
    # 11 => EYE Div Xfade telescope Sequence         ::odmkImgDivXfadeTelescope::
    # 12 => Eye Echo BPM Sequencer                   ::odmkEyeEchoBpm::
    # 20 => EYE Color Enhance BPM Algorithm          ::odmkImgColorEnhance::
    # 21 => EYE Color Enhance Telescope Algorithm    ::odmkImgCEnTelescope::
    # 22 => EYE Pixel Random Replace Algorithm       ::odmkPxlRndReplace::
    # 23 => EYE Dual Bipolar Telescope Algorithm     ::odmkDualBipolarTelesc::
    # 24 => EYE Bananasloth Recursive Algorithm      ::odmkBSlothRecurs::



    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE image array processing prototype::---*
    # // *--------------------------------------------------------------* //
    
    def odmkEYEprototype(self, imgArray, xLength, framesPerSec, FrameCtrl, imgOutDir, imgOutNm='None', usrCtrl=0):
        ''' basic prototype function '''
    
        if imgOutNm != 'None':
            imgPrototypeNm = imgOutNm
        else:
            imgPrototypeNm = 'imgXfadeRot'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / FrameCtrl))
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---FrameCtrl = '+str(FrameCtrl)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgArray)
        numxfadeImg = numImg * FrameCtrl
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        alphaX = np.linspace(0.0, 1.0, FrameCtrl)
    
        imgPrototypeNmArray = []
        imgPrototypeArray = []
    
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades - 1):
            if frameCnt <= numFrames:    # process until end of total length
            
                newDimX = SzX
                newDimY = SzY
                
                # example cloned img
                imgClone1 = imgArray[j]
    
                img = Image.new( 'RGB', (255,255), "black") # create a new black image
                imgPxl = img.load() # create the pixel map
    
                # scroll forward -> backward -> forward... through img array
                if (frameCnt % numxfadeImg) == 0:
                    xfdirection = -xfdirection
                if xfdirection == 1:
                    imgPIL1 = Image.fromarray(imgArray[j % (numImg-1)])
                    imgPIL2 = Image.fromarray(imgArray[(j % (numImg-1)) + 1])
                else:
                    imgPIL1 = Image.fromarray(imgArray[(numImg-2) - (j % numImg) + 1])
                    imgPIL2 = Image.fromarray(imgArray[(numImg-2) - (j % numImg)])
    
                # process frameCtrl timed sub-routine:
                for i in range(FrameCtrl):
    
                    # example usrCtrl update                
                    if newDimX > 2:
                        newDimX -= 2*hop_sz
                    if newDimY > 2:
                        newDimY -= 2*hop_sz
                    
                    # example algorithmic process
                    alphaB = Image.blend(imgPIL1, imgPIL2, alphaX[i])
    
                    # process sub-frame
                    for j in range(SzY):
                        for k in range(SzX):
                            #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
                                imgClone1[j, k, :] = alphaB[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
    
                    # ***sort out img.size vs. imgList[0].shape[1]
                    # bit-banging
                    for i in range(img.size[0]):    # for every pixel:
                        for j in range(img.size[1]):
                            imgPxl[i,j] = (i, j, 1) # set the colour accordingly
    
    
                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    imgPrototypeFull = imgOutDir+imgPrototypeNm+strInc+'.jpg'
                    
                    # write img to disk
                    misc.imsave(imgPrototypeFull, alphaB)
                    
                    # optional write to internal buffers
                    imgPrototypeNmArray.append(imgPrototypeFull)
                    imgPrototypeArray.append(alphaB)
                    frameCnt += 1
    
        # return
        return [imgPrototypeArray, imgPrototypeNmArray]
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Random Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgRndSel(self, imgList, numFrames, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgRndSelNm = imgOutNm
        else:
            imgRndSelNm = 'imgRndSelOut'
    
        imgRndSelNmArray = []
        imgRndSelArray = []
        
        rIdxArray = randomIdxArray(numFrames, len(imgList))
        
        imgCount = numFrames
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(numFrames):
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgNormalizeNm = imgRndSelNm+strInc+'.jpg'
            imgRndSelFull = imgOutDir+imgNormalizeNm
            misc.imsave(imgRndSelFull, imgList[rIdxArray[i]])
            
        return [imgRndSelArray, imgRndSelNmArray]
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image BPM Random Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgRndBpm(self, imgList, numFrames, frameHold, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgRndSelNm = imgOutNm
        else:
            imgRndSelNm = 'imgRndSelOut'
    
        imgRndSelNmArray = []
        imgRndSelArray = []
        
        rIdxArray = randomIdxArray(numFrames, len(imgList))
        
        imgCount = numFrames
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        k = 0
        for i in range(numFrames):
            if frameHold[i] == 1:
                k += 1
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgNormalizeNm = imgRndSelNm+strInc+'.jpg'
            imgRndSelFull = imgOutDir+imgNormalizeNm
            misc.imsave(imgRndSelFull, imgList[rIdxArray[k]])
            
        return [imgRndSelArray, imgRndSelNmArray]
        
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Rotate Sequence Algorithm::---*')
    # // *--------------------------------------------------------------* //
    
    def odmkImgRotateSeq(self, imgSrc, numFrames, imgOutDir, imgOutNm='None', rotDirection=0):
        ''' outputs a sequence of rotated images (static img input)
            360 deg - period = numFrames
            (ndimage.rotate: ex, rotate image by 45 deg:
             rotate_gz1 = ndimage.rotate(gz1, 45, reshape=False)) '''
    
        if imgOutNm != 'None':
            imgRotateSeqNm = imgOutNm
        else:
            imgRotateSeqNm = 'imgRotateSeqOut'
    
        zn = cyclicZn(numFrames-1)    # less one, then repeat zn[0] for full 360
    
        imgRotNmArray = []
        imgRotArray = []
    
        imgCount = numFrames
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(numFrames):
            ang = (atan2(zn[i % (numFrames-1)].imag, zn[i % (numFrames-1)].real))*180/np.pi
            if rotDirection == 1:
                ang = -ang
            rotate_gz1 = ndimage.rotate(imgSrc, ang, reshape=False)
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgRotateSeqFull = imgOutDir+imgRotateSeqNm+strInc+'.jpg'
            misc.imsave(imgRotateSeqFull, rotate_gz1)
            imgRotNmArray.append(imgRotateSeqFull)
            imgRotArray.append(rotate_gz1)
    
        return [imgRotArray, imgRotNmArray]
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image CrossFade Sequence Algorithm::---*')
    # // *--------------------------------------------------------------* //    
        
    def odmkImgXfade(self, imgList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of images fading from img1 -> img2 for xLength seconds
            Linearly alpha-blends images using PIL Image.blend
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgXfadeNm = imgOutNm
        else:
            imgXfadeNm = 'ImgXfade'
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---xfadeFrames = '+str(xfadeFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgList)
        numxfadeImg = numImg * xfadeFrames
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
    
        imgXfadeNmArray = []
        imgXfadeArray = []
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades - 1):
            if frameCnt <= numFrames:
                if (frameCnt % numxfadeImg) == 0:
                    xfdirection = -xfdirection
                if xfdirection == 1:
                    imgPIL1 = Image.fromarray(imgList[j % (numImg-1)])
                    imgPIL2 = Image.fromarray(imgList[(j % (numImg-1)) + 1])
                else:
                    imgPIL1 = Image.fromarray(imgList[(numImg-2) - (j % numImg) + 1])
                    imgPIL2 = Image.fromarray(imgList[(numImg-2) - (j % numImg)])
                for i in range(xfadeFrames):
                    alphaB = Image.blend(imgPIL1, imgPIL2, alphaX[i])
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    imgXfadeFull = imgOutDir+imgXfadeNm+strInc+'.jpg'
                    misc.imsave(imgXfadeFull, alphaB)
                    imgXfadeNmArray.append(imgXfadeFull)
                    imgXfadeArray.append(alphaB)
                    frameCnt += 1
    
        return [imgXfadeArray, imgXfadeNmArray]
        
        
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image CrossFade Sequence Algorithm::---*')
    # // *--------------------------------------------------------------* //    
        
    def odmkImgXfadeAll(self, imgList, xfadeFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of images fading from img1 -> img2
            for all images in src directory. 
            The final total frame count is determined by #src images * xfadeFrames (?check?)
            Linearly alpha-blends images using PIL Image.blend
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgXfadeNm = imgOutNm
        else:
            imgXfadeNm = 'imgRotateSeqOut'
    
        numImg = len(imgList)
        imgCount = xfadeFrames * (numImg - 1)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
    
        imgXfadeNmArray = []
        imgXfadeArray = []
    
        for j in range(numImg - 1):
            imgPIL1 = Image.fromarray(imgList[j])
            imgPIL2 = Image.fromarray(imgList[j + 1])
            for i in range(xfadeFrames):
                alphaB = Image.blend(imgPIL1, imgPIL2, alphaX[i])
                nextInc += 1
                zr = ''
                for j in range(n_digits - len(str(nextInc))):
                    zr += '0'
                strInc = zr+str(nextInc)
                imgXfadeFull = imgOutDir+imgXfadeNm+strInc+'.jpg'
                misc.imsave(imgXfadeFull, alphaB)
                imgXfadeNmArray.append(imgXfadeFull)
                imgXfadeArray.append(alphaB)
    
        return [imgXfadeArray, imgXfadeNmArray]    
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image solarize CrossFade Sequence::---*')
    # // *--------------------------------------------------------------* //    
        
    def odmkImgSolXfade(self, imgList, numFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of images fading from img1 -> img2
            Linearly alpha-blends images using PIL Image.blend
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgXfadeNm = imgOutNm
        else:
            imgXfadeNm = 'imgSolXfadeOut'
    
        numImg = len(imgList)
        imgCount = numFrames * (numImg - 1)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        solarX = np.linspace(0.0, 255.0, numFrames)
    
        imgXfadeNmArray = []
        imgXfadeArray = []
    
        for j in range(numImg):
            imgPIL1 = Image.fromarray(imgList[j])
            # imgPIL2 = Image.fromarray(imgList[j + 1])
            for i in range(numFrames):
                solarB = ImageOps.solarize(imgPIL1, solarX[i])
                nextInc += 1
                zr = ''
                for j in range(n_digits - len(str(nextInc))):
                    zr += '0'
                strInc = zr+str(nextInc)
                imgXfadeFull = imgOutDir+imgXfadeNm+strInc+'.jpg'
                misc.imsave(imgXfadeFull, solarB)
                imgXfadeNmArray.append(imgXfadeFull)
                imgXfadeArray.append(solarB)
    
        return [imgXfadeArray, imgXfadeNmArray]
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Division CrossFade Sequence Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgDivXfade(self, imgList, numFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of images fading from img1 -> img2
            Experimental Division Blending images using numpy
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgDivXfadeNm = imgOutNm
        else:
            imgDivXfadeNm = 'imgDivXfade'
    
        numImg = len(imgList)
        imgCount = numFrames * (numImg - 1)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        alphaX = np.linspace(0.0001, 1.0, numFrames)
    
        imgDivXfadeNmArray = []
        imgDivXfadeArray = []
    
        for j in range(numImg - 1):
            imgPIL1 = imgList[j]
            imgPIL2 = imgList[j + 1]
            for i in range(numFrames):
                # image division blend algorithm..
                c = imgPIL1/((imgPIL2.astype('float')+1)/(256*alphaX[i]))
                # saturating function - if c[m,n] > 255, set to 255:
                imgDIVB = c*(c < 255)+255*np.ones(np.shape(c))*(c > 255)
                nextInc += 1
                zr = ''
                for j in range(n_digits - len(str(nextInc))):
                    zr += '0'
                strInc = zr+str(nextInc)
                imgDivXfadeFull = imgOutDir+imgDivXfadeNm+strInc+'.jpg'
                misc.imsave(imgDivXfadeFull, imgDIVB)
                imgDivXfadeNmArray.append(imgDivXfadeFull)
                imgDivXfadeArray.append(imgDIVB)
    
        return [imgDivXfadeArray, imgDivXfadeNmArray]
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image CrossFade Rotate Sequence Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgXfadeRot(self, imgList, sLength, framesPerSec, xFrames, imgOutDir, imgOutNm='None', rotDirection=0):
        ''' outputs a sequence of images fading from img1 -> img2
            Linearly alpha-blends images using PIL Image.blend
            imgList: python list of img objects
            sLength the lenth of output sequence (total number of frames)
            framesPerSec: video frames per second
            xFrames: number of frames for each cross-fade (framesPer(beat/sec))
            imgOutDir: output directory path/name
            rotation direction: 0=counterclockwise ; 1=clockwise
            assume images in imgList are numpy arrays '''
    
        if imgOutNm != 'None':
            imgXfadeRotNm = imgOutNm
        else:
            imgXfadeRotNm = 'imgXfadeRot'
    
        zn = cyclicZn(xFrames-1)    # less one, then repeat zn[0] for full 360
    
        numFrames = int(ceil(sLength * framesPerSec))
        numXfades = int(ceil(numFrames / xFrames))
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---xFrames = '+str(xFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgList)
        numxfadeImg = numImg * xFrames
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        alphaX = np.linspace(0.0, 1.0, xFrames)
    
        imgXfadeRotNmArray = []
        imgXfadeRotArray = []
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades - 1):
            if frameCnt <= numFrames:
                if (frameCnt % numxfadeImg) == 0:
                    xfdirection = -xfdirection
                if xfdirection == 1:
                    imgPIL1 = Image.fromarray(imgList[j % (numImg-1)])
                    imgPIL2 = Image.fromarray(imgList[(j % (numImg-1)) + 1])
                else:
                    imgPIL1 = Image.fromarray(imgList[(numImg-2) - (j % numImg) + 1])
                    imgPIL2 = Image.fromarray(imgList[(numImg-2) - (j % numImg)])
                for i in range(xFrames):
                    alphaB = Image.blend(imgPIL1, imgPIL2, alphaX[i])
                    ang = (atan2(zn[i % (xFrames-1)].imag, zn[i % (xFrames-1)].real))*180/np.pi
                    if rotDirection == 1:
                        ang = -ang
                    rotate_alphaB = ndimage.rotate(alphaB, ang, reshape=False)
        
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    imgXfadeRotFull = imgOutDir+imgXfadeRotNm+strInc+'.jpg'
                    misc.imsave(imgXfadeRotFull, rotate_alphaB)
                    imgXfadeRotNmArray.append(imgXfadeRotFull)
                    imgXfadeRotArray.append(alphaB)
                    frameCnt += 1
    
        return [imgXfadeRotArray, imgXfadeRotNmArray]
        
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Telescope Sequence Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgTelescope(self, imgList, xLength, framesPerSec, framesPerBeat, imgOutDir, imgOutNm='None', inOrOut=0):
        ''' outputs a sequence of telescoping images (zoom in or zoom out)
            The period of telescoping sequence synched to framesPerBeat
            assumes images in imgList are normalized (scaled/cropped) numpy arrays '''
    
    
        if imgOutNm != 'None':
            imgTelescNm = imgOutNm
        else:
            imgTelescNm = 'imgTelescope'
    
        # find number of source images in src dir
        numSrcImg = len(imgList)
        # initialize SzX, SzY to dimensions of src img
        SzX = imgList[0].shape[1]
        SzY = imgList[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numBeats = int(ceil(numFrames / framesPerBeat))         
    
        hop_sz = 4*ceil(np.log(SzX/framesPerBeat))  # number of pixels to scale img each iteration
    
        #imgTelescopeNmArray = []
        #imgTelescopeArray = []
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        for i in range(numBeats):
            newDimX = SzX
            newDimY = SzY
            imgClone = imgList[i%numSrcImg]    # mod to rotate through src img
            for t in range(framesPerBeat):
                if newDimX > 2:
                    #newDimX -= 2*hop_sz
                    newDimX -= hop_sz
                if newDimY > 2:
                    #newDimY -= 2*hop_sz
                    newDimY -= hop_sz
                # scale image to new dimensions
                imgItr = self.odmkEyeRescale(imgClone, newDimX, newDimY)
                # region = (left, upper, right, lower)
                # subbox = (i + 1, i + 1, newDimX, newDimY)
                for j in range(SzY):
                    for k in range(SzX):
                        #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
                        if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                            #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
                            imgClone[j, k, :] = imgItr[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
                nextInc += 1
                if nextInc < numFrames:
                    zr = ''
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        for j in range(n_digits - len(str(numFrames - (nextInc)))):
                            zr += '0'
                        strInc = zr+str(numFrames - (nextInc))
                    imgTelescFull = imgOutDir+imgTelescNm+strInc+'.jpg'
                    misc.imsave(imgTelescFull, imgClone)
                else:
                    break


#    def odmkImgTelescope(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0):
#        ''' DualBipolarTelesc function '''
#    
#        if imgOutNm != 'None':
#            imgDualBipolarTelescNm = imgOutNm
#        else:
#            imgDualBipolarTelescNm = 'imgDualBipolarTelesc'
#    
#        SzX = imgArray[0].shape[1]
#        SzY = imgArray[0].shape[0]
#    
#        numFrames = int(ceil(xLength * framesPerSec))
#        numXfades = int(ceil(numFrames / xfadeFrames))
#    
#        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
#        print('// *---numFrames = '+str(numFrames)+'---*')
#        print('// *---numxfadeImg = '+str(xfadeFrames)+'---*')
#        print('// *---numXfades = '+str(numXfades)+'---*')    
#    
#        numImg = len(imgArray)
#    
#        n_digits = int(ceil(np.log10(numFrames))) + 2
#        nextInc = 0
#        # example internal control signal 
#        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
#    
#        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
#        imgRndIdxArray = randomIdxArray(numFrames, numImg)
#   
#        # imgDualBipolarTelescNmArray = []
#        # imgDualBipolarTelescArray = []
# 
#       
#    
#        frameCnt = 0
#        xfdirection = 1    # 1 = forward; -1 = backward
#        for j in range(numXfades):
#                                # random img grap from image Array
#
#            if (frameCnt % numXfades) == 0:
#                xfdirection = -xfdirection
#            if xfdirection == 1:
#                imgClone1 = imgArray[imgRndIdxArray[j]]
#                imgClone2 = imgArray[imgRndIdxArray[j+1]]                
#            else:
#                imgClone1 = imgArray[imgRndIdxArray[(numImg-2) - j]]
#                imgClone2 = imgArray[imgRndIdxArray[(numImg-2) - j+1]]
#                
#            # initialize output 
#            imgBpTsc1 = imgClone1
#            imgBpTsc2 = imgClone2
#            imgBpTscOut = imgClone1
#
#            newDimX = SzX     # start with master img dimensions
#            newDimY = SzY
#                              
#            # calculate X-Y random focal point to launch telescope
#            # focalRnd1 = randomPxlLoc(SzX, SzY)
#            # focalRnd2 = randomPxlLoc(SzX, SzY)
#
#
#            # functionalize then call for each telescope    
#            for t in range(xfadeFrames):
#                if frameCnt < numFrames:    # process until end of total length
#
#                    if newDimX > hop_sz:
#                        #newDimX -= 2*hop_sz
#                        newDimX -= hop_sz
#                    if newDimY > hop_sz:
#                        #newDimY -= 2*hop_sz
#                        newDimY -= hop_sz
#                        
#                    # scale image to new dimensions
#                    imgItr1 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
#                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
#
#
#                    # region = (left, upper, right, lower)
#                    # subbox = (i + 1, i + 1, newDimX, newDimY)
#                    for j in range(SzY):
#                        for k in range(SzX):
#                            
#                            # calculate bipolar subframes then interleave (out = priority)
#                            # will require handling frame edges (no write out of frame)
#                            # update index math to handle random focal points
#                            
#                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
#                                #if (j == 853 or int(j - (SzY_v-newDimY)/2) == 853):
#                                #print('*****j = '+str(j)+'; k = '+str(k)+'; imgItrY = '+str(int(j - (SzY_v-newDimY)/2))+'; imgItrX = '+str(int(k - (SzX_v-newDimX)/2)))
#                                #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
#                                imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
#                                imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
#
#                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
#
#                    # update name counter & concat with base img name    
#                    nextInc += 1
#                    zr = ''
#    
#                    # rewrite inOrOut as bipolar toggle switch in loop above
#                    if inOrOut == 1:
#                        for j in range(n_digits - len(str(nextInc))):
#                            zr += '0'
#                        strInc = zr+str(nextInc)
#                    else:
#                        for j in range(n_digits - len(str(numFrames - (nextInc)))):
#                            zr += '0'
#                        strInc = zr+str(numFrames - (nextInc))
#                    imgDualBipolarTelescFull = imgOutDir+imgDualBipolarTelescNm+strInc+'.jpg'
#                    misc.imsave(imgDualBipolarTelescFull, imgBpTscOut)
#    
#                    # optional write to internal buffers
#                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
#                    # imgDualBipolarTelescArray.append(imgBpTsc)
#                    
#                    frameCnt += 1
#    
#        return
#        # return [imgDualBipolarTelescArray, imgDualBipolarTelescNmArray]
   
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Img Xfade Telescope Sequence Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgXfadeTelescope(self, imgList, framesPerBeat, imgOutDir, inOut=0, imgOutNm='None'):
        ''' outputs a sequence of telescoping images (zoom in or zoom out)
            The period of telescoping sequence synched to framesPerBeat
            assumes images in imgList are normalized (scaled/cropped) numpy arrays '''
    
        if imgOutNm != 'None':
            imgTelescNm = imgOutNm
        else:
            imgTelescNm = 'imgTelescope'
    
        # find number of source images in src dir
        numSrcImg = len(imgList)
        # initialize SzX, SzY to dimensions of src img
        SzX = imgList[0].shape[1]
        SzY = imgList[0].shape[0]
    
        numFrames = numSrcImg * framesPerBeat
    
        alphaX = np.linspace(0.0, 1.0, framesPerBeat)               
    
        hop_sz = ceil(np.log(SzX/framesPerBeat))  # number of pixels to scale img each iteration
    
        #imgTelescopeNmArray = []
        #imgTelescopeArray = []
    
        imgCount = numFrames    # counter to increment output name
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(numSrcImg):
            imgClone1 = imgList[i]
            imgClone2 = imgList[(i + 1) % numSrcImg]  # final img+1 -> first img
            newDimX = SzX
            newDimY = SzY
            for t in range(framesPerBeat):
                if newDimX > 2:
                    newDimX -= 2*hop_sz
                if newDimY > 2:
                    newDimY -= 2*hop_sz
                # scale image to new dimensions
                imgItr1 = Image.fromarray(self.odmkEyeRescale(imgClone1, newDimX, newDimY))
                imgItr2 = Image.fromarray(self.odmkEyeRescale(imgClone2, newDimX, newDimY))
                alphaB = Image.blend(imgItr1, imgItr2, alphaX[t])
                alphaX = np.array(alphaB)
                # region = (left, upper, right, lower)
                # subbox = (i + 1, i + 1, newDimX, newDimY)
                for j in range(SzY):
                    for k in range(SzX):
                        #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
                        if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                            #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
                            imgClone1[j, k, :] = alphaX[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
                nextInc += 1
                zr = ''
                if inOrOut == 1:
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                else:
                    for j in range(n_digits - len(str(imgCount - (nextInc)))):
                        zr += '0'
                    strInc = zr+str(imgCount - (nextInc))
                imgTelescFull = imgOutDir+imgTelescNm+strInc+'.jpg'
                misc.imsave(imgTelescFull, imgClone1)
                
                
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Img DivXfade Telescope Sequence Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgDivXfadeTelescope(self, imgList, framesPerBeat, imgOutDir, inOut=0, imgOutNm='None'):
        ''' outputs a sequence of telescoping images (zoom in or zoom out)
            The period of telescoping sequence synched to framesPerBeat
            assumes images in imgList are normalized (scaled/cropped) numpy arrays '''
    
        if imgOutNm != 'None':
            imgTelescNm = imgOutNm
        else:
            imgTelescNm = 'imgTelescope'
    
        # find number of source images in src dir
        numSrcImg = len(imgList)
        # initialize SzX, SzY to dimensions of src img
        SzX = imgList[0].shape[1]
        SzY = imgList[0].shape[0]
    
        numFrames = numSrcImg * framesPerBeat
    
        alphaX = np.linspace(0.0001, 1.0, framesPerBeat)               
    
        hop_sz = ceil(np.log(SzX/framesPerBeat))  # number of pixels to scale img each iteration
    
        #imgTelescopeNmArray = []
        #imgTelescopeArray = []
    
        imgCount = numFrames    # counter to increment output name
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(numSrcImg):
            imgClone1 = imgList[i]
            imgClone2 = imgList[(i + 1) % numSrcImg]  # final img+1 -> first img
            newDimX = SzX
            newDimY = SzY
            for t in range(framesPerBeat):
                if newDimX > 2:
                    newDimX -= 2*hop_sz
                if newDimY > 2:
                    newDimY -= 2*hop_sz
                # scale image to new dimensions
                imgItr1 = self.odmkEyeRescale(imgClone1, newDimX, newDimY)
                imgItr2 = self.odmkEyeRescale(imgClone2, newDimX, newDimY)
                # image division blend algorithm..
                c = imgItr1/((imgItr2.astype('float')+1)/(256*alphaX[t]))
                # saturating function - if c[m,n] > 255, set to 255:
                imgDIVB = c*(c < 255)+255*np.ones(np.shape(c))*(c > 255)
                # region = (left, upper, right, lower)
                # subbox = (i + 1, i + 1, newDimX, newDimY)
                for j in range(SzY):
                    for k in range(SzX):
                        #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
                        if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                            #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
                            imgClone1[j, k, :] = imgDIVB[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
                nextInc += 1
                zr = ''
                if inOrOut == 1:
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                else:
                    for j in range(n_digits - len(str(imgCount - (nextInc)))):
                        zr += '0'
                    strInc = zr+str(imgCount - (nextInc))
                imgTelescFull = imgOutDir+imgTelescNm+strInc+'.jpg'
                misc.imsave(imgTelescFull, imgClone1) 
                
        return
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE ECHO Bpm Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkEyeEchoBpm(self, imgList, numFrames, echoTrig, numEcho, echoStep, imgOutDir, echoDecay = 'None', imgOutNm='None'):
        ''' Outputs a random sequence of img with downbeat echoes repeat images
            numFrames => total # frames
            echoTrig => trigger echoes [1D array of 0 & 1] - length = numFrames
            numEcho => number of repeats each downbeat
            echoStep => number of frames between each echo
            echoDecay => 'None': no decay ; 'cos': cosine decay
        '''
    
        if imgOutNm != 'None':
            eyeEchoBpmNm = imgOutNm
        else:
            eyeEchoBpmNm = 'eyeEchoBpm'
    
        # eyeEchoBpmNmArray = []
        # eyeEchoBpmArray = []
        
        rIdxArray = randomIdxArray(numFrames, len(imgList))
        rIdxEchoArray = randomIdxArray(numFrames, len(imgList))
        
        if echoDecay == 'cos':
            qtrcos = quarterCos(numEcho)    # returns a quarter period cosine wav of length n    
        
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        k = 0
        echoFrames = 0
        for i in range(numFrames):
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgNormalizeNm = eyeEchoBpmNm+strInc+'.jpg'
            imgRndSelFull = imgOutDir+imgNormalizeNm
            if echoTrig[i] == 1:
                echoImgIdx = rIdxEchoArray[k]
                misc.imsave(imgRndSelFull, imgList[echoImgIdx])
                echoFrames = (numEcho + 1) * echoStep
                k += 1
                h = 1
            elif (h <= echoFrames and (h % echoStep) == 0):
                misc.imsave(imgRndSelFull, imgList[echoImgIdx])
                h += 1
            else:
                misc.imsave(imgRndSelFull, imgList[rIdxArray[i]])
                h += 1
            
        # return [eyeEchoBpmArray, eyeEchoBpmNmArray]
        return      
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Color Enhance::---*')
    # // *--------------------------------------------------------------* //    
        
    def odmkImgColorEnhance(self, imgList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of images fading from img1 -> img2 for xLength seconds
            Linearly alpha-blends images using PIL Image.blend
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgcEnhancerNm = imgOutNm
        else:
            imgcEnhancerNm = 'ImgXfade'
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---xfadeFrames = '+str(xfadeFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgList)
        numxfadeImg = numImg * xfadeFrames
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        factorX = np.linspace(0.1, 2.0, xfadeFrames)
    
        imgcEnhancerNmArray = []
        imgcEnhancerArray = []
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades - 1):
            if frameCnt <= numFrames:
                if (frameCnt % numxfadeImg) == 0:
                    xfdirection = -xfdirection
                if xfdirection == 1:
                    imgPIL1 = Image.fromarray(imgList[j % numImg])
                    cEnhancer = ImageEnhance.Color(imgPIL1)
                else:
                    imgPIL1 = Image.fromarray(imgList[(numImg-1) - (j % numImg) ])
                    cEnhancer = ImageEnhance.Color(imgPIL1)
                for i in range(xfadeFrames):
                    imgCEn = cEnhancer.enhance(factorX[i])
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    imgcEnhancerFull = imgOutDir+imgcEnhancerNm+strInc+'.jpg'
                    misc.imsave(imgcEnhancerFull, imgCEn)
                    imgcEnhancerNmArray.append(imgcEnhancerFull)
                    imgcEnhancerArray.append(imgCEn)
                    frameCnt += 1
    
        return [imgcEnhancerArray, imgcEnhancerNmArray]
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Color Enhance::---*')
    # // *--------------------------------------------------------------* //    
        
    def odmkImgCEnTelescope(self, imgList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOut=0):
        ''' outputs a sequence of images fading from img1 -> img2 for xLength seconds
            Linearly alpha-blends images using PIL Image.blend
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgCEnTelescopeNm = imgOutNm
        else:
            imgCEnTelescopeNm = 'ImgXfade'
    
        # initialize SzX, SzY to dimensions of src img
        SzX = imgList[0].shape[1]
        SzY = imgList[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numBeats = int(ceil(numFrames / xfadeFrames))         
    
        #hop_sz = 4*ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
        #hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
        hop_sz = 1
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---xfadeFrames = '+str(xfadeFrames)+'---*')
        print('// *---numBeats = '+str(numBeats)+'---*')    
    
        # find number of source images in src dir
        numSrcImg = len(imgList)
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        factorX = np.linspace(0.1, 2.0, xfadeFrames)
    
        #imgcEnhancerNmArray = []
        #imgcEnhancerArray = []
    
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        for i in range(numBeats):
            newDimX = SzX
            newDimY = SzY
            imgClone = imgList[i%numSrcImg]    # mod to rotate through src img
            for t in range(xfadeFrames):
                if newDimX > 2:
                    #newDimX -= 2*hop_sz
                    newDimX -= hop_sz
                if newDimY > 2:
                    #newDimY -= 2*hop_sz
                    newDimY -= hop_sz
                # scale image to new dimensions
                imgItr = self.odmkEyeRescale(imgClone, newDimX, newDimY)
                # region = (left, upper, right, lower)
                # subbox = (i + 1, i + 1, newDimX, newDimY)
                for j in range(SzY):
                    for k in range(SzX):
                        #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
                        if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                            #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
                            imgClone[j, k, :] = imgItr[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
                nextInc += 1
                if nextInc < numFrames:
                    zr = ''
                    if inOut == 1:
                        imgCloneCEn = Image.fromarray(imgClone)
                        cEnhancer = ImageEnhance.Color(imgCloneCEn)           
                        imgCloneCEn = cEnhancer.enhance(factorX[t])
                        imgCloneCEn = np.array(imgCloneCEn)
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        imgCloneCEn = Image.fromarray(imgClone)
                        cEnhancer = ImageEnhance.Color(imgCloneCEn)           
                        imgCloneCEn = cEnhancer.enhance(factorX[(xfadeFrames-1)-t])
                        imgCloneCEn = np.array(imgCloneCEn)                    
                        for j in range(n_digits - len(str(numFrames - (nextInc)))):
                            zr += '0'
                        strInc = zr+str(numFrames - (nextInc))
                    imgTelescFull = imgOutDir+imgCEnTelescopeNm+strInc+'.jpg'
                    misc.imsave(imgTelescFull, imgCloneCEn)
                else:
                    break
    
    # convert from Image image to Numpy array:
    # arr = np.array(img)
    
    # convert from Numpy array to Image image:
    # img = Image.fromarray(array)
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Pixel Random Replace::---*
    # // *--------------------------------------------------------------* //
    
    def odmkPxlRndReplace(self, imgArray, sLength, framesPerSec, frameCtrl, pxlCtrl, imgOutDir, imgOutNm='None', numPxl=0):
        ''' swap n pixels per frame from consecutive images'''
    
        if imgOutNm != 'None':
            pxlRndReplaceNm = imgOutNm
        else:
            pxlRndReplaceNm = 'pxlRndReplace'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(sLength * framesPerSec))
        numXfades = int(ceil(numFrames / frameCtrl))
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---FrameCtrl = '+str(frameCtrl)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgArray)
        numxfadeImg = numImg * frameCtrl
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
    
        pxlRndReplaceNmArray = []
        pxlRndReplaceArray = []
    
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades - 1):
            if frameCnt <= numFrames:    # process until end of total length
            
                newDimX = SzX
                newDimY = SzY
                
                # example cloned img
                imgClone1 = imgArray[j]
    
                img = Image.new( 'RGB', (255,255), "black") # create a new black image
                imgPxl = img.load() # create the pixel map
    
                # scroll forward -> backward -> forward... through img array
                if (frameCnt % numxfadeImg) == 0:
                    xfdirection = -xfdirection
                if xfdirection == 1:
                    imgPIL1 = Image.fromarray(imgArray[j % (numImg-1)])
                    imgPIL2 = Image.fromarray(imgArray[(j % (numImg-1)) + 1])
                else:
                    imgPIL1 = Image.fromarray(imgArray[(numImg-2) - (j % numImg) + 1])
                    imgPIL2 = Image.fromarray(imgArray[(numImg-2) - (j % numImg)])
    
                # process frameCtrl timed sub-routine:
                for i in range(frameCtrl):
    
                    # example usrCtrl update                
                    if newDimX > 2:
                        newDimX -= 2*hop_sz
                    if newDimY > 2:
                        newDimY -= 2*hop_sz
                    
                    # example algorithmic process
                    alphaB = Image.blend(imgPIL1, imgPIL2, alphaX[i])
    
                    # process sub-frame
                    for j in range(SzY):
                        for k in range(SzX):
                            #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
                                imgClone1[j, k, :] = alphaB[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
    
                    # ***sort out img.size vs. imgList[0].shape[1]
                    # bit-banging
                    for i in range(img.size[0]):    # for every pixel:
                        for j in range(img.size[1]):
                            imgPxl[i,j] = (i, j, 1) # set the colour accordingly
    
    
                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    pxlRndReplaceFull = imgOutDir+pxlRndReplaceNm+strInc+'.jpg'
                    
                    # write img to disk
                    misc.imsave(pxlRndReplaceFull, alphaB)
                    
                    # optional write to internal buffers
                    pxlRndReplaceNmArray.append(pxlRndReplaceFull)
                    pxlRndReplaceArray.append(alphaB)
                    frameCnt += 1
    
        # return
        return [pxlRndReplaceArray, pxlRndReplaceNmArray]
   
   
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Dual Bipolar Oscillating telescope f::---*
    # // *--------------------------------------------------------------* //
    
    def odmkDualBipolarTelesc(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0):
        ''' DualBipolarTelesc function '''
    
        if imgOutNm != 'None':
            imgDualBipolarTelescNm = imgOutNm
        else:
            imgDualBipolarTelescNm = 'imgDualBipolarTelesc'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---numxfadeImg = '+str(xfadeFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgArray)
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
    
        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
        imgRndIdxArray = randomIdxArray(numFrames, numImg)
   
        # imgDualBipolarTelescNmArray = []
        # imgDualBipolarTelescArray = []
 
       
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades):
                                # random img grap from image Array

            if (frameCnt % numXfades) == 0:
                xfdirection = -xfdirection
            if xfdirection == 1:
                imgClone1 = imgArray[imgRndIdxArray[j]]
                imgClone2 = imgArray[imgRndIdxArray[j+1]]                
            else:
                imgClone1 = imgArray[imgRndIdxArray[(numImg-2) - j]]
                imgClone2 = imgArray[imgRndIdxArray[(numImg-2) - j+1]]
                
            # initialize output 
            imgBpTsc1 = imgClone1
            imgBpTsc2 = imgClone2
            imgBpTscOut = imgClone1

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            focalRnd1 = randomPxlLoc(SzX, SzY)
            if (focalRnd1[0] < SzX/2):
                offsetX = -int(SzX/2 - focalRnd1[0])
            else:
                offsetX = int(focalRnd1[0] - SzX/2)
            if (focalRnd1[1] < SzY/2):
                offsetY = -int(SzY/2 - focalRnd1[1])
            else:
                offsetY = int(focalRnd1[1] - SzY/2)

            #print('focalRnd1 = ['+str(focalRnd1[0])+', '+str(focalRnd1[1])+']')
            #print('offsetX = '+str(offsetX))
            #print('offsetY = '+str(offsetY))

            # focalRnd2 = randomPxlLoc(SzX, SzY)


            # functionalize then call for each telescope    
            for t in range(xfadeFrames):
                if frameCnt < numFrames:    # process until end of total length

                    if newDimX > hop_sz:
                        #newDimX -= 2*hop_sz
                        newDimX -= hop_sz
                    if newDimY > hop_sz:
                        #newDimY -= 2*hop_sz
                        newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)


                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    for j in range(SzY):
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                
                                # imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                # imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # rewrite inOrOut as bipolar toggle switch in loop above
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        for j in range(n_digits - len(str(numFrames - (nextInc)))):
                            zr += '0'
                        strInc = zr+str(numFrames - (nextInc))
                    imgDualBipolarTelescFull = imgOutDir+imgDualBipolarTelescNm+strInc+'.jpg'
                    misc.imsave(imgDualBipolarTelescFull, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)
                    
                    frameCnt += 1
    
        return
        # return [imgDualBipolarTelescArray, imgDualBipolarTelescNmArray]   


    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Bananasloth Recursion::---*
    # // *--------------------------------------------------------------* //
    
    def odmkBSlothRecurs(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0):
        ''' BSlothRecurs function '''
    
        if imgOutNm != 'None':
            imgBSlothRecursNm = imgOutNm
        else:
            imgBSlothRecursNm = 'imgBSlothRecurs'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---numxfadeImg = '+str(xfadeFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgArray)
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
    
        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
        imgRndIdxArray = randomIdxArray(numFrames, numImg)
   
        # imgBSlothRecursNmArray = []
        # imgBSlothRecursArray = []
 
       
    
        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades):
                                # random img grap from image Array

            if (frameCnt % numXfades) == 0:
                xfdirection = -xfdirection
            if xfdirection == 1:
                imgClone1 = imgArray[imgRndIdxArray[j]]
                imgClone2 = imgArray[imgRndIdxArray[j+1]]                
            else:
                imgClone1 = imgArray[imgRndIdxArray[(numImg-2) - j]]
                imgClone2 = imgArray[imgRndIdxArray[(numImg-2) - j+1]]
                
            # initialize output 
            imgBpTsc1 = imgClone1
            imgBpTsc2 = imgClone2
            imgBpTscOut = imgClone1

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            focalRnd1 = randomPxlLoc(SzX, SzY)
            if (focalRnd1[0] < SzX/2):
                offsetX = -int(SzX/2 - focalRnd1[0])
            else:
                offsetX = int(focalRnd1[0] - SzX/2)
            if (focalRnd1[1] < SzY/2):
                offsetY = -int(SzY/2 - focalRnd1[1])
            else:
                offsetY = int(focalRnd1[1] - SzY/2)

            #print('focalRnd1 = ['+str(focalRnd1[0])+', '+str(focalRnd1[1])+']')
            #print('offsetX = '+str(offsetX))
            #print('offsetY = '+str(offsetY))

            # focalRnd2 = randomPxlLoc(SzX, SzY)


            # functionalize then call for each telescope    
            for t in range(xfadeFrames):
                if frameCnt < numFrames:    # process until end of total length

                    if newDimX > hop_sz:
                        #newDimX -= 2*hop_sz
                        newDimX -= hop_sz
                    if newDimY > hop_sz:
                        #newDimY -= 2*hop_sz
                        newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)


                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    for j in range(SzY):
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                
                                # imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                # imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # rewrite inOrOut as bipolar toggle switch in loop above
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        for j in range(n_digits - len(str(numFrames - (nextInc)))):
                            zr += '0'
                        strInc = zr+str(numFrames - (nextInc))
                    imgBSlothRecursFull = imgOutDir+imgBSlothRecursNm+strInc+'.jpg'
                    misc.imsave(imgBSlothRecursFull, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)
                    
                    frameCnt += 1
    
        return
        # return [imgBSlothRecursArray, imgBSlothRecursNmArray]

    
    # /////////////////////////////////////////////////////////////////////////
    # #########################################################################
    # end : function definitions
    # #########################################################################
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //
