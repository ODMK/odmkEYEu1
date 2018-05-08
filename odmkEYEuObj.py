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
import glob, shutil
from math import atan2, floor, ceil
import random
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance

# optional
import matplotlib.pyplot as plt

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


class odmkFibonacci:
    ''' Iterator that generates numbers in the Fibonacci sequence '''

    def __init__(self, max):
        self.max = max

    def __iter__(self):
        self.a = 0
        self.b = 1
        return self

    def __next__(self):
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib


# // *--------------------------------------------------------------* //
# // *---::ODMKEYE - concat all files in directory list::---*
# // *--------------------------------------------------------------* //


def convertJPGtoBMP(jpgDir, bmpDir, reName='None'):
    ''' converts .jpg  images to .bmp images,
        renames and concatenates all files into new arrays. '''


    imgFileList = []
    imgFileList.extend(sorted(glob.glob(jpgDir+'*.jpg')))

    #pdb.set_trace()

    imgCount = len(imgFileList)
    n_digits = int(ceil(np.log10(imgCount))) + 2
    nextInc = 0
    for i in range(imgCount):

        imgObj = Image.open(imgFileList[i])        
        
        nextInc += 1
        zr = ''
        for j in range(n_digits - len(str(nextInc))):
            zr += '0'
        strInc = zr+str(nextInc)

        imgNormalizeNm = reName+strInc+'.bmp'
        
        imgNmFull = os.path.join(bmpDir+imgNormalizeNm)
        imgObj.save(imgNmFull, 'bmp')

    return


# a random 0 or 1
# rndIdx = round(random.random())

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
        --imgFormat: img format
        --SzX, SzY: video dimensions
        --framesPerSec: video frames per second
        --fs: audio sample rate
        --rootDir: root directory of Python code
        >>tbEYEu = eyeInst.odmkEYEu(eyeLength, 93, 30.0, 41000, rootDir)
    '''

    def __init__(self, xLength, bpm, timeSig, SzX, SzY, imgFormat, framesPerSec=30.0, fs=48000, cntrlEYE=0, rootDir='None'):

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
        
        self.imgFormat = imgFormat    # imgFormat => { fbmp, fjpg }

        self.framesPerSec = framesPerSec
        
        self.cntrlEYE = cntrlEYE

        self.eyeDir = rootDir+'eye/eyeSrc/'
        os.makedirs(self.eyeDir, exist_ok=True)
        
        print('\n*****odmkEYEuObj: An odmkEYEu object has been instanced with:')
        print('xLength = '+str(self.xLength)+', bpm = '+str(self.bpm)+', timeSig = '+str(self.timeSig)+', framesPerSec = '+str(self.framesPerSec)+' ; fs = '+str(self.fs))


        # // *-------------------------------------------------------------* //
        # // *---::Set Master Dimensions::---* //
        self.mstrSzX = SzX
        self.mstrSzY = SzY        

        
        # // *-------------------------------------------------------------* //
        
        print('\n')
        print('// *------------------------------------------------------* //')
        print('// *---::*****odmkEYEuObj: Instantiate clocking/sequencing object::---*')
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
            if (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')  or filename.endswith('.bmp')):
                imgSrcList.append(filename)
                imgPath = srcDir+filename
                imgObjTemp = misc.imread(imgPath)
                
                #pdb.set_trace()
                # if grey-scale, convert to RGB
                if (len(imgObjTemp.shape) < 3):
                    imgObjTemp = np.dstack([imgObjTemp.astype(np.uint8)] * 3)
                
                imgObjList.append(imgObjTemp)
        imgCount = len(imgSrcList)
        print('\nFound '+str(imgCount)+' images in the folder:\n')
        print(imgSrcList[0]+' ... '+imgSrcList[-1])
        # print('\nCreated python lists:')
        # print('<<imgObjList>> (ndimage objects)')
        # print('<<imgSrcList>> (img names)\n')
        # print('// *--------------------------------------------------------* //')
    
        return [imgObjList, imgSrcList]



    # // *********************************************************************** //    
    # // *********************************************************************** //
    # // *---::ODMK img Pixel-Banging Basic Func::---*
    # // *********************************************************************** //
    # // *********************************************************************** //



    # *---FIXIT---*
    

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



    def eyeBox1(self, img1, img2, boxSzX, boxSzY, pxlLoc, alpha):
        ''' alpha blends random box from img2 onto img1
            sub dimensions defined by pxlLoc [x, y]
            img1: img4 PIL images
            ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''
    
    
        if img1.size != img2.size:
            print('ERROR: img1, img2 must be same Dim')
            return        
    
       
        SzX = img1.size[0]
        SzY = img1.size[1]
        #imgDim = [SzX, SzY]

        
        subDim_L = int(pxlLoc[0] - boxSzX//2)
        if subDim_L < 0:
            subDim_L = 0
        subDim_R = int(pxlLoc[0] + boxSzX//2)
        if subDim_R > SzX:
            subDim_R = SzX-1
        subDim_B = int(pxlLoc[1] - boxSzY//2)
        if subDim_B < 0:
            subDim_B = 0
        subDim_T = int(pxlLoc[1] + boxSzX//2)
        if subDim_T > SzY:
            subDim_T = SzY-1
        
        
        # coordinates = (left, lower, right, upper)
        boxCC = (subDim_L, subDim_B, subDim_R, subDim_T)
        
        if alpha == 1:
            eyeSubCC = img2.copy().crop(boxCC)       
    
    
        else:        
            eyeSub1 = img1.copy().crop(boxCC)
            eyeSub2 = img2.copy().crop(boxCC)    
    
            eyeSubCC = Image.blend(eyeSub1, eyeSub2, alpha)
            eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)
    
        #print('eyeSubCC X = '+str(eyeSubCC.shape[1])+', eyeSubCC Y = '+str(eyeSubCC.shape[0])+)
        #pdb.set_trace()
    
        # left frame
        img1.paste(eyeSubCC, boxCC)
    
        
        return img1



    # // *********************************************************************** //
    # // *---::ODMK horizontal trails::---*
    # // *********************************************************************** //


    def horizTrails(self, img, SzX, SzY, nTrail, ctrl):
        ''' reconstructs img with left & right trails
            img - a PIL image (Image.fromarray(eye2))
            nTrail controls number of trails
            ctrl controls x-axis coordinates '''
        
        if ctrl == 0:    # linear
            trails = np.linspace(0, SzX/2, nTrail+2)
            trails = trails[1:len(trails)-1]
        
        else:    # log spaced    
            trails = np.log(np.linspace(1,10,nTrail+2))
            sc = 1/max(trails)
            trails = trails[1:len(trails)-1]
            trails = trails*sc*int(SzX/2)
        
        alpha = np.linspace(0, 1, nTrail)
        
        # coordinates = (left, lower, right, upper)
        boxL = (0, 0, int(SzX / 2), SzY)
        eyeSubL = img.crop(boxL)
        
        boxR = (int(SzX / 2), 0, SzX, SzY)
        eyeSubR = img.crop(boxR)
        
        
        alphaCnt = 0
        for trail in trails:
        
            itrail = int(trail)
        
            boxL_itr1 = (itrail, 0, int(SzX / 2), SzY)
            eyeSubL_itr1 = img.crop(boxL_itr1)
            #print('\nboxL_itr1 = '+str(boxL_itr1))    
            
            boxR_itr1 = (int(SzX / 2), 0, SzX - itrail, SzY)
            eyeSubR_itr1 = img.crop(boxR_itr1)
            #print('\nboxR_itr1 = '+str(boxR_itr1))
            # if tcnt == 2:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            #pdb.set_trace()  
            
            boxL_base1 = (0, 0, int(SzX / 2) - itrail, SzY)
            eyeSubL_base1 = img.crop(boxL_base1)
            
            boxR_base1 = (int((SzX / 2) + itrail), 0, SzX, SzY)
            eyeSubR_base1 = img.crop(boxR_base1)
            
            eyeFadeL_itr1 = Image.blend(eyeSubL_itr1, eyeSubL_base1, alpha[alphaCnt])
            eyeFadeR_itr1 = Image.blend(eyeSubR_itr1, eyeSubR_base1, alpha[alphaCnt])
            
            #pdb.set_trace()
            
            boxL_sb1 = (0, 0, int(SzX / 2) - itrail, SzY)
            boxR_sb1 = (itrail, 0, int(SzX / 2), SzY)    
           
            
            #eye1subL.paste(eye1subL_itr1, boxL_sb1)
            #eye2subR.paste(eye2subR_itr1, boxR_sb1)
            eyeSubL.paste(eyeFadeL_itr1, boxL_sb1)
            eyeSubR.paste(eyeFadeR_itr1, boxR_sb1)    
            
            
            img.paste(eyeSubL, boxL)
            img.paste(eyeSubR, boxR)
        
            alphaCnt += 1    
    
        return img
    
    
    
    def eyeMirrorHV4(self, img, SzX, SzY, ctrl='UL'):
        ''' generates composite 4 frame resized images
            SzX, SzY: output image dimentions
            img: PIL image object
            ctrl: { UL, UR, LL, LR} - selects base quadrant '''
        
        
        if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
            print('ERROR: ctrl must be {UL, UR, LL, LR} <string>')
            return
        
        imgSzX = img.size[0]
        imgSzY = img.size[1]
        
        imgDim = (SzX, SzY)
        
        if (SzX == 2*imgSzX and SzY == 2*imgSzY):
            subDim = (imgSzX, imgSzY)
            imgX = img
        else:
            subDim = (int(SzX/2), int(SzY/2))
            imgX = img.resize(subDim)
        
        
        mirrorImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480))
        
    
        #boxUL = (0, SzY, SzX-1, 2*SzY-1)
        boxUL = (0, subDim[1], subDim[0], imgDim[1])
        boxUR = (subDim[0], subDim[1], imgDim[0], imgDim[1]) 
        boxLL = (0, 0, subDim[0], subDim[1])
        boxLR = (subDim[0], 0, imgDim[0], subDim[1])
        
    
        if ctrl == 'LL':
            eyeSubUL = imgX
            eyeSubUR = ImageOps.mirror(imgX)
            eyeSubLL = ImageOps.flip(imgX)
            eyeSubLR = ImageOps.flip(imgX)
            eyeSubLR = ImageOps.mirror(eyeSubLR)        
            
            
        if ctrl == 'LR':
            eyeSubUL = ImageOps.mirror(imgX)
            eyeSubUR = imgX
            eyeSubLL = ImageOps.flip(imgX)
            eyeSubLL = ImageOps.mirror(eyeSubLL)
            eyeSubLR = ImageOps.flip(imgX)
            
            
        if ctrl == 'UL':
            eyeSubUL = ImageOps.flip(imgX)
            eyeSubUR = ImageOps.flip(imgX)
            eyeSubUR = ImageOps.mirror(eyeSubUR)
            eyeSubLL = imgX
            eyeSubLR = ImageOps.mirror(imgX)
    
    
        if ctrl == 'UR':
            eyeSubUL = ImageOps.flip(imgX)
            eyeSubUL = ImageOps.mirror(eyeSubUL)
            eyeSubUR = ImageOps.flip(imgX)
            eyeSubLL = ImageOps.mirror(imgX)
            eyeSubLR = imgX
    
    
        #pdb.set_trace()
    
        mirrorImg.paste(eyeSubUL, boxUL)
        mirrorImg.paste(eyeSubUR, boxUR)
        mirrorImg.paste(eyeSubLL, boxLL)
        mirrorImg.paste(eyeSubLR, boxLR)
    
        
        #eyeOutFileName = 'eyeMirrorHV4'
        #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
        #misc.imsave(eyeMirrorHV4Full, mirrorImg)
        
        return mirrorImg
        
        
    def eyeMirrorTemporalHV4(self, img1, img2, img3, img4, SzX, SzY, ctrl='UL'):
        ''' generates composite 4 frame resized images
            Order = clockwise from ctrl setting
            SzX, SzY: output image dimentions
            img: PIL image object
            ctrl: { UL, UR, LL, LR} - selects base quadrant '''
        
        
        if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
            print('ERROR: ctrl must be {UL, UR, LL, LR} <string>')
            return
        
        imgSzX = img1.size[0]
        imgSzY = img1.size[1]

        imgDim = (SzX, SzY)


        mirrorTemporalImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480))
        
        if (SzX == 2*imgSzX and SzY == 2*imgSzY):
            subDim = (imgSzX, imgSzY)
            imgX1 = img1
            imgX2 = img2
            imgX3 = img3
            imgX4 = img4
        else:
            subDim = (int(SzX/2), int(SzY/2))
            imgX1 = img1.resize(subDim)
            imgX2 = img2.resize(subDim)
            imgX3 = img3.resize(subDim)
            imgX4 = img4.resize(subDim)


        #boxUL = (0, SzY, SzX-1, 2*SzY-1)
        boxUL = (0, subDim[1], subDim[0], imgDim[1])
        boxUR = (subDim[0], subDim[1], imgDim[0], imgDim[1]) 
        boxLL = (0, 0, subDim[0], subDim[1])
        boxLR = (subDim[0], 0, imgDim[0], subDim[1])
        
    
        if ctrl == 'LL':
            eyeSubUL = imgX1
            eyeSubUR = ImageOps.mirror(imgX2)
            eyeSubLL = ImageOps.flip(imgX3)
            eyeSubLR = ImageOps.flip(imgX4)
            eyeSubLR = ImageOps.mirror(eyeSubLR)        
            
            
        if ctrl == 'LR':
            eyeSubUL = ImageOps.mirror(imgX4)
            eyeSubUR = imgX1
            eyeSubLL = ImageOps.flip(imgX2)
            eyeSubLL = ImageOps.mirror(eyeSubLL)
            eyeSubLR = ImageOps.flip(imgX3)
            
            
        if ctrl == 'UL':
            eyeSubUL = ImageOps.flip(imgX3)
            eyeSubUR = ImageOps.flip(imgX4)
            eyeSubUR = ImageOps.mirror(eyeSubUR)
            eyeSubLL = imgX1
            eyeSubLR = ImageOps.mirror(imgX2)
    
    
        if ctrl == 'UR':
            eyeSubUL = ImageOps.flip(imgX2)
            eyeSubUL = ImageOps.mirror(eyeSubUL)
            eyeSubUR = ImageOps.flip(imgX3)
            eyeSubLL = ImageOps.mirror(imgX4)
            eyeSubLR = imgX1
    
    
        #pdb.set_trace()
    
        mirrorTemporalImg.paste(eyeSubUL, boxUL)
        mirrorTemporalImg.paste(eyeSubUR, boxUR)
        mirrorTemporalImg.paste(eyeSubLL, boxLL)
        mirrorTemporalImg.paste(eyeSubLR, boxLR)
    
        
        #eyeOutFileName = 'eyeMirrorHV4'
        #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
        #misc.imsave(eyeMirrorHV4Full, mirrorImg)
        
        return mirrorTemporalImg

    
    
    def eyePhiHorizFrame1(self, img1, img2, img3, ctrl=1):
        ''' generates composite 4 frame resized images
            sub dimensions defined by pxlLoc [x, y]
            img1: img4 PIL images
            ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''
    
    
        if (img1.size != img2.size != img3.size):
            print('ERROR: img1, img2 must be same Dim')
            return        
    
        phi = 1.6180339887
        phiInv = 0.6180339887
        oneMphiInv = 0.38196601129
       
        SzX = img1.size[0]
        SzY = img1.size[1]
        imgDim = [SzX, SzY]
        
        # toggle background on/off
        if ctrl == 0:
            phiFrameImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480)) 
        else:    
            phiFrameImg = img1.copy()   
        
        dimH_L = int(SzX*oneMphiInv)
        dimH_R = int(SzX*phiInv)
        
        dimC_B = int(SzY*oneMphiInv)
        dimC_T = int(SzY*phiInv)
        
        dimCC_L = int(dimH_L/2)
        dimCC_R = SzX - int(dimH_L/2)
        dimCC_B = int(dimC_B/2)
        dimCC_T = SzY - int(dimC_B/2)    
        
        
        # coordinates = (left, lower, right, upper)
    #    boxL = (0, 0, dimH_L, SzY)
    #    boxC = (dimH_L, 0, dimH_R, SzY)
    #    boxR = (dimH_R, 0, dimH_R, SzY)    
    #    
    #    boxC_B = (dimH_L, dimC_T, dimH_R, SzY)
    #    boxC_T = (dimH_L, 0, dimH_R, dimC_B)    
      
        boxCC = (dimCC_L, dimCC_B, dimCC_R, dimCC_T)
        
    
        boxL_B = (0, dimC_T, dimH_L, SzY)
        boxL_T = (0, 0, dimH_L, dimC_B)
        
        boxR_B = (dimH_R, dimC_T, SzX, SzY)
        boxR_T = (dimH_R, 0, SzX, dimC_B)
    
    
        eyeSubLT = img1.copy().crop(boxL_T)
        eyeSubLB = img2.copy().crop(boxL_B)
        
        #eyeSubTC = img2.resize(subDim)
        #eyeSubBC = img2.copy().crop(boxC_B)
    
        eyeSubRT = img2.copy().crop(boxR_T)
        eyeSubRB = img1.copy().crop(boxR_B) 
    
    
        subDimCC = (dimCC_R - int(dimH_L/2), dimCC_T - int(dimC_B/2))
    
        eyeSubCC = img1.copy().resize(subDimCC)
        eyeSubCC2 = img2.copy().resize(subDimCC)
        eyeSubCC = Image.blend(eyeSubCC, eyeSubCC2, 0.5)
        eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)
    
    
        # left frame
        phiFrameImg.paste(eyeSubLT, boxL_T)
        phiFrameImg.paste(eyeSubLB, boxL_B)
        
        # rigth frame
        phiFrameImg.paste(eyeSubRT, boxR_T)
        phiFrameImg.paste(eyeSubRB, boxR_B)
    
        # center frame
        #phiFrameImg.paste(eyeSubTC, boxC_T)
        #phiFrameImg.paste(eyeSubBC, boxC_B)
        
        # center center
        phiFrameImg.paste(eyeSubCC, boxCC)
    
        
        #eyeOutFileName = 'eyeMirrorHV4'
        #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
        #misc.imsave(eyeMirrorHV4Full, mirrorImg)
        
        return phiHorizFrameImg


    def eyePhiFrame1(self, img1, img2, ctrl=1):
        ''' generates composite 4 frame resized images
            sub dimensions defined by pxlLoc [x, y]
            img1: img4 PIL images
            ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''
    
    
        if img1.size != img2.size:
            print('ERROR: img1, img2 must be same Dim')
            return        
    
        phi = 1.6180339887
        phiInv = 0.6180339887
        oneMphiInv = 0.38196601129
       
        SzX = img1.size[0]
        SzY = img1.size[1]
        imgDim = [SzX, SzY]
        
        # toggle background on/off
        if ctrl == 0:
            phiFrameImg = Image.new('RGB', imgDim)  # e.g. ('RGB', (640, 480)) 
        else:    
            phiFrameImg = img1.copy()   
        
        dimH_L = int(SzX*oneMphiInv)
        dimH_R = int(SzX*phiInv)
        
        dimC_B = int(SzY*oneMphiInv)
        dimC_T = int(SzY*phiInv)
        
        dimCC_L = int(dimH_L/2)
        dimCC_R = SzX - int(dimH_L/2)
        dimCC_B = int(dimC_B/2)
        dimCC_T = SzY - int(dimC_B/2)    
        
        
        # coordinates = (left, lower, right, upper)
    #    boxL = (0, 0, dimH_L, SzY)
    #    boxC = (dimH_L, 0, dimH_R, SzY)
    #    boxR = (dimH_R, 0, dimH_R, SzY)    
    #    
    #    boxC_B = (dimH_L, dimC_T, dimH_R, SzY)
    #    boxC_T = (dimH_L, 0, dimH_R, dimC_B)    
      
        boxCC = (dimCC_L, dimCC_B, dimCC_R, dimCC_T)
        
    
        boxL_B = (0, dimC_T, dimH_L, SzY)
        boxL_T = (0, 0, dimH_L, dimC_B)
        
        boxR_B = (dimH_R, dimC_T, SzX, SzY)
        boxR_T = (dimH_R, 0, SzX, dimC_B)
    
    
        eyeSubLT = img1.copy().crop(boxL_T)
        eyeSubLB = img2.copy().crop(boxL_B)
        
        #eyeSubTC = img2.resize(subDim)
        #eyeSubBC = img2.copy().crop(boxC_B)
    
        eyeSubRT = img2.copy().crop(boxR_T)
        eyeSubRB = img1.copy().crop(boxR_B) 
    
    
        subDimCC = (dimCC_R - int(dimH_L/2), dimCC_T - int(dimC_B/2))
    
        eyeSubCC = img1.copy().resize(subDimCC)
        eyeSubCC2 = img2.copy().resize(subDimCC)
        eyeSubCC = Image.blend(eyeSubCC, eyeSubCC2, 0.5)
        eyeSubCC = ImageOps.autocontrast(eyeSubCC, cutoff=0)
    
    
        # left frame
        phiFrameImg.paste(eyeSubLT, boxL_T)
        phiFrameImg.paste(eyeSubLB, boxL_B)
        
        # rigth frame
        phiFrameImg.paste(eyeSubRT, boxR_T)
        phiFrameImg.paste(eyeSubRB, boxR_B)
    
        # center frame
        #phiFrameImg.paste(eyeSubTC, boxC_T)
        #phiFrameImg.paste(eyeSubBC, boxC_B)
        
        # center center
        phiFrameImg.paste(eyeSubCC, boxCC)
    
        
        #eyeOutFileName = 'eyeMirrorHV4'
        #eyeMirrorHV4Full = imgOutDir+eyeOutFileName+'.jpg'
        #misc.imsave(eyeMirrorHV4Full, mirrorImg)
        
        return phiFrameImg


    def eyePhiFrame2(self, img1, img2, ctrl='UL'):
        ''' generates subframes increasing by fibonacci sequence
            sub dimensions defined by pxlLoc [x, y]
            img1: img4 PIL images
            ctrl: toggle background - 0 = no background img (black), 1 = use img1 '''
    
    
        if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
            print('ERROR: ctrl must be {UL, UR, LL, LR} <string>')
            return
    
        if img1.size != img2.size:
            print('ERROR: img1, img2 must be same Dim')
            return        

       
        SzX = img1.size[0]
        SzY = img1.size[1]
        imgDim = [SzX, SzY]
        
        # coordinates = (left, lower, right, upper)
    
        if ctrl == 'LL':
            boxQTR = (0, int(SzY/2), int(SzX/2), SzY)        
        elif ctrl == 'LR':
            boxQTR = (int(SzX/2), int(SzY/2), SzX, SzY)        
        elif ctrl == 'UL':
            boxQTR = (0, 0, int(SzX/2), int(SzY/2))
        elif ctrl == 'UR':
            boxQTR = (int(SzX/2), 0, SzX, int(SzY/2))
            
            
        phiFrameImg = img1.copy()
        phiSubImg1 = img1.copy().crop(boxQTR)
        phiSubImg1 = phiSubImg1.resize(imgDim)
        phiSubImg2 = img2.copy().crop(boxQTR)
        phiSubImg2 = phiSubImg2.resize(imgDim)
    
    
        dimXarray = []
        for xx in odmkFibonacci(SzX):
            if xx > 8:
                dimXarray.append(xx)
                # print(str(xx))
                
        dimYarray = []
        for yy in dimXarray:
            dimYarray.append(int(yy*SzY/SzX))

       
        for bb in range(len(dimXarray)):
            
            if bb%2==1:
                boxBB = (0, 0, dimXarray[len(dimXarray) - bb - 1], dimYarray[len(dimYarray) - bb - 1])
                eyeSub = phiSubImg1.copy().crop(boxBB)
                phiFrameImg.paste(eyeSub, boxBB)
            else:
                boxBB = (0, 0, dimXarray[len(dimXarray) - bb - 1], dimYarray[len(dimYarray) - bb - 1])
                eyeSub = phiSubImg2.copy().crop(boxBB)
                phiFrameImg.paste(eyeSub, boxBB)
               
        
        return phiFrameImg


    # // *********************************************************************** //    
    # // *********************************************************************** //
    # // *---::ODMK img Src Sequencing func::---*
    # // *********************************************************************** //
    # // *********************************************************************** //



    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - interlace img files in two directoroies::---*
    # // *--------------------------------------------------------------* //

    def imgScanDir(self, dirList, numFrames, xCtrl):
        ''' Scans frames from a list of img directories, creates new renamed sequence
            xLength - length of final sequence (seconds)
            framesPerSec - video frames per second
            framesPerBeat - video frames per beat (tempo sync)
            xCtrl - directory selector, quantized to # of src dir
            returns: imgArray - sequenced list of fullpath img names'''

        # check xCtrl is bounded by len(dirList)
        if (max(xCtrl) > len(dirList)):
            print('ERROR: xCtrl value must not exceed number of input src dir')
            return 1
            
        imgArray = [[] for i in range(len(dirList))]
        for i in range(len(dirList)):
            tmpArray = []
            tmpArray.extend(sorted(glob.glob(dirList[i]+'*')))
            # print('Length of imgArray'+str(i)+' = '+str(len(tmpArray)))
            for j in tmpArray:
                imgArray[i].append(j)

        srcImgArray = []

        nextInc = 0
        idxArray = [0 for i in range(len(dirList))]   # array of index to track srcDir index position
        for i in range(numFrames):
            nextInc += 1

            # srcDir length are arbitrary - read frames (% by srcDir length to roll) then inc corresponding index
            srcImgArray.append(imgArray[int(xCtrl[i])][idxArray[int(xCtrl[i])]%len(imgArray[int(xCtrl[i])])])
            idxArray[int(xCtrl[i])]+=1            
            
            #pdb.set_trace()
    
        return srcImgArray


    # // *********************************************************************** //    
    # // *********************************************************************** //
    # // *---::ODMK img Pre-processing func::---*
    # // *********************************************************************** //
    # // *********************************************************************** //
    
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
        if (len(img.shape) < 3):
            imgRescale = misc.imresize(img, [SzY, SzX], interp='cubic')
        else:
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
    
    
    
    def cropZoom(self, img, zoom_factor, **kwargs):
    
        SzY, SzX = img.shape[:2]
        
        # ensure that the final image is >=, then crop
        if SzX % 2 != 0:
            SzXX = SzX + 1
        else:
            SzXX = SzX
            
        if SzY % 2 != 0:
            SzYY = SzY + 1
        else:
            SzYY = SzY


        # width and height of the zoomed image
        #zoomSzY = int(np.round(zoom_factor * SzYY))
        #zoomSzX = int(np.round(zoom_factor * SzXX))
    
        # for multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        # *** (xxx,) * n => (xxx, xxx, ... xxx) repeated n times
        # zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
        zoom_tuple = [zoom_factor,] * 2 + [1,] * (img.ndim - 2)    # outputs [2, 2, 1]
    

        # bounding box of the clip region within the input array
        # img[bottom:top, left:right] - crops image to range


        deltaX = SzXX // 4
        deltaY = SzYY // 4
        halfSzX = SzXX // 2
        halfSzY = SzYY // 2

    
        # out = ndimage.zoom(img[deltaY:zoomSzY - deltaY, deltaX:zoomSzX - deltaX], zoom_tuple, **kwargs)
        out = ndimage.zoom(img[deltaY:deltaY + halfSzY, deltaX:deltaX + halfSzX], zoom_tuple, **kwargs)

        
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        if (out.shape[0] != SzY or out.shape[1] != SzX):
            out = out[0:SzY, 0:SzX]

        #pdb.set_trace()

    
        # if zoom_factor == 1, just return the input array
        else:
            out = img
        return out

    
    
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
 

    def odmkScaleAll(self, srcDir, SzX, SzY, outDir, high=0, outName='None'):
        ''' rescales and normalizes all .jpg files in image object list.
            scales, zooms, & crops images to dimensions SzX, SzY
            Assumes srcObjList of ndimage objects exists in program memory
            Typically importAllJpg is run first. '''
    

        os.makedirs(outDir, exist_ok=True)
        
        if outName != 'None':
            imgScaledOutNm = outName
        else:
            imgScaledOutNm = 'odmkScaleAllOut'

            
        imgFileList = []
        imgFileList.extend(sorted(glob.glob(srcDir+'*')))


        imgCount = len(imgFileList)
        # Find num digits required to represent max index
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for k in range(imgCount):
            srcObj = misc.imread(imgFileList[k])
            imgScaled = self.odmkEyeDim(srcObj, SzX, SzY, high)
            # auto increment output file name
            nextInc += 1
            zr = ''    # reset lead-zero count to zero each itr
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgScaledNm = imgScaledOutNm+strInc+'.jpg'

            imgScaledFull = outDir+imgScaledNm
            misc.imsave(imgScaledFull, imgScaled)
      
        print('Saved Scaled images to the following location:')
        print(imgScaledFull)
    
        print('\nScaled all images in the source directory\n')
        print('Renamed files: "processScaled00X.jpg"')

    
        print('\n')
    
        print('// *--------------------------------------------------------------* //')
    
        return



    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - phiFrame2 img files in two directoroies::---*
    # // *--------------------------------------------------------------* //

    def imgPhiFrame2Array(self, imgList1, imgList2, numFrames, xCtrl='UL'):
        ''' Scans frames from a list of img directories, creates new renamed sequence
            xLength - length of final sequence (seconds)
            framesPerSec - video frames per second
            framesPerBeat - video frames per beat (tempo sync)
            xCtrl - directory selector, quantized to # of src dir
            returns: imgArray - sequenced list of fullpath img names'''

        if not(xCtrl=='UL' or xCtrl=='UR' or xCtrl=='LL' or xCtrl=='LR'):
            print('ERROR: xCtrl must be {UL, UR, LL, LR} <string>')
            return


        srcImgArray = []

        nextInc = 0
        for i in range(numFrames):
            nextInc += 1         
            
            eye1img = misc.imread(imgList1[i])
            eye2img = misc.imread(imgList2[i]) 
            
            phiFrame2 = self.eyePhiFrame2(eye1img, eye2img, xCtrl)

            # srcDir length are arbitrary - read frames (% by srcDir length to roll) then inc corresponding index
            srcImgArray.append(phiFrame2)            
            
            #pdb.set_trace()
    
        return srcImgArray


   
    # // *********************************************************************** //    
    # // *********************************************************************** //
    # // *---::ODMK img Post-processing func::---*
    # // *********************************************************************** //
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


    def concatAllDir(self, dirList, concatDir, reName):
        ''' Takes a list of directories containing .jpg /.bmp images,
            renames and concatenates all files into new arrays.
            The new file names are formatted for processing with ffmpeg
            if w = 1, write files into output directory.
            if concatDir specified, change output dir name.
            if concatName specified, change output file names'''


#        imgFileList = []
#        for i in range(len(dirList)):
#            if self.imgFormat=='fjpg':
#                imgFileList.extend(sorted(glob.glob(dirList[i]+'*.jpg')))                
#            elif self.imgFormat=='fbmp':
#                imgFileList.extend(sorted(glob.glob(dirList[i]+'*.bmp')))
                
        imgFileList = []
        for i in range(len(dirList)):
            imgFileList.extend(sorted(glob.glob(dirList[i]+'*')))                
                


        imgCount = len(imgFileList)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(imgCount):
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)

            if self.imgFormat=='fjpg':
                imgNormalizeNm = reName+strInc+'.jpg' 
            elif self.imgFormat=='fbmp':
                imgNormalizeNm = reName+strInc+'.bmp'

            currentNm = os.path.split(imgFileList[i])[1]
            shutil.copy(imgFileList[i], concatDir)
            currentFile = os.path.join(concatDir+currentNm)
            
            imgConcatNmFull = os.path.join(concatDir+imgNormalizeNm)
            os.rename(currentFile, imgConcatNmFull)

        return


    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - interlace img files in two directoroies::---*
    # // *--------------------------------------------------------------* //

    def imgInterlaceDir(self, dir1, dir2, interlaceDir, reName):
        ''' Takes two directories containing .jpg /.bmp images,
            renames and interlaces all files into output dir. 
            The shortest directory length determines output (min length)'''


        imgFileList1 = []
        imgFileList2 = []
        

        if self.imgFormat=='fjpg':
            imgFileList1.extend(sorted(glob.glob(dir1+'*.jpg')))
            imgFileList2.extend(sorted(glob.glob(dir2+'*.jpg')))    
        elif self.imgFormat=='fbmp':
            imgFileList1.extend(sorted(glob.glob(dir1+'*.bmp')))
            imgFileList2.extend(sorted(glob.glob(dir2+'*.bmp')))     

        imgCount = 2*min(len(imgFileList1), len(imgFileList2))
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        dirIdx = 0
        for i in range(imgCount):
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            # print('strInc = '+str(strInc))
            if self.imgFormat=='fjpg':
                imgNormalizeNm = reName+strInc+'.jpg' 
            elif self.imgFormat=='fbmp':
                imgNormalizeNm = reName+strInc+'.bmp'
            #imgConcatNmFull = concatDir+imgNormalizeNm
            if i%2 == 1: 
                currentNm = os.path.split(imgFileList1[dirIdx])[1]
                shutil.copy(imgFileList1[dirIdx], interlaceDir)
                currentFile = os.path.join(interlaceDir+currentNm)
                dirIdx += 1
            else:
                currentNm = os.path.split(imgFileList2[dirIdx])[1]                
                shutil.copy(imgFileList2[dirIdx], interlaceDir)
                currentFile = os.path.join(interlaceDir+currentNm)
            
            imgInterlaceDirNmFull = os.path.join(interlaceDir+imgNormalizeNm)
            os.rename(currentFile, imgInterlaceDirNmFull)
            #pdb.set_trace()
    
        return


    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - interlace img files in two directoroies::---*
    # // *--------------------------------------------------------------* //

    def imgXfadeDir(self, dir1, dir2, xCtrl, interlaceDir, reName):
        ''' Takes two directories containing .jpg /.bmp images,
            renames and interlaces all files into output dir. 
            The shortest directory length determines output (min length)'''


        imgFileList1 = []
        imgFileList2 = []
        

        if self.imgFormat=='fjpg':
            imgFileList1.extend(sorted(glob.glob(dir1+'*.jpg')))
            imgFileList2.extend(sorted(glob.glob(dir2+'*.jpg')))    
        elif self.imgFormat=='fbmp':
            imgFileList1.extend(sorted(glob.glob(dir1+'*.bmp')))
            imgFileList2.extend(sorted(glob.glob(dir2+'*.bmp')))     

        imgCount = 2*min(len(imgFileList1), len(imgFileList2))
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        dirIdx1 = 0
        dirIdx2 = 0
        for i in range(imgCount):
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            # print('strInc = '+str(strInc))
            if self.imgFormat=='fjpg':
                imgNormalizeNm = reName+strInc+'.jpg' 
            elif self.imgFormat=='fbmp':
                imgNormalizeNm = reName+strInc+'.bmp'
            #imgConcatNmFull = concatDir+imgNormalizeNm
            if xCtrl[i] == 1: 
                currentNm = os.path.split(imgFileList1[dirIdx1])[1]
                shutil.copy(imgFileList1[dirIdx1], interlaceDir)
                currentFile = os.path.join(interlaceDir+currentNm)
                dirIdx1 += 1
            else:
                currentNm = os.path.split(imgFileList2[dirIdx2])[1]                
                shutil.copy(imgFileList2[dirIdx2], interlaceDir)
                currentFile = os.path.join(interlaceDir+currentNm)
                dirIdx2 += 1
            
            imgInterlaceDirNmFull = os.path.join(interlaceDir+imgNormalizeNm)
            os.rename(currentFile, imgInterlaceDirNmFull)
            #pdb.set_trace()
    
        return





    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - mirrorHV4 list of files in directory::---*
    # // *--------------------------------------------------------------* //
    
    def mirrorHV4AllImg(self, srcDir, SzX, SzY, mirrorHV4Dir, mirrorHV4Nm, ctrl='LR'):
        ''' mirrors image horizontally & vertically - outputs 4 mirrored subframes
            srcDir: input image directory
            SzX, SzY: output image frame size (input img pre-scaled if required)
            mirrorHV4Dir: processed img output directory
            mirrorHV4Nm: processed img root name
            ctrl: quadrant location of initial subframe {UL, UR, LL, LR}'''
    
        # check write condition, if w=0 ignore outDir
        if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
            print('ERROR: ctrl must be {UL, UR, LL, LR}')
            return
            
        # If Dir does not exist, makedir:
        os.makedirs(mirrorHV4Dir, exist_ok=True)
            
        [imgObjList, imgSrcList] = self.importAllJpg(srcDir)


        imgCount = len(imgObjList)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(imgCount):
            
            pilImg = Image.fromarray(imgObjList[i])
            imgMirrorHV4 = self.eyeMirrorHV4(pilImg, SzX, SzY, ctrl)           
            
            nextInc += 1
            zr = ''
            for h in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgMirrorHV4Nm = mirrorHV4Nm+strInc+'.jpg'

            imgMirrorHV4NmFull = mirrorHV4Dir+imgMirrorHV4Nm
            misc.imsave(imgMirrorHV4NmFull, imgMirrorHV4)
    
        return


    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - mirrorHV4 list of files in directory::---*
    # // *--------------------------------------------------------------* //
    
    def mirrorTemporalHV4AllImg(self, srcDir, SzX, SzY, mirrorTemporalHV4Dir, mirrorTemporalHV4Nm, frameDly, ctrl='LR'):
        ''' mirrors image horizontally & vertically - outputs 4 mirrored subframes
            srcDir: input image directory
            SzX, SzY: output image frame size (input img pre-scaled if required)
            mirrorHV4Dir: processed img output directory
            mirrorHV4Nm: processed img root name
            ctrl: quadrant location of initial subframe {UL, UR, LL, LR}'''
    
        # check write condition, if w=0 ignore outDir
        if not(ctrl=='UL' or ctrl=='UR' or ctrl=='LL' or ctrl=='LR'):
            print('ERROR: ctrl must be {UL, UR, LL, LR}')
            return
            
        # If Dir does not exist, makedir:
        os.makedirs(mirrorTemporalHV4Dir, exist_ok=True)
            
        [imgObjList, imgSrcList] = self.importAllJpg(srcDir)


        imgCount = len(imgObjList)
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(imgCount):
            
            pilImg1 = Image.fromarray(imgObjList[i])
            pilImg2 = Image.fromarray(imgObjList[(i + frameDly) % imgCount])
            pilImg3 = Image.fromarray(imgObjList[(i + 2*frameDly) % imgCount])
            pilImg4 = Image.fromarray(imgObjList[(i + 3*frameDly) % imgCount])            
            #imgMirrorHV4 = self.eyeMirrorHV4(pilImg, SzX, SzY, ctrl)
            
            imgMirrorTemporalHV4 = self.eyeMirrorTemporalHV4(pilImg1, pilImg2, pilImg3, pilImg4, SzX, SzY, ctrl)
            
            nextInc += 1
            zr = ''
            for h in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgMirrorTemporalHV4Nm = mirrorTemporalHV4Nm+strInc+'.jpg'

            imgMirrorTemporalHV4NmFull = mirrorTemporalHV4Dir+imgMirrorTemporalHV4Nm
            misc.imsave(imgMirrorTemporalHV4NmFull, imgMirrorTemporalHV4)
    
        return

    
    # // *********************************************************************** //    
    # // *********************************************************************** //
    # // *---::ODMK img Pixel-Banging Algorithms::---*
    # // *********************************************************************** //
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
    # 26 => EYE Bananasloth Glitch 1                 ::odmkBSlothGlitch1::
    # 28 => EYE lfo horizontal trails                ::odmkLfoHtrails::


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
    # // *---::ODMKEYE - Image Linear F<->B Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgLinSel(self, imgList, numFrames, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgLinSelNm = imgOutNm
        else:
            imgLinSelNm = 'imgLinSelOut'

        
        idx = 0
        
        imgCount = numFrames
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 1
        xfdirection = 1
        for i in range(numFrames):
            
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgNormalizeNm = imgLinSelNm+strInc+'.jpg'
            imgLinSelFull = imgOutDir+imgNormalizeNm
            misc.imsave(imgLinSelFull, imgList[idx])

            if (nextInc % len(imgList)-1) == 0:
                xfdirection = -xfdirection
            if xfdirection == 1:
                idx += 1                
            else:
                idx -= 1
           
            nextInc += 1
            
        return    

    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Linear F<->B Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgLinSol(self, imgFileList, numFrames, xFrames, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgLinSelNm = imgOutNm
        else:
            imgLinSelNm = 'imgLinSolOut'

        numImg = len(imgFileList)
        solarX = np.linspace(5.0, 255.0, xFrames)
       
        idx = 0
        
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 1
        xfdirection = 1
        for i in range(numFrames):
            
            solarA = misc.imread(imgFileList[i % numImg])
            
            imgPIL1 = Image.fromarray(solarA)
            
            solarB = ImageOps.solarize(imgPIL1, solarX[i % xFrames])
            solarB = ImageOps.autocontrast(solarB, cutoff=0)
            
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgNormalizeNm = imgLinSelNm+strInc+'.jpg'
            imgLinSelFull = imgOutDir+imgNormalizeNm
            misc.imsave(imgLinSelFull, solarB)

            if (nextInc % numImg-1) == 0:
                xfdirection = -xfdirection
            if xfdirection == 1:
                idx += 1                
            else:
                idx -= 1
           
            nextInc += 1
            
        return        
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Linear F<->B Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgRotLinSel(self, imgFileList, numFrames, xFrames, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgRndSelNm = imgOutNm
        else:
            imgRndSelNm = 'imgRndSelOut'
    
    
        zn = cyclicZn(xFrames-1)    # less one, then repeat zn[0] for full 360
        
        idx = 0
        
        numImg = len(imgFileList)
        imgCount = numFrames
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 1
        xfdirection = 1
        for i in range(numFrames):
            
            raw_gz1 = misc.imread(imgFileList[i % numImg])
            imgPIL1 = Image.fromarray(raw_gz1)
            ang = (atan2(zn[i % (xFrames-1)].imag, zn[i % (xFrames-1)].real))*180/np.pi
            rotate_gz1 = ndimage.rotate(imgPIL1, ang, reshape=False)            
            rotate_gz1 = self.cropZoom(rotate_gz1, 2)

            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgNormalizeNm = imgRndSelNm+strInc+'.jpg'
            imgRndSelFull = imgOutDir+imgNormalizeNm
            misc.imsave(imgRndSelFull, rotate_gz1)

            if (nextInc % numImg-1) == 0:
                xfdirection = -xfdirection
            if xfdirection == 1:
                idx += 1                
            else:
                idx -= 1
           
            nextInc += 1
            
        #return [imgRndSelArray, imgRndSelNmArray]
        return    
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Random Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgRndSel(self, imgList, numFrames, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgRndSelNm = imgOutNm
        else:
            imgRndSelNm = 'imgRndSelOut'
    
        # imgRndSelNmArray = []
        # imgRndSelArray = []
        
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
            
        #return [imgRndSelArray, imgRndSelNmArray]
        return
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image BPM Random Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgRndBpm(self, imgList, numFrames, frameHold, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgRndSelNm = imgOutNm
        else:
            imgRndSelNm = 'imgRndSelOut'
    
        #imgRndSelNmArray = []
        #imgRndSelArray = []
        
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
            
        #return [imgRndSelArray, imgRndSelNmArray]
        return
        
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image BPM Random Select Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgRndLfo(self, imgList, numFrames, ctrlLfo, imgOutDir, imgOutNm='None'):
    
        if imgOutNm != 'None':
            imgRndSelNm = imgOutNm
        else:
            imgRndSelNm = 'imgRndSelOut'
    
        #imgRndSelNmArray = []
        #imgRndSelArray = []
        
        rIdxArray = randomIdxArray(numFrames, len(imgList))
        
        
        # generate an LFO sin control signal - 1 cycle per xfadeFrames
        T = 1.0 / framesPerSec    # sample period
        LFO_T = 8 * xfadeFrames * T
        # create composite sin source
        x = np.linspace(0.0, (xfadeFrames+1)*T, xfadeFrames+1)
        x = x[0:len(x)-1]
        LFOAMP = 200
        LFO = LFOAMP*np.sin(LFO_T * 2.0*np.pi * x)        
        
        
        
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
            
        #return [imgRndSelArray, imgRndSelNmArray]
        return        
        
    
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
    
        return
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image Rotate Sequence Algorithm::---*')
    # // *--------------------------------------------------------------* //
    
    def odmkImgRotateAlt(self, imgFileList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of rotated images (static img input)
            360 deg - period = numFrames
            (ndimage.rotate: ex, rotate image by 45 deg:
             rotate_gz1 = ndimage.rotate(gz1, 45, reshape=False)) '''
    
        if imgOutNm != 'None':
            imgRotateSeqNm = imgOutNm
        else:
            imgRotateSeqNm = 'imgRotateSeqOut'
            
        imgObjTemp1 = misc.imread(imgFileList[0])  
        

        SzX = imgObjTemp1.shape[1]
        SzY = imgObjTemp1.shape[0]
        
        numImg = len(imgFileList)

        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
        # finalXfade = int(floor(numFrames / xfadeFrames))
        numFinalXfade = int(ceil(numFrames - (floor(numFrames / xfadeFrames) * xfadeFrames)))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---Total number of frames = '+str(numFrames)+'---*')
        print('// *---number of img per xfade = '+str(xfadeFrames)+'---*')
        print('// *---number of xfades = '+str(numXfades)+'---*')
        print('// *---number of img for Final xfade = '+str(numFinalXfade)+'---*')
        print('\nProcessing ... ... ...\n\n')

    
        zn = cyclicZn((xfadeFrames*4)-1)    # less one, then repeat zn[0] for full 360
    
    
        imgCount = numFrames
        n_digits = int(ceil(np.log10(imgCount))) + 2
        nextInc = 0
        for i in range(numFrames):
            
            if i%(xfadeFrames*4)==0:
                gz1 = misc.imread(imgFileList[(2*i+90) % numImg])
                gz2 = misc.imread(imgFileList[(2*i+1+90) % numImg])
            #gz2 = misc.imread(imgFileList[(2*i+1) % numImg])
            
            ang = (atan2(zn[i % ((xfadeFrames*4)-1)].imag, zn[i % ((xfadeFrames*4)-1)].real))*180/np.pi
            #if ((i % 2) == 0):
            if ((i % 4) == 1 or (i % 4) == 2):
                rotate_gz = ndimage.rotate(gz1, ang, reshape=False)
                #zoom_gz1 = self.cropZoom(rotate_gz1, 2)
            else:
                rotate_gz = ndimage.rotate(gz2, -ang, reshape=False)
                #zoom_gz1 = self.cropZoom(rotate_gz1, 2)
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            imgRotateSeqFull = imgOutDir+imgRotateSeqNm+strInc+'.jpg'
            misc.imsave(imgRotateSeqFull, rotate_gz)
    
        return    
    
    
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
    
        # imgXfadeNmArray = []
        # imgXfadeArray = []
    
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
                # imgXfadeNmArray.append(imgXfadeFull)
                # imgXfadeArray.append(alphaB)
    
        #return [imgXfadeArray, imgXfadeNmArray]   
        return
    
    
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image solarize CrossFade Sequence::---*')
    # // *--------------------------------------------------------------* //    
        
    def odmkImgSolXfade(self, imgList, sLength, framesPerSec, xFrames, imgOutDir, imgOutNm='None'):
        ''' outputs a sequence of images fading from img1 -> img2
            Linearly alpha-blends images using PIL Image.blend
            assume images in imgList are numpy arrays'''
    
        if imgOutNm != 'None':
            imgXfadeNm = imgOutNm
        else:
            imgXfadeNm = 'imgSolXfadeOut'
    
        numFrames = int(ceil(sLength * framesPerSec))
        numXfades = int(ceil(numFrames / xFrames))
    
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---xFrames = '+str(xFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgList)

        n_digits = int(ceil(np.log10(numFrames))) + 2
        solarX = np.linspace(5.0, 255.0, xFrames)

        frameCnt = 0
        indexCnt = 0
        xfdirection = -1
        for j in range(numXfades - 1):
            if frameCnt <= numFrames:
                if (j % int(numImg/3)) == 0:
                    xfdirection = -xfdirection
                if xfdirection == 1:
                    indexCnt += 1
                    imgPIL1 = Image.fromarray(imgList[3*indexCnt])
                else:
                    indexCnt -= 1
                    imgPIL1 = Image.fromarray(imgList[3*indexCnt])
                #pdb.set_trace()
                # imgPIL2 = Image.fromarray(imgList[j + 1])
                for i in range(xFrames):
                    #solarB = ImageOps.solarize(imgPIL1, solarX[xFrames-i-1])
                    solarB = ImageOps.solarize(imgPIL1, solarX[i])
                    solarB = ImageOps.autocontrast(solarB, cutoff=0)
                    #solarB = ImageOps.autocontrast(solarB, cutoff=0)
                    zr = ''
                    for k in range(n_digits - len(str(frameCnt))):
                        zr += '0'
                    strInc = zr+str(frameCnt)
                    imgXfadeFull = imgOutDir+imgXfadeNm+strInc+'.jpg'
                    misc.imsave(imgXfadeFull, solarB)
                    frameCnt += 1
    
        return
    
    
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
                    rotate_alphaZ = self.cropZoom(rotate_alphaB, 2)

        
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    imgXfadeRotFull = imgOutDir+imgXfadeRotNm+strInc+'.jpg'
                    misc.imsave(imgXfadeRotFull, rotate_alphaZ)
                    frameCnt += 1
    
        return
      

    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - Image CrossFade Rotate Sequence Algorithm::---*
    # // *--------------------------------------------------------------* //
    
    def odmkImgXfadeShift(self, imgList, sLength, framesPerSec, xFrames, imgOutDir, imgOutNm='None', rotDirection=0):
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
    
        # imgXfadeRotNmArray = []
        # imgXfadeRotArray = []
    
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
                    # imgXfadeRotNmArray.append(imgXfadeRotFull)
                    # imgXfadeRotArray.append(alphaB)
                    frameCnt += 1
    
        # return [imgXfadeRotArray, imgXfadeRotNmArray]
        return      
      
      
      
    
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

    #def odmkPxlRndReplace(self, imgArray, sLength, framesPerSec, frameCtrl, imgOutDir, imgOutNm='None'):
    def odmkPxlRndReplace(self, imgArray, sLength, framesPerSec, imgOutDir, imgOutNm='None'):
        ''' swap n pixels per frame from consecutive images'''
    
        if imgOutNm != 'None':
            pxlRndReplaceNm = imgOutNm
        else:
            pxlRndReplaceNm = 'pxlRndReplace'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(sLength * framesPerSec))
        #numXfades = int(ceil(numFrames / frameCtrl))

        print('// *---numFrames = '+str(numFrames)+'---*')
        #print('// *---FrameCtrl = '+str(frameCtrl)+'---*')
        #print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgArray)
        #numxfadeImg = numImg * frameCtrl
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0

        eyeBox1 = Image.fromarray(imgArray[randomIdx(numImg)])
        
        frameCnt = 0

        for j in range(numFrames):

            imgPIL1 = Image.fromarray(imgArray[j % (numImg-1)])
          
            boxSzX = round(SzX*random.random()) + 30
            boxSzY = round(SzY*random.random()) + 30

            rndPxlLoc = randomPxlLoc(SzX, SzY)
            alpha = random.random()

            eyeBox1 = self.eyeBox1(eyeBox1, imgPIL1, boxSzX, boxSzY, rndPxlLoc, alpha)



            # update name counter & concat with base img name    
            nextInc += 1
            zr = ''
            for j in range(n_digits - len(str(nextInc))):
                zr += '0'
            strInc = zr+str(nextInc)
            pxlRndReplaceFull = imgOutDir+pxlRndReplaceNm+strInc+'.jpg'
            
            # write img to disk
            misc.imsave(pxlRndReplaceFull, eyeBox1)
            
            frameCnt += 1
    
        return
  
  
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Pixel Random Replace::---*
    # // *--------------------------------------------------------------* //

    #def odmkPxlRndReplace(self, imgArray, sLength, framesPerSec, frameCtrl, imgOutDir, imgOutNm='None'):
    def odmkPxlRndRotate(self, imgArray, sLength, framesPerSec, framesPerBeat, imgOutDir, imgOutNm='None'):
        ''' swap n pixels per frame from consecutive images'''
    
        if imgOutNm != 'None':
            pxlRndReplaceNm = imgOutNm
        else:
            pxlRndReplaceNm = 'pxlRndReplace'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(sLength * framesPerSec))

        framesPerBeat = int(2 * framesPerBeat)

        numBeats = int(ceil(numFrames / framesPerBeat))


        print('// *---numFrames = '+str(numFrames)+'---*')
        #print('// *---FrameCtrl = '+str(frameCtrl)+'---*')
        #print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgArray)
        #numxfadeImg = numImg * frameCtrl
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0

        eyeBase1 = Image.fromarray(imgArray[randomIdx(numImg)])
    
        #zn = cyclicZn(framesPerBeat-1)    # less one, then repeat zn[0] for full 360
        zn = cyclicZn(2*framesPerBeat)    # less one, then repeat zn[0] for full 360

        zoomScalarXArr = np.linspace(16, SzX//3, framesPerBeat) 
        zoomScalarYArr = np.linspace(9, SzY//3, framesPerBeat)
            
        alphaX = np.linspace(0.0001, 1.0, framesPerBeat)



        frameCnt = 0
        for j in range(numBeats - 1):
            
            imgZoom1 = Image.fromarray(imgArray[randomIdx(numImg)])
            imgZoom2 = Image.fromarray(imgArray[randomIdx(numImg)])
            zoomVertex1 = randomPxlLoc(SzX, SzY)
            rotDirection = round(random.random())
            
            if frameCnt <= numFrames:
                
                for i in range(framesPerBeat):

                    # background image base (pxlRndReplace)
                    #----------------------------------------------------------
                    
                    boxSzX = round(SzX*random.random()) + 16
                    boxSzY = round(SzY*random.random()) + 9                    
 
                    rndPxlLoc = randomPxlLoc(SzX, SzY)
                    alpha = random.random()

                    # imgPIL1 = Image.fromarray(imgArray[i % (numImg-1)])
                    imgPIL1 = Image.fromarray(imgArray[randomIdx(numImg)])
    
                    eyeBase1 = self.eyeBox1(eyeBase1, imgPIL1, boxSzX, boxSzY, rndPxlLoc, alpha)

                    #----------------------------------------------------------

                    # setup crop coordinates for Zoom sub image
                    # coordinates = (left, bottom, right, top)

                    if (zoomVertex1[0] - int(zoomScalarXArr[i])) < 0:
                        zBoxLeft = 0
                    else:
                        zBoxLeft = zoomVertex1[0] - int(zoomScalarXArr[i])

                    if (zoomVertex1[1] - int(zoomScalarYArr[i])) < 0:
                        zBoxBottom = 0
                    else:
                        zBoxBottom = zoomVertex1[1] - int(zoomScalarYArr[i])                   

                    if (zoomVertex1[0] + int(zoomScalarXArr[i])) > SzX:
                        zBoxRight = SzX
                    else:
                        zBoxRight = zoomVertex1[0] + int(zoomScalarXArr[i])

                    if (zoomVertex1[1] + int(zoomScalarYArr[i])) > SzX:
                        zBoxTop = SzX
                    else:
                        zBoxTop = zoomVertex1[1] + int(zoomScalarYArr[i])

                   
                    boxZoom = (int(zBoxLeft), int(zBoxBottom), int(zBoxRight), int(zBoxTop))
                    
                    subSzX = boxZoom[2] - boxZoom[0]
                    subSzY = boxZoom[3] - boxZoom[1]
                    
                    
                    subLeft = int(SzX//2 - subSzX//2)
                    subBottom = int(SzY//2 - subSzY//2)
                    subRight = subLeft + subSzX
                    subTop = subBottom + subSzY
                    
                    subBox = (subLeft, subBottom, subRight, subTop)
                    
                    #pdb.set_trace()                  
                    
                    #eyeSub1 = imgZoom1.crop(boxZoom) 
                    #eyeSub2 = imgZoom2.crop(boxZoom)
                    eyeSub1 = imgZoom1.crop(subBox) 
                    eyeSub2 = imgZoom2.crop(subBox)

                    
                    alphaB = Image.blend(eyeSub1, eyeSub2, alphaX[i])
                    
                    alphaB = np.array(alphaB)
                    ang = (atan2(zn[i % (framesPerBeat-1)].imag, zn[i % (framesPerBeat-1)].real))*180/np.pi
                    if rotDirection == 1:
                        ang = -ang

                    rotate_alphaB = ndimage.rotate(alphaB, ang, reshape=False)

                    
                    rotate_alphaZ = self.cropZoom(rotate_alphaB, 2)
                    rotate_alphaZ = Image.fromarray(rotate_alphaZ)
                    

                    #----------------------------------------------------------
                    
                    eyeBase1.paste(rotate_alphaZ, boxZoom)


                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    pxlRndReplaceFull = imgOutDir+pxlRndReplaceNm+strInc+'.jpg'
                    
                    # write img to disk
                    misc.imsave(pxlRndReplaceFull, eyeBase1)
            
                    frameCnt += 1
    
        return  
  
  
   
    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Dual Bipolar Oscillating telescope f::---*
    # // *--------------------------------------------------------------* //
    
    def odmkLfoParascope(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0):
        ''' lfoParascope function '''
    
        if imgOutNm != 'None':
            imgLfoParascopeNm = imgOutNm
        else:
            imgLfoParascopeNm = 'imgLfoParascope'
    
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
        #hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
        hop_sz = 1
        
        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
        imgRndIdxArray = randomIdxArray(numFrames, numImg)
   
        # imgDualBipolarTelescNmArray = []
        # imgDualBipolarTelescArray = []

        # generate an LFO sin control signal - 1 cycle per xfadeFrames
        T = 1.0 / framesPerSec    # sample period
        LFO_T = 8 * xfadeFrames * T
        # create composite sin source
        x = np.linspace(0.0, (xfadeFrames+1)*T, xfadeFrames+1)
        x = x[0:len(x)-1]
        LFOAMP = 200
        LFO = LFOAMP*np.sin(LFO_T * 2.0*np.pi * x)

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
            imgBpTsc1 = imgClone1.copy()
            imgBpTsc2 = imgClone2.copy()
            imgBpTscOut = imgClone1.copy()

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
#            focalRnd1 = randomPxlLoc(SzX, SzY)
#            if (focalRnd1[0] < SzX/2):
#                offsetX = -int(SzX/2 - focalRnd1[0])
#            else:
#                offsetX = int(focalRnd1[0] - SzX/2)
#            if (focalRnd1[1] < SzY/2):
#                offsetY = -int(SzY/2 - focalRnd1[1])
#            else:
#                offsetY = int(focalRnd1[1] - SzY/2)

            #print('focalRnd1 = ['+str(focalRnd1[0])+', '+str(focalRnd1[1])+']')
            #print('offsetX = '+str(offsetX))
            #print('offsetY = '+str(offsetY))

            # focalRnd2 = randomPxlLoc(SzX, SzY)


            # functionalize then call for each telescope    
            for t in range(xfadeFrames):
                if frameCnt < numFrames:    # process until end of total length

                    print('LFO of t = '+str(int(LFO[t])))
                    hop_sz_mod = LFO[t] * hop_sz

                    if newDimX > hop_sz_mod:
                        #newDimX -= 2*hop_sz
                        newDimX -= hop_sz_mod
                        newDimX = int(newDimX)
                    if newDimY > hop_sz_mod:
                        #newDimY -= 2*hop_sz
                        newDimY -= hop_sz_mod
                        newDimY = int(newDimY)
                           
                    #print('*****LFO[t] = '+str(LFO[t])+', hop_sz_mod = '+str(hop_sz_mod)+', newDimX = '+str(newDimX)+', newDimY = '+str(newDimY))
                        
                    #pdb.set_trace()  
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgClone1, newDimX, newDimY)
                    imgItr2 = self.odmkEyeRescale(imgClone2, newDimX, newDimY)                    

                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    r = 0
                    for j in range(SzY):
                        s = 0
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                            if ( (j >= (SzY-newDimY)/2) and (j < (SzY - (SzY-newDimY)/2)) and (k >= (SzX-newDimX)/2) and (k < (SzX - (SzX-newDimX)/2)) ):
                                                                
                                #print('*****j = '+str(j)+'; k = '+str(k)+'; r = '+str(r)+'; s = '+str(s))
                                imgBpTsc1[j, k, :] = imgItr1[r, s, :]
                                imgBpTsc2[j, k, :] = imgItr2[r, s, :]
                                
                            if (k >= (SzX-newDimX)/2) and (k < (SzX - (SzX-newDimX)/2)):
                                s += 1
                        if (j >= (SzY-newDimY)/2) and (j < (SzY - (SzY-newDimY)/2)):
                            r += 1                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
                    #imgBpTscOut = self.eyePhiFrame1(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2))
                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)

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
                    imgLfoParascopeFull = imgOutDir+imgLfoParascopeNm+strInc+'.jpg'
                    misc.imsave(imgLfoParascopeFull, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)
                    
                    frameCnt += 1
    
        return
        # return [imgLfoParascopeArray, imgLfoParascopeNmArray]   


    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Bananasloth Recursion::---*
    # // *--------------------------------------------------------------* //
    
    def odmkBSlothRecurs(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', indexOffset='None'):
        ''' BSlothRecurs function '''
    
        if imgOutNm != 'None':
            imgBSlothRecursNm = imgOutNm
        else:
            imgBSlothRecursNm = 'imgBSlothRecurs'
            
            
#        if indexOffset != 'None':
#            idxOffset = indexOffset
#        else:
#            idxOffset = 0       
            
    
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
        #hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
        #hop_sz = 5
    
        imgRndIdxArray = randomIdxArray(numFrames, numImg)   

        # imgBSlothRecursNmArray = []
        # imgBSlothRecursArray = []

        frameCnt = 0
        xfdirection = 1    # 1 = forward; -1 = backward
        for j in range(numXfades):
            
            hop_sz = round(16*random.random())
            if hop_sz == 0:
                hop_sz = 5      # avoid zero - 5 is good default
            
            
            inOrOut = round(random.random())        
            
            # forword <-> backward img grap from rnd img dir
            if (frameCnt % numXfades) == 0:
                xfdirection = -xfdirection
            if xfdirection == 1:
                imgClone1 = imgArray[imgRndIdxArray[j]]
                imgClone2 = imgArray[imgRndIdxArray[j+1]]                
            else:
                imgClone1 = imgArray[imgRndIdxArray[(numImg-2) - j]]
                imgClone2 = imgArray[imgRndIdxArray[(numImg-2) - j+1]]

                
            # initialize output
            # ***** normally use imgClone1.copy(), but this function abuses the assignment
            imgBpTsc1 = imgClone1
            imgBpTsc2 = imgClone2
            imgBpTscOut = imgClone1

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            pxlLocScaler = 3*(round(7*random.random())+1)
            #pxlLocScaler = 3*(round(5*random.random())+1)
            pxlLocScaler2x = pxlLocScaler * 2
            focalRnd1 = randomPxlLoc(int(SzX/pxlLocScaler), int(SzY/pxlLocScaler))
            if (focalRnd1[0] < SzX/pxlLocScaler2x):
                offsetX = -int(SzX/pxlLocScaler2x - focalRnd1[0])
            else:
                offsetX = int(focalRnd1[0] - SzX/pxlLocScaler2x)
            if (focalRnd1[1] < SzY/pxlLocScaler2x):
                offsetY = -int(SzY/pxlLocScaler2x - focalRnd1[1])
            else:
                offsetY = int(focalRnd1[1] - SzY/pxlLocScaler2x)


            rotateSubFrame1 = 0             # randomly set rotation on/off
            rotDirection1 = 0
            if (round(13*random.random()) > 7):
                offsetX = 0
                offsetY = 0
                rotateSubFrame1 = 1
                hop_sz = round(56*random.random())
                if (round(random.random())):
                    rotDirection1 = 1


            # functionalize then call for each telescope
#            xfadeMpy = round(6*random.random())
#            if xfadeMpy == 0:
#                xfadeMpy = 1
#            xFrames = int(xfadeFrames/xfadeMpy)    # randomly mpy xfadeFrames 1:3
            xFrames = xfadeFrames
            
            alphaX = np.linspace(0.0, 1.0, xFrames)   
            zn = cyclicZn(xFrames-1)    # less one, then repeat zn[0] for full 360            
            
            xCnt  = xFrames
            curInc = nextInc
            
            for t in range(xFrames):
                if frameCnt < numFrames:    # process until end of total length

                    if newDimX > hop_sz + 1:
                        #newDimX -= 2*hop_sz
                        newDimX -= hop_sz
                    if newDimY > hop_sz + 1:
                        #newDimY -= 2*hop_sz
                        newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgBpTsc2, newDimX, newDimY)
                    if rotateSubFrame1 == 1:
                        
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection1 == 1:
                            ang = -ang
                            
                        imgItr1 = ndimage.rotate(imgItr1, ang, reshape=False)
                        
                        imgItr1 = self.cropZoom(imgItr1, 2)

                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    if rotateSubFrame1 == 1:
                    
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection1 == 0:
                            ang = -ang
                            
                        imgItr2 = ndimage.rotate(imgItr2, ang, reshape=False)
                        
                        imgItr2 = self.cropZoom(imgItr2, 2)


                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    for j in range(SzY):
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                            if ( (j >= (SzY-newDimY)/2) and (j < (SzY - (SzY-newDimY)/2)) and (k >= (SzX-newDimX)/2) and (k < (SzX - (SzX-newDimX)/2)) ):                                
                                
                                #imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                #imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]

                                    
                                if ( (j+offsetY >= 0)  and (j+offsetY < SzY) and (k+offsetX >= 0)  and (k+offsetX < SzX) ):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                    
                                    

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # inOrOut controls direction of parascope func
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        for j in range(n_digits - len(str(curInc + xCnt))):
                            zr += '0'
                        strInc = zr+str(curInc + xCnt)
                    if self.imgFormat == 'fjpg':
                        imgBSlothRecursFull = imgOutDir+imgBSlothRecursNm+strInc+'.jpg'
                    else:
                        imgBSlothRecursFull = imgOutDir+imgBSlothRecursNm+strInc+'.bmp'                            
                    misc.imsave(imgBSlothRecursFull, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)
                    
                    xCnt -= 1
                    frameCnt += 1
    
        return
        # return [imgBSlothRecursArray, imgBSlothRecursNmArray]



    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Bananasloth Recursion::---*
    # // *--------------------------------------------------------------* //
    
    def odmkBSlothGlitch2(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0, indexOffset='None'):
        ''' BSlothGlitch1 function '''
    
        if imgOutNm != 'None':
            imgBSlothGlitch1Nm = imgOutNm
        else:
            imgBSlothGlitch1Nm = 'imgBSlothGlitch1'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
        # finalXfade = int(floor(numFrames / xfadeFrames))
        numFinalXfade = int(ceil(numFrames - (floor(numFrames / xfadeFrames) * xfadeFrames)))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---Total number of frames = '+str(numFrames)+'---*')
        print('// *---number of img per xfade = '+str(xfadeFrames)+'---*')
        print('// *---number of xfades = '+str(numXfades)+'---*')
        print('// *---number of img for Final xfade = '+str(numFinalXfade)+'---*')
        print('\nProcessing ... ... ...\n\n')
    
        numImg = len(imgArray)
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
    
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
            # ***** normally use imgClone1.copy(), but this function abuses the assignment
            imgBpTsc1 = imgClone1
            imgBpTsc2 = imgClone2
            imgBpTscOut = imgClone1

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            focalRnd1 = randomPxlLoc(int(SzX/2), int(SzY))
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

            rotateSubFrame1 = 0             # randomly set rotation on/off
            rotDirection1 = 0
            if (round(8*random.random()) > 6):
                rotateSubFrame1 = 1
                if (round(random.random())):
                    rotDirection1 = 1

            rotateSubFrame2 = 0             # randomly set rotation on/off
            rotDirection2 = 0            
            if (round(8*random.random()) > 6):
                rotateSubFrame2 = 1
                if (round(random.random())):
                    rotDirection2 = 1
                    
            
            # functionalize then call for each telescope
#            xfadeMpy = round(6*random.random())
#            if xfadeMpy == 0:
#                xfadeMpy = 1
#            xFrames = int(xfadeFrames/xfadeMpy)    # randomly mpy xfadeFrames 1:3
            xFrames = xfadeFrames            
            
            
            alphaX = np.linspace(0.0, 1.0, xFrames)   
            #zn = cyclicZn(2*xFrames-1)    # less one, then repeat zn[0] for full 360 
            zn = cyclicZn(2*xFrames)    # less one, then repeat zn[0] for full 360 

            # handles final xfade 
            if j == numXfades-1:
                xCnt = numFinalXfade
            else:
                xCnt = xFrames
            curInc = nextInc
            
            for t in range(xFrames):
                if frameCnt < numFrames:    # process until end of total length

                    if newDimX > hop_sz:
                        newDimX -= 3*hop_sz
                        #newDimX -= hop_sz
                    if newDimY > hop_sz:
                        newDimY -= 3*hop_sz
                        #newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgBpTsc2, newDimX, newDimY)
                    if rotateSubFrame1 == 1:
                        
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection1 == 1:
                            ang = -ang
                            
                        imgItr1 = ndimage.rotate(imgItr1, ang, reshape=False)
                        imgItr1 = self.cropZoom(imgItr1, 2)
                        #imgItr1 = Image.fromarray(imgItr1)

                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    if rotateSubFrame2 == 1:
                    
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection2 == 1:
                            ang = -ang
                            
                        imgItr2 = ndimage.rotate(imgItr2, ang, reshape=False)
                        imgItr2 = self.cropZoom(imgItr2, 2)
                        #imgItr2 = Image.fromarray(imgItr2)

                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    for j in range(SzY):
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                           # ***** FIXIT ***** - refer to 23 above
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                
                                # imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                # imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # inOrOut controls direction of parascope func
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        #for j in range(n_digits - len(str(numFrames - (nextInc)))):
                        for j in range(n_digits - len(str(curInc + xCnt))):
                            zr += '0'
                        strInc = zr+str(curInc + xCnt)
                    imgBSlothGlitch1Full = imgOutDir+imgBSlothGlitch1Nm+strInc+'.jpg'
                    misc.imsave(imgBSlothGlitch1Full, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)

                    xCnt -= 1                    
                    frameCnt += 1
                    
        print('\n// *---Final Frame Count = '+str(frameCnt)+'---*')
    
        return



    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Bananasloth Recursion::---*
    # // *--------------------------------------------------------------* //
    
    def odmkBSlothGlitch3(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0, indexOffset='None'):
        ''' BSlothGlitch3 function '''
    
        if imgOutNm != 'None':
            imgBSlothGlitch3Nm = imgOutNm
        else:
            imgBSlothGlitch3Nm = 'imgBSlothGlitch3'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
        # finalXfade = int(floor(numFrames / xfadeFrames))
        numFinalXfade = int(ceil(numFrames - (floor(numFrames / xfadeFrames) * xfadeFrames)))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---Total number of frames = '+str(numFrames)+'---*')
        print('// *---number of img per xfade = '+str(xfadeFrames)+'---*')
        print('// *---number of xfades = '+str(numXfades)+'---*')
        print('// *---number of img for Final xfade = '+str(numFinalXfade)+'---*')
        print('\nProcessing ... ... ...\n\n')
    
        numImg = len(imgArray)
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
    
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
            # ***** normally use imgClone1.copy(), but this function abuses the assignment
            imgBpTsc1 = imgClone1
            imgBpTsc2 = imgClone2
            imgBpTscOut = imgClone1

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            focalRnd1 = randomPxlLoc(int(SzX/2), int(SzY))
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

            rotateSubFrame1 = 0             # randomly set rotation on/off
            rotDirection1 = 0
            if (round(8*random.random()) > 5):
                rotateSubFrame1 = 1
                if (round(random.random())):
                    rotDirection1 = 1

            rotateSubFrame2 = 0             # randomly set rotation on/off
            rotDirection2 = 0            
            if (round(8*random.random()) > 2):
                rotateSubFrame2 = 1
                if (round(random.random())):
                    rotDirection2 = 1
                    
            
            # functionalize then call for each telescope
#            xfadeMpy = round(6*random.random())
#            if xfadeMpy == 0:
#                xfadeMpy = 1
#            xFrames = int(xfadeFrames/xfadeMpy)    # randomly mpy xfadeFrames 1:3
            xFrames = xfadeFrames            
            
            
            alphaX = np.linspace(0.0, 1.0, xFrames)   
            #zn = cyclicZn(2*xFrames-1)    # less one, then repeat zn[0] for full 360 
            zn = cyclicZn(2*xFrames)    # less one, then repeat zn[0] for full 360 

            # handles final xfade 
            if j == numXfades-1:
                xCnt = numFinalXfade
            else:
                xCnt = xFrames
            curInc = nextInc
            
            for t in range(xFrames):
                if frameCnt < numFrames:    # process until end of total length

                    if newDimX > hop_sz:
                        newDimX -= 9*hop_sz
                        #newDimX -= hop_sz
                    if newDimY > hop_sz:
                        newDimY -= 7*hop_sz
                        #newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgBpTsc2, newDimX, newDimY)
                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    if rotateSubFrame1 == 1:
                        
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection1 == 1:
                            ang = -ang
                            
                        imgItr1 = ndimage.rotate(imgItr2, ang, reshape=False)
                        imgItr1 = self.cropZoom(imgItr1, 2)
                        #imgItr1 = Image.fromarray(imgItr1)

                    if rotateSubFrame2 == 1:
                    
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection2 == 1:
                            ang = -ang
                            
                        imgItr2 = ndimage.rotate(imgItr1, ang, reshape=False)
                        imgItr2 = self.cropZoom(imgItr2, 2)
                        #imgItr2 = Image.fromarray(imgItr2)

                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    for j in range(SzY):
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                           # ***** FIXIT ***** - refer to 23 above
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                
                                # imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                # imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # inOrOut controls direction of parascope func
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        #for j in range(n_digits - len(str(numFrames - (nextInc)))):
                        for j in range(n_digits - len(str(curInc + xCnt))):
                            zr += '0'
                        strInc = zr+str(curInc + xCnt)
                    imgBSlothGlitch3Full = imgOutDir+imgBSlothGlitch3Nm+strInc+'.jpg'
                    misc.imsave(imgBSlothGlitch3Full, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)

                    xCnt -= 1                    
                    frameCnt += 1
                    
        print('\n// *---Final Frame Count = '+str(frameCnt)+'---*')
    
        return



    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE Bananasloth Recursion::---*
    # // *--------------------------------------------------------------* //
    
    def odmkBSlothGlitch1(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0, indexOffset='None'):
        ''' BSlothGlitch1 function '''
    
        if imgOutNm != 'None':
            imgBSlothGlitch1Nm = imgOutNm
        else:
            imgBSlothGlitch1Nm = 'imgBSlothGlitch1'
    
        SzX = imgArray[0].shape[1]
        SzY = imgArray[0].shape[0]
    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
        # finalXfade = int(floor(numFrames / xfadeFrames))
        numFinalXfade = int(ceil(numFrames - (floor(numFrames / xfadeFrames) * xfadeFrames)))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---Total number of frames = '+str(numFrames)+'---*')
        print('// *---number of img per xfade = '+str(xfadeFrames)+'---*')
        print('// *---number of xfades = '+str(numXfades)+'---*')
        print('// *---number of img for Final xfade = '+str(numFinalXfade)+'---*')
        print('\nProcessing ... ... ...\n\n')
    
        numImg = len(imgArray)
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
    
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
            # ***** normally use imgClone1.copy(), but this function abuses the assignment
            imgBpTsc1 = imgClone1
            imgBpTsc2 = imgClone2
            imgBpTscOut = imgClone1

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            focalRnd1 = randomPxlLoc(int(SzX/2), int(SzY))
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

            rotateSubFrame1 = 0             # randomly set rotation on/off
            rotDirection1 = 0
            if (round(8*random.random()) > 6):
                rotateSubFrame1 = 1
                if (round(random.random())):
                    rotDirection1 = 1

            rotateSubFrame2 = 0             # randomly set rotation on/off
            rotDirection2 = 0            
            if (round(8*random.random()) > 6):
                rotateSubFrame2 = 1
                if (round(random.random())):
                    rotDirection2 = 1
                    
            
            # functionalize then call for each telescope
#            xfadeMpy = round(6*random.random())
#            if xfadeMpy == 0:
#                xfadeMpy = 1
#            xFrames = int(xfadeFrames/xfadeMpy)    # randomly mpy xfadeFrames 1:3
            xFrames = xfadeFrames            
            
            
            alphaX = np.linspace(0.0, 1.0, xFrames)   
            zn = cyclicZn(xFrames-1)    # less one, then repeat zn[0] for full 360            

            # handles final xfade 
            if j == numXfades-1:
                xCnt = numFinalXfade
            else:
                xCnt = xFrames
            curInc = nextInc
            
            for t in range(xFrames):
                if frameCnt < numFrames:    # process until end of total length

                    if newDimX > hop_sz:
                        #newDimX -= 2*hop_sz
                        newDimX -= hop_sz
                    if newDimY > hop_sz:
                        #newDimY -= 2*hop_sz
                        newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    imgItr1 = self.odmkEyeRescale(imgBpTsc2, newDimX, newDimY)
                    if rotateSubFrame1 == 1:
                        
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection1 == 1:
                            ang = -ang
                            
                        imgItr1 = ndimage.rotate(imgItr1, ang, reshape=False)
                        imgItr1 = self.cropZoom(imgItr1, 2)
                        #imgItr1 = Image.fromarray(imgItr1)

                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    if rotateSubFrame2 == 1:
                    
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection2 == 1:
                            ang = -ang
                            
                        imgItr2 = ndimage.rotate(imgItr2, ang, reshape=False)
                        imgItr2 = self.cropZoom(imgItr2, 2)
                        #imgItr2 = Image.fromarray(imgItr2)

                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)
                    for j in range(SzY):
                        for k in range(SzX):
                            
                            # calculate bipolar subframes then interleave (out = priority)
                            # will require handling frame edges (no write out of frame)
                            # update index math to handle random focal points
                           
                           # ***** FIXIT ***** - refer to 23 above
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                
                                # imgBpTsc1[j, k, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                # imgBpTsc2[j, k, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # inOrOut controls direction of parascope func
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        #for j in range(n_digits - len(str(numFrames - (nextInc)))):
                        for j in range(n_digits - len(str(curInc + xCnt))):
                            zr += '0'
                        strInc = zr+str(curInc + xCnt)
                    imgBSlothGlitch1Full = imgOutDir+imgBSlothGlitch1Nm+strInc+'.jpg'
                    misc.imsave(imgBSlothGlitch1Full, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)

                    xCnt -= 1                    
                    frameCnt += 1
                    
        print('\n// *---Final Frame Count = '+str(frameCnt)+'---*')
    
        return


    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - ODMKEYE - EYE FxSlothCultLife Glitch::---*
    # // *--------------------------------------------------------------* //
    
    def odmkFxSlothCult(self, imgFileList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0):
        ''' BSlothGlitch1 function '''


        imgObjTemp1 = misc.imread(imgFileList[0])  
        imgObjTemp2 = misc.imread(imgFileList[0])        

        SzX = imgObjTemp1.shape[1]
        SzY = imgObjTemp1.shape[0]
        
        #pdb.set_trace()
        
#        if imgObjTemp2.shape[1] != SzX or imgObjTemp2.shape[0] != SzY:
#            print('ERROR: image arrays must be the same size')
#            return 1
    
        if imgOutNm != 'None':
            imgFxSlothCultNm = imgOutNm
        else:
            imgFxSlothCultNm = 'imgFxSlothCult'


    
        numFrames = int(ceil(xLength * framesPerSec))
        numXfades = int(ceil(numFrames / xfadeFrames))
        numFinalXfade = int(ceil(numFrames - (floor(numFrames / xfadeFrames) * xfadeFrames)))
    
        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
        print('// *---numFrames = '+str(numFrames)+'---*')
        print('// *---numxfadeImg = '+str(xfadeFrames)+'---*')
        print('// *---numXfades = '+str(numXfades)+'---*')    
    
        numImg = len(imgFileList)        
    
        n_digits = int(ceil(np.log10(numFrames))) + 2
        nextInc = 0
        # example internal control signal 
        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
    
 
    
        frameCnt = 0
        for j in range(numXfades):
                                # random img grap from image Array

                
            # initialize output
            # ***** normally use imgClone1.copy(), but this function abuses the assignment
            #imgBpTsc1 = misc.imread(imgFileList[j*xfadeFrames % numImg])
            #imgBpTscOut = misc.imread(imgFileList[j*xfadeFrames % numImg])
            imgBpTsc1 = misc.imread(imgFileList[(j*xfadeFrames+5560) % numImg])
            imgBpTscOut = misc.imread(imgFileList[(j*xfadeFrames+4560) % numImg])

            newDimX = SzX     # start with master img dimensions
            newDimY = SzY
                              
            # calculate X-Y random focal point to launch telescope
            focalRnd1 = randomPxlLoc(int(SzX/2), int(SzY))
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

            rotateSubFrame1 = 0             # randomly set rotation on/off
            rotDirection1 = 0
#            if (round(random.random())):
#                rotateSubFrame1 = 1
#                if (round(random.random())):
#                    rotDirection1 = 1

            rotateSubFrame2 = 0             # randomly set rotation on/off
            rotDirection2 = 0            
#            if (round(random.random())):
#                rotateSubFrame2 = 1
#                if (round(random.random())):
#                    rotDirection2 = 1

            # functionalize then call for each telescope
#            xfadeMpy = round(3*random.random())
#            if xfadeMpy == 0:
#                xfadeMpy = 1
#            xFrames = xfadeFrames*xfadeMpy    # randomly mpy xfadeFrames 1:3
            xFrames = xfadeFrames
            
            
            alphaX = np.linspace(0.0, 1.0, xFrames)   
            zn = cyclicZn(xFrames-1)    # less one, then repeat zn[0] for full 360            

            # handles final xfade 
            if j == numXfades-1:
                xCnt = numFinalXfade
            else:
                xCnt = xFrames
            curInc = nextInc
            
            for t in range(xFrames):
                if frameCnt < numFrames:    # process until end of total length
                
                    #imgBpTsc2 = misc.imread(imgFileList[frameCnt % numImg])
                    imgBpTsc2 = misc.imread(imgFileList[(frameCnt+5560) % numImg])

                    if newDimX > hop_sz:
                        #newDimX -= 2*hop_sz
                        newDimX -= hop_sz
                    if newDimY > hop_sz:
                        #newDimY -= 2*hop_sz
                        newDimY -= hop_sz
                        
                    # scale image to new dimensions
                    #imgObjTemp1 = misc.imread(imgFileList[frameCnt % numImg])
                    imgObjTemp1 = misc.imread(imgFileList[(frameCnt+5560) % numImg])
                    imgItr1 = self.odmkEyeRescale(imgObjTemp1, newDimX, newDimY)
                    if rotateSubFrame1 == 1:
                        
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection1 == 1:
                            ang = -ang
                            
                        imgItr1 = ndimage.rotate(imgItr1, ang, reshape=False)

                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
                    if rotateSubFrame2 == 1:
                    
                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
                        if rotDirection2 == 1:
                            ang = -ang
                            
                        imgItr2 = ndimage.rotate(imgItr2, ang, reshape=False)


                    # region = (left, upper, right, lower)
                    # subbox = (i + 1, i + 1, newDimX, newDimY)

                    for j in range(SzY):
                        for k in range(SzX):
                            
                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
                                
                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                

                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    # inOrOut controls direction of parascope func
                    if inOrOut == 1:
                        for j in range(n_digits - len(str(nextInc))):
                            zr += '0'
                        strInc = zr+str(nextInc)
                    else:
                        for j in range(n_digits - len(str(curInc + xCnt))):
                            zr += '0'
                        strInc = zr+str(curInc + xCnt)
                    imgFxSlothCultFull = imgOutDir+imgFxSlothCultNm+strInc+'.jpg'
                    misc.imsave(imgFxSlothCultFull, imgBpTscOut)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)

                    xCnt -= 1                    
                    frameCnt += 1
    
        return
        # return [imgBSlothRecursArray, imgBSlothRecursNmArray]


#    # // *--------------------------------------------------------------* //
#    # // *---::ODMKEYE - ODMKEYE - EYE FxSlothCultLife Glitch::---*
#    # // *--------------------------------------------------------------* //
#    
#    def odmkFxSlothCult(self, imgSrcDir1, imgSrcDir2, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOrOut=0):
#        ''' BSlothGlitch1 function '''
#
#
#        imgFileList1 = []
#        imgFileList2 = []
#        imgFileList1.extend(sorted(glob.glob(imgSrcDir1+'*')))
#        imgFileList2.extend(sorted(glob.glob(imgSrcDir2+'*')))
#
#        imgObjTemp1 = misc.imread(imgFileList1[0])  
#        imgObjTemp2 = misc.imread(imgFileList1[0])        
#
#        SzX = imgObjTemp1.shape[1]
#        SzY = imgObjTemp1.shape[0]
#        
#        #pdb.set_trace()
#        
#        if imgObjTemp2.shape[1] != SzX or imgObjTemp2.shape[0] != SzY:
#            print('ERROR: image arrays must be the same size')
#            return 1
#    
#        if imgOutNm != 'None':
#            imgFxSlothCultNm = imgOutNm
#        else:
#            imgFxSlothCultNm = 'imgFxSlothCult'
#
#
#    
#        numFrames = int(ceil(xLength * framesPerSec))
#        numXfades = int(ceil(numFrames / xfadeFrames))
#        numFinalXfade = int(ceil(numFrames - (floor(numFrames / xfadeFrames) * xfadeFrames)))
#    
#        print('// *---source image dimensions = '+str(SzX)+' x '+str(SzY))
#        print('// *---numFrames = '+str(numFrames)+'---*')
#        print('// *---numxfadeImg = '+str(xfadeFrames)+'---*')
#        print('// *---numXfades = '+str(numXfades)+'---*')    
#    
#        numImg1 = len(imgFileList1)
#        numImg2 = len(imgFileList2)        
#    
#        n_digits = int(ceil(np.log10(numFrames))) + 2
#        nextInc = 0
#        # example internal control signal 
#        hop_sz = ceil(np.log(SzX/xfadeFrames))  # number of pixels to scale img each iteration
#    
# 
#    
#        frameCnt = 0
#        for j in range(numXfades):
#                                # random img grap from image Array
#
#                
#            # initialize output
#            # ***** normally use imgClone1.copy(), but this function abuses the assignment
#            imgBpTsc1 = misc.imread(imgFileList2[j*xfadeFrames % numImg2])
#            imgBpTscOut = misc.imread(imgFileList2[j*xfadeFrames % numImg2])
#
#            newDimX = SzX     # start with master img dimensions
#            newDimY = SzY
#                              
#            # calculate X-Y random focal point to launch telescope
#            focalRnd1 = randomPxlLoc(int(SzX/2), int(SzY))
#            if (focalRnd1[0] < SzX/2):
#                offsetX = -int(SzX/2 - focalRnd1[0])
#            else:
#                offsetX = int(focalRnd1[0] - SzX/2)
#            if (focalRnd1[1] < SzY/2):
#                offsetY = -int(SzY/2 - focalRnd1[1])
#            else:
#                offsetY = int(focalRnd1[1] - SzY/2)
#
#            #print('focalRnd1 = ['+str(focalRnd1[0])+', '+str(focalRnd1[1])+']')
#            #print('offsetX = '+str(offsetX))
#            #print('offsetY = '+str(offsetY))
#
#            # focalRnd2 = randomPxlLoc(SzX, SzY)
#
#            rotateSubFrame1 = 0             # randomly set rotation on/off
#            rotDirection1 = 0
##            if (round(random.random())):
##                rotateSubFrame1 = 1
##                if (round(random.random())):
##                    rotDirection1 = 1
#
#            rotateSubFrame2 = 0             # randomly set rotation on/off
#            rotDirection2 = 0            
##            if (round(random.random())):
##                rotateSubFrame2 = 1
##                if (round(random.random())):
##                    rotDirection2 = 1
#
#            # functionalize then call for each telescope
#            xfadeMpy = round(3*random.random())
#            if xfadeMpy == 0:
#                xfadeMpy = 1
#            xFrames = xfadeFrames*xfadeMpy    # randomly mpy xfadeFrames 1:3
#            
#            alphaX = np.linspace(0.0, 1.0, xFrames)   
#            zn = cyclicZn(xFrames-1)    # less one, then repeat zn[0] for full 360            
#
#            # handles final xfade 
#            if j == numXfades-1:
#                xCnt = numFinalXfade
#            else:
#                xCnt = xFrames
#            curInc = nextInc
#            
#            for t in range(xFrames):
#                if frameCnt < numFrames:    # process until end of total length
#                
#                    imgBpTsc2 = misc.imread(imgFileList2[frameCnt % numImg2])
#
#                    if newDimX > hop_sz:
#                        #newDimX -= 2*hop_sz
#                        newDimX -= hop_sz
#                    if newDimY > hop_sz:
#                        #newDimY -= 2*hop_sz
#                        newDimY -= hop_sz
#                        
#                    # scale image to new dimensions
#                    imgObjTemp1 = misc.imread(imgFileList1[frameCnt % numImg1])
#                    imgItr1 = self.odmkEyeRescale(imgObjTemp1, newDimX, newDimY)
#                    if rotateSubFrame1 == 1:
#                        
#                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
#                        if rotDirection1 == 1:
#                            ang = -ang
#                            
#                        imgItr1 = ndimage.rotate(imgItr1, ang, reshape=False)
#
#                    imgItr2 = self.odmkEyeRescale(imgBpTsc1, newDimX, newDimY)
#                    if rotateSubFrame2 == 1:
#                    
#                        ang = (atan2(zn[t % (xFrames-1)].imag, zn[t % (xFrames-1)].real))*180/np.pi
#                        if rotDirection2 == 1:
#                            ang = -ang
#                            
#                        imgItr2 = ndimage.rotate(imgItr2, ang, reshape=False)
#
#
#                    # region = (left, upper, right, lower)
#                    # subbox = (i + 1, i + 1, newDimX, newDimY)
#
#                    for j in range(SzY):
#                        for k in range(SzX):
#                            
#                            if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
#                                
#                                if (j+offsetY >= 0  and j+offsetY < SzY) and (k+offsetX >= 0  and k+offsetX < SzX):
#                                    #print('*****j = '+str(j)+'; k = '+str(k)+'; j+offsetY = '+str(j+offsetY)+'; k+offsetX = '+str(k+offsetX))
#                                    imgBpTsc1[j+offsetY, k+offsetX, :] = imgItr1[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]
#                                    imgBpTsc2[j+offsetY, k+offsetX, :] = imgItr2[int(j - (SzY-newDimY)/2), int(k - (SzX-newDimX)/2), :]                                
#
#                    imgBpTscOut = Image.blend(Image.fromarray(imgBpTsc1), Image.fromarray(imgBpTsc2), alphaX[t])
#                    imgBpTscOut = ImageOps.autocontrast(imgBpTscOut, cutoff=0)
#
#                    # update name counter & concat with base img name    
#                    nextInc += 1
#                    zr = ''
#    
#                    # inOrOut controls direction of parascope func
#                    if inOrOut == 1:
#                        for j in range(n_digits - len(str(nextInc))):
#                            zr += '0'
#                        strInc = zr+str(nextInc)
#                    else:
#                        for j in range(n_digits - len(str(curInc + xCnt))):
#                            zr += '0'
#                        strInc = zr+str(curInc + (xFrames - xCnt))
#                    imgFxSlothCultFull = imgOutDir+imgFxSlothCultNm+strInc+'.jpg'
#                    misc.imsave(imgFxSlothCultFull, imgBpTscOut)
#    
#                    # optional write to internal buffers
#                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
#                    # imgDualBipolarTelescArray.append(imgBpTsc)
#
#                    xCnt -= 1                    
#                    frameCnt += 1
#    
#        return
#        # return [imgBSlothRecursArray, imgBSlothRecursNmArray]




    # // *--------------------------------------------------------------* //
    # // *---::ODMKEYE - EYE LFO Horizontal trails f::---*
    # // *--------------------------------------------------------------* //
    
    def odmkLfoHtrails(self, imgArray, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None'):
        ''' LfoHtrails function '''
    
        if imgOutNm != 'None':
            imgLfoHtrailsNm = imgOutNm
        else:
            imgLfoHtrailsNm = 'imgLfoHtrails'
    
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

        
        alphaX = np.linspace(0.0, 1.0, xfadeFrames)
        imgRndIdxArray = randomIdxArray(numFrames, numImg)
   
        # imgDualBipolarTelescNmArray = []
        # imgDualBipolarTelescArray = []

        # generate an LFO sin control signal - 1 cycle per xfadeFrames
        T = 1.0 / framesPerSec    # sample period
        #LFO_T = 8 * xfadeFrames * T
        LFO_T = 8 * xfadeFrames * T
        
        
        # create composite sin source
        x = np.linspace(0.0, (xfadeFrames+1)*T, xfadeFrames+1)
        x = x[0:len(x)-1]
        LFOAMP = 3
        LFO = LFOAMP*np.sin(LFO_T * 2.0*np.pi * x) + 3
        

        #pdb.set_trace()

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
               
            imgClone1 = Image.fromarray(imgClone1)
            imgClone2 = Image.fromarray(imgClone2)               
               
            # initialize output 
#            imgBpTsc1 = imgClone1.copy()
#            imgBpTsc2 = imgClone2.copy()
#            imgBpTscOut = imgClone1.copy()

                              
            alphaBl = round(random.random())                
                             


            # functionalize then call for each telescope    
            for t in range(xfadeFrames):
                if frameCnt < numFrames:    # process until end of total length

                    print('LFO of t = '+str(int(LFO[t])))
                    nFrameTrail = int(LFO[t])
                    #nFrameTrail = 5
                    
                    imgCpy1 = imgClone1.copy()
                    imgCpy2 = imgClone2.copy()

                    if t > 0:
                        imgCpy1 = self.horizTrails(imgClone1, SzX, SzY, nFrameTrail, 0)
                        imgCpy2 = self.horizTrails(imgClone2, SzX, SzY, nFrameTrail, 0)

                    if alphaBl == 1:
                        imgCpy1 = Image.blend(imgCpy1, imgCpy2, alphaX[t])

                    # update name counter & concat with base img name    
                    nextInc += 1
                    zr = ''
    
                    for j in range(n_digits - len(str(nextInc))):
                        zr += '0'
                    strInc = zr+str(nextInc)
                    imgLfoHtrailsFull = imgOutDir+imgLfoHtrailsNm+strInc+'.jpg'
                    misc.imsave(imgLfoHtrailsFull, imgCpy1)
    
                    # optional write to internal buffers
                    # imgDualBipolarTelescNmArray.append(imgDualBipolarTelescFull)
                    # imgDualBipolarTelescArray.append(imgBpTsc)
                    
                    frameCnt += 1
    
        return
        # return [imgLfoParascopeArray, imgLfoParascopeNmArray]


    
    # /////////////////////////////////////////////////////////////////////////
    # #########################################################################
    # end : function definitions
    # #########################################################################
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# Notes:
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ***
# convert from Image image to Numpy array:
# arr = array(img)

# convert from Numpy array to Image image:
# img = Image.fromarray(array)





# #########################################################################
# Tools
# #########################################################################



## // *---------------------------------------------------------------------* //
## // *---Plotting---*
## // *---------------------------------------------------------------------* //
#
## define a sub-range for wave plot visibility
#tLen = 50
#
#fnum = 1
#pltTitle = 'Input Signal y1 (first '+str(tLen)+' samples)'
#pltXlabel = 'y1: '+str(testFreq1)+' Hz'
#pltYlabel = 'Magnitude'
#
#
#sig = y1[0:tLen]
## define a linear space from 0 to 1/2 Fs for x-axis:
#xaxis = np.linspace(0, tLen, tLen)
#
#
#odmkplt.odmkPlot1D(fnum, sig, xaxis, pltTitle, pltXlabel, pltYlabel)
#
#
#fnum = 2
#pltTitle = 'FFT Mag: y1_FFTscale '+str(testFreq1)+' Hz'
#pltXlabel = 'Frequency: 0 - '+str(fs / 2)+' Hz'
#pltYlabel = 'Magnitude (scaled by 2/N)'
#
## sig <= direct
#
## define a linear space from 0 to 1/2 Fs for x-axis:
#xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
#odmkplt.odmkPlot1D(fnum, y1_FFTscale, xfnyq, pltTitle, pltXlabel, pltYlabel)
#
#
#
## // *---------------------------------------------------------------------* //
## // *---Multi Plot - source signal array vs. FFT MAG out array---*
## // *---------------------------------------------------------------------* //
#
#fnum = 3
#pltTitle = 'Input Signals: sinArray (first '+str(tLen)+' samples)'
#pltXlabel = 'sinArray time-domain wav'
#pltYlabel = 'Magnitude'
#
## define a linear space from 0 to 1/2 Fs for x-axis:
#xaxis = np.linspace(0, tLen, tLen)
#
#odmkplt.odmkMultiPlot1D(fnum, sinArray, xaxis, pltTitle, pltXlabel, pltYlabel, colorMp='hsv')
#
#
#fnum = 4
#pltTitle = 'FFT Mag: yScaleArray multi-osc '
#pltXlabel = 'Frequency: 0 - '+str(fs / 2)+' Hz'
#pltYlabel = 'Magnitude (scaled by 2/N)'
#
## define a linear space from 0 to 1/2 Fs for x-axis:
#xfnyq = np.linspace(0.0, 1.0/(2.0*T), N/2)
#
#odmkplt.odmkMultiPlot1D(fnum, yScaleArray, xfnyq, pltTitle, pltXlabel, pltYlabel, colorMp='hsv')






