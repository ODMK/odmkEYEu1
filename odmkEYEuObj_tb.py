# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((odmkEYEuObj_tb.py))::__
#
# Python ODMK img processing research testbench
# testbench for odmkEYEuObj.py
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
# from PIL import Image
# from PIL import ImageOps
# from PIL import ImageEnhance

rootDir = 'C:/odmkDev/odmkCode/odmkPython/'
audioScrDir = 'C:/odmkDev/odmkCode/odmkPython/audio/wavsrc/'
audioOutDir = 'C:/odmkDev/odmkCode/odmkPython/audio/wavout/'

#sys.path.insert(0, 'C:/odmkDev/odmkCode/odmkPython/util')
sys.path.insert(0, rootDir+'util')
from odmkClear import *

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(1, rootDir+'eye')
import odmkEYEuObj as eye

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(2, rootDir+'DSP')
import odmkClocks as clks
import odmkSigGen1 as sigGen

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(3, rootDir+'audio')
import native_wav as wavio

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


def randomIdx(n, k):
    '''for an array of k elements, returns a list of random indexes
       of length n (n integers rangin from 0:k-1)'''
    randIdx = []
    for i in range(n):
        randIdx.append(round(random.random()*(k-1)))
    return randIdx


# def reverseIdx(imgArray):




print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMK EYE IMG-2-VIDEO Experiments::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')


xLength = 15
fs = 48000.0        # audio sample rate:
framesPerSec = 60.0 # video frames per second:
# ub313 bpm = 178 , 267
bpm = 120    #glamourGoat14 = 93
timeSig = 4         # time signature: 4 = 4/4; 3 = 3/4

eyeDir = rootDir+'eye/eyeSrc/'


#SzX = 360
#SzY = 240

# mstrSzX = 360
# mstrSzY = 240

#mstrSzX = 1280
#mstrSzY = 960

# ?D Standard video aspect ratio
SzX = 1280
SzY = 720

# SD Standard video aspect ratio
#mstrSzX = 640
#mstrSzY = 480

# Golden ratio frames:
#mstrSzX = 1076
#mstrSzY = 666

#mstrSzX = 1257
#mstrSzY = 777

#mstrSzX = 1436
#mstrSzY = 888

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Instantiate objects::---*')
print('// *--------------------------------------------------------------* //')        


cntrlLoadX = 1
cntrlScaleAll = 0
cntrlEYE = 24
postProcess = 0

eyeTb = eye.odmkEYEu(xLength, bpm, timeSig, SzX, SzY, framesPerSec=30.0, fs=48000, cntrlEYE=cntrlEYE, rootDir='None')
print('\nAn odmkEYEu object has been instanced with:')
print('xLength = '+str(xLength)+'; bpm = '+str(bpm)+' ; timeSig = '+str(timeSig)+' ; framesPerSec = '+str(framesPerSec)+' ; fs = '+str(fs))


eyeClks = clks.odmkClocks(xLength, fs, bpm, framesPerSec)
print('\nAn odmkEYEu object has been instanced with:')
print('xLength = '+str(xLength)+'; fs = '+str(fs)+'; bpm = '+str(bpm)+' ; framesPerSec = '+str(framesPerSec))


framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))


if cntrlLoadX == 1:
    # jpgSrcDir: set the directory path used for processing .jpg files
    #jpgSrcDir = eyeDir+'totwnEyesSrc/'
    #jpgSrcDir = eyeDir+'odmkGorgulan56Scale1280x960/'    # pre-scaled or scaled img list
    #jpgSrcDir = eyeDir+'odmkGorgulan56Scale360x240/'
    #pgSrcDir = eyeDir+'bananasloth/'
    #jpgSrcDir = eyeDir+'bananaslothScale360x240/'
    jpgSrcDir = eyeDir+'bananaslothScale1280x720/'
    #jpgSrcDir = eyeDir+'odmkGorgulanSrc/'
    
# cntrlScaleAll = 1    
# cntrlScaleAll - Image scale processing
# <0> Bypass  => No scaling
# <1> Scale

if cntrlScaleAll == 1:
    high = 1    # "head crop" cntrl - <0> => center crop, <1> crop from img top
    # *** Define Dir & File name for Scaling
    # scaleFileName: file name used for output processed .jpg (appended with 000X)
    scaleFileName = 'bananasloth'
    # scaleOutDirName: set the directory path used for output processed .jpg
    scaleOutDirName = 'bananaslothScale1280x720/'
    
    
if cntrlEYE != 0:
    # eyeOutFileName: file name used for output processed .jpg (appended with 000X)
    #eyeOutFileName = 'eyeEchoGorgulan1v'
    eyeOutFileName = 'bananaslothTelescope'
    # eyeOutDirName: set the directory path used for output processed .jpg
    #eyeOutDirName = 'eyeEchoGorgulan1vDir/'
    eyeOutDirName = 'bananaslothTelescopeDir/'

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

# <<<selects image post-processing function>>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 1  => Image Random Select Algorithm              ::odmkImgRndSel::
# postProcess = 0

if postProcess == 1:
    # repeatReverseAllImg:
    n = 2                         # number of repetitions (including 1st copy)
    srcDir = 'baliMaskYoukaiDir/'       # source image directory 
    eyeOutDirName = 'baliMaskYoukaiDirRpt/'    # output image directory 
    eyeOutFileName = 'baliMaskYoukai'         # output image name 
    
elif postProcess == 2:
    # repeatReverseAllImg - operates on directory
    n = 2                         # number of repetitions (including 1st copy)
    srcDir = 'baliMaskYoukaiDir/'       # source image directory
    eyeOutDirName = 'baliMaskYoukaiDirRpt/'    # output image directory 
    eyeOutFileName = 'baliMaskYoukai'         # output image name



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# Bypass <0> ; choose raw image dir <1> ; or pre-scaled images <2>
#cntrlLoadX = 2    #*** defined above


if cntrlLoadX == 1:    # process raw .jpg folder

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Import all source img (.jpg) in a directory::---*')
    print('// *--------------------------------------------------------------* //')

    # jpgSrcDir = eyeDir+'process/'
    # jpgSrcDir = eyeDir+'kei777/'
    # jpgSrcDir = eyeDir+'totwnEyesSrc/'    #*** defined above
    [imgSrcArray, imgSrcNmArray] = eyeTb.importAllJpg(jpgSrcDir)

    print('\nCreated python lists:')
    print('<<imgSrcArray>> (image data object list)')
    print('<<imgSrcNmArray>> (image data object names)\n')


print('// *--------------------------------------------------------------* //')

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

# Normalizes (Scales/Crops) all images in specified directory
# Images are normalized to master X,Y dimensions using comination of
# resizing and cropping in order to preserve as much of the original image
# as possible. High==1 crops from top of image to prevent headchops

# cntrlScaleAll = 0    # defined above

if cntrlScaleAll == 1:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Scale all imgages in img obj array::---*')
    print('// *--------------------------------------------------------------* //')

    # define output directory for scaled img
    # Dir path must end with backslash /
    processScaleDir = eyeDir+scaleOutDirName
    os.makedirs(processScaleDir, exist_ok=True)
    
    # individual file root name
    outNameRoot = scaleFileName    
    

    [processScaledArray, processScaledNmArray] = eyeTb.odmkScaleAll(imgSrcArray, SzX, SzY, w=1, high=0, outDir=processScaleDir, outName=outNameRoot)

    print('\nCreated python lists:')
    print('<<processScaledArray>> (img data objects) and <<processScaledNmArray>> (img names)\n')

    print('Saved Scaled images to the following location:')
    print(processScaleDir)
    print('\n')

else:    # Bypass
    pass

# // *********************************************************************** //

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : Loading & Formatting img and sound files
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : ODMK pixel banging method calls
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

# *****CHECKIT*****

if cntrlEYE == 1:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Random Select Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames:
    # randomly select an image from the source dir, hold for h frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgRndSelNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgRndSeldir = eyeDir+eyeOutDirName
    os.makedirs(imgRndSeldir, exist_ok=True)  # If Dir does not exist, makedir

    numFrames = xLength * framesPerSec

    #[imgRndSelOutA, imgRndSelNmA] = odmkImgRndSel(processScaledArray, numFrames, imgRndSeldir, imgOutNm='keiroz777')
    #[imgRndSelOutA, imgRndSelNmA] = odmkImgRndSel(processScaledArray, numFrames, imgRndSeldir, imgOutNm=imgRndSelNm)
    [imgRndSelOutA, imgRndSelNmA] = eye.odmkImgRndSel(imgSrcArray, numFrames, imgRndSeldir, imgOutNm=imgRndSelNm)

    print('\nodmkImgRndSel function output => Created python lists:')
    print('<<<imgRndSelOutA>>>   (ndimage objects)')
    print('<<<imgRndSelNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(imgRndSeldir)

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


elif cntrlEYE == 2:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Random BPM Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames:
    # randomly select an image from the source dir, hold for n frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgRndBpmNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgRndBpmdir = eyeDir+eyeOutDirName
    # If Dir does not exist, makedir:
    os.makedirs(imgRndBpmdir, exist_ok=True)

    # generate the downFrames sequence:
    eyeDFrames = eyeClks.clkDownFrames()

    # final video length in seconds
    numFrames = xLength * framesPerSec

    [imgRndBpmOutA, imgRndBpmNmA] =  eye.odmkImgRndBpm(imgSrcArray, numFrames, eyeDFrames, imgRndBpmdir, imgOutNm=imgRndBpmNm)

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

# ***GOOD -> FUNCTIONALIZEIT***

elif cntrlEYE == 3:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Random Select Telescope Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames: 
    # randomly select an image from the source dir, hold for h frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgTelescRndNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgTelescRnddir = eyeDir+eyeOutDirName
    # If Dir does not exist, makedir:
    os.makedirs( imgTelescRnddir, exist_ok=True)

    # For now, assume tlPeriod * tlItr matches final video length
    frameRate = 30
    tlItr = 7       # number of periods of telescoping period
    tlPeriod = 30   # number of frames for telescoping period
    numFrames = tlItr * tlPeriod
    # final video length in seconds = numFrames / 30
    # eyeLength = numFrames / frameRate

    # defined above!
    SzX = mstrSzX
    SzY = mstrSzY

    inOrOut = 0     # telescope direction: 0 = in, 1 = out
    hop_sz = 4

    imgTelescRndNmArray = []
    imgTelescRndArray = []

    #create an array of random index
    #use numFrames + tlItr to allow to increment starting image each tlPeriod
    rIdxArray = randomIdx(numFrames+tlItr, len(imgSrcArray))

    imgCount = numFrames
    n_digits = int(ceil(np.log10(imgCount))) + 2
    nextInc = 0
    for i in range(tlItr):
        newDimX = SzX
        newDimY = SzY
        imgClone = imgSrcArray[rIdxArray[i]]
        for t in range(tlPeriod):
            # select random image telescope in for select period
            if newDimX > 2:
                newDimX -= hop_sz
            if newDimY > 2:
                newDimY -= hop_sz
            # scale image to new dimensions
            imgItr = odmkEyeRescale(imgClone, newDimX, newDimY)
            # region = (left, upper, right, lower)
            # subbox = (i + 1, i + 1, newDimX, newDimY)
            for j in range(SzY):
                for k in range(SzX):
                    if ((j > t+hop_sz) and (j < newDimY) and (k > t+hop_sz) and (k < newDimX)):
                        imgClone[j, k, :] = imgItr[j - t, k - t, :]
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
            imgNormalizeNm = imgTelescRndNm+strInc+'.jpg'
            imgRndSelFull = imgTelescRnddir+imgNormalizeNm
            misc.imsave(imgRndSelFull, imgClone)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# *-----BYPASS BEGIN-----*

# *****CHECKIT*****

elif cntrlEYE == 4:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Output a sequence of rotates images::---*')
    print('// *-----360 deg rotation period = numFrames-----*')
    print('// *--------------------------------------------------------------* //')

    # num_gz = 192
    numFrames = 111
    # zn = cyclicZn(numFrames) -- called in func
    
    rotD = 0    # rotation direction: 0=counterclockwise ; 1=clockwise

    imgSrc = processScaledArray[9]

    outDir = eyeDir+'imgRotateOut/'
    os.makedirs(outDir, exist_ok=True)
    outNm = 'imgRotate'

    [imgRotArray, imgRotNmArray] = eye.odmkImgRotateSeq(imgSrc, numFrames, outDir, imgOutNm=outNm, rotDirection=rotD)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

# // *---------------------------------------------------------------------* //
# // *--Image Rotate & alternate sequence--*
# // *---------------------------------------------------------------------* //

# *****FUNCTIONALIZEIT*****

elif cntrlEYE == 5:

    # dir where processed img files are stored:
    gzdir = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/exp1/'

    eye_name = 'odmkRotFlash1'

    # num_gz = 192
    num_gz = 256
    zn = cyclicZn(num_gz)

    gz1NmArray = []
    gz1Array = []

    nextInc = 0
    for i in range(num_gz):
        zr = ''
        ang = (atan2(zn[i].imag, zn[i].real))*180/np.pi
        if ((i % 4) == 1 or (i % 4) == 2):
            rotate_gz1 = ndimage.rotate(gz1, ang, reshape=False)
        else:
            rotate_gz1 = ndimage.rotate(gzRESIZEx02, ang, reshape=False)
        nextInc += 1
        # Find num digits required to represent max index
        n_digits = int(ceil(np.log10(num_gz))) + 2
        for j in range(n_digits - len(str(nextInc))):
            zr += '0'
        strInc = zr+str(nextInc)
        gz1Nm = gzdir+eye_name+strInc+'.jpg'
        misc.imsave(gz1Nm, rotate_gz1)
        gz1NmArray.append(gz1Nm)
        gz1Array.append(rotate_gz1)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

if cntrlEYE == 6:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Image CrossFade Sequencer::---*')
    print('// *---Crossfade a sequence of images, period = framesPerBeat---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outXfadeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outXfadeDir = eyeDir+eyeOutDirName
    os.makedirs(outXfadeDir, exist_ok=True)

    xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 6)) #2 images for 2 frames
    sLength = xLength
    
    [xfadeOutA, xfadeOutNmA] = eye.odmkImgXfade(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

    print('\nodmkImgDivXfade function output => Created python lists:')
    print('<<<xfadeOutA>>>   (ndimage objects)')
    print('<<<xfadeOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(outXfadeDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# *****CHECKIT*****
if cntrlEYE == 666:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Image CrossFade Sequencer::---*')
    print('// *---Crossfade a sequence of images, period = framesPerBeat---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outXfadeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outXfadeDir = eyeDir+eyeOutDirName
    os.makedirs(outXfadeDir, exist_ok=True)

    xfadeFrames = 2*int(np.ceil(eyeClks.framesPerBeat))

    [xfadeOutA, xfadeOutNmA] = eye.odmkImgXfadeAll(imgSrcArray, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

    print('\nodmkImgDivXfade function output => Created python lists:')
    print('<<<xfadeOutA>>>   (ndimage objects)')
    print('<<<xfadeOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(outXfadeDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

if cntrlEYE == 67:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Image Solarize CrossFade Sequencer::---*')
    print('// *---Crossfade a sequence of images, period = framesPerBeat---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outSolXfadeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outSolXfadeDir = eyeDir+eyeOutDirName
    os.makedirs(outSolXfadeDir, exist_ok=True)

    framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))

    [xSolOutA, xSolOutNmA] = eye.odmkImgSolXfade(imgSrcArray, framesPerBeat, outSolXfadeDir, imgOutNm=outSolXfadeNm)

    #[xfadeRepeatA, xfadeRepeatNmA] = repeatDir(outXfadeDir, 3, w=1, repeatDir=eyeDir+'eyeXfadeRepeat/', repeatName='eyeXfadeR')

    print('\nodmkImgSolXfade function output => Created python lists:')
    print('<<<xSolOutA>>>   (ndimage objects)')
    print('<<<xSolOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(eyeDir+'outSolXfadeDir')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

if cntrlEYE == 7:
# *****CURRENT=GOOD / CHECK ALT PARAMS, move repeat to post-proc*****

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Divide CrossFade Sequencer::---*')
    print('// *---Divide Crossfade a sequence of images, period = framesPerBeat-*')
    print('// *--------------------------------------------------------------* //')

    srcDivXfadeDir = eyeDir+'eyeSrcExp23/'
    outDivXfadeDir = eyeDir+'eyeDivXfadeOut/'
    os.makedirs(outDivXfadeDir, exist_ok=True)

    [imgSrcObj, imgSrcObjNm] = eye.importAllJpg(srcDivXfadeDir)

    framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))

    [divXfadeOutA, divXfadeOutNmA] = eye.odmkImgDivXfade(imgSrcArray, framesPerBeat, outDivXfadeDir, imgOutNm='eyeDivXfade')

    #[divXfadeRepeatA, divXfadeRepeatNmA] = repeatDir(outDivXfadeDir, 3, w=1, repeatDir=eyeDir+'eyeDivXfadeRepeat/', repeatName='eyeDivXfadeR')

    print('\nodmkImgDivXfade function output => Created python lists:')
    print('<<<divXfadeOutA>>>   (ndimage objects)')
    print('<<<divXfadeOutNmA>>> (img names)')

#    print('\nrepeatDir function output => Created python lists:')
#    print('<<<divXfadeRepeatA>>>   (ndimage objects)')
#    print('<<<divXfadeRepeatNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(eyeDir+'eyeDivXfadeRepeat')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# // *--------------------------------------------------------------* //
# // *---::ODMKEYE - Image CrossFade Rotate Sequencer::---*')
# // *--------------------------------------------------------------* //


if cntrlEYE == 8:
# *****CURRENT - CHECKIT*****

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Image CrossFade Rotate::---*')
    print('// *-----Crossfade & Rotate a sequence of images---*')
    print('// *-----sequence period = framesPerBeat---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outXfadeRotNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outXfadeRotDir = eyeDir+eyeOutDirName
    os.makedirs(outXfadeRotDir, exist_ok=True)


    # [imgSrcObj, imgSrcObjNm] = importAllJpg(srcXfadeRotDir)

    rotD = 0    # rotation direction: 0=counterclockwise ; 1=clockwise
    framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))
    #xFrames = framesPerBeat

    xFrames = int(ceil(xLength * framesPerSec) / 2)   # 1 fade whole length

    [xfadeRotOutA, xfadeRotOutNmA] = eye.odmkImgXfadeRot(imgSrcArray, xLength, framesPerSec, xFrames, outXfadeRotDir, imgOutNm=outXfadeRotNm, rotDirection=rotD)

    #[xfadeRotRepeatA, xfadeRotRepeatNmA] = repeatDir(outXfadeRotDir, 3, w=1, repeatDir=eyeDir+'eyeXfadeRotRepeat/', repeatName='eyeXfadeRotR')

    print('\nodmkImgXfadeRot function output => Created python lists:')
    print('<<<xfadeRotOutA>>>   (ndimage objects)')
    print('<<<xfadeRotOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(outXfadeRotDir)

    #print('\nrepeatDir function output => Created python lists:')
    #print('<<<xfadeRotRepeatA>>>   (ndimage objects)')
    #print('<<<xfadeRotRepeatNmA>>> (img names)')

    #print('\noutput processed img to the following directory:')
    #print(eyeDir+'eyeXfadeRotRepeat')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

## *-----BYPASS BEGIN-----*
if cntrlEYE == 9:
## *****CHECKIT*****

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image telescope Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # iterate zoom out & overlay, zoom in & overlay for n frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgTelescNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgTelescOutDir = eyeDir+eyeOutDirName
    # If Dir does not exist, makedir:
    os.makedirs( imgTelescOutDir, exist_ok=True)


    # num frames per beat (quantized)
    fpb = int(np.ceil(eyeClks.framesPerBeat))
    fps = framesPerSec

    inOrOut = 0     # telescope direction: 0 = in, 1 = out
        

    eye.odmkImgTelescope(imgSrcArray, xLength, fps, fpb, imgTelescOutDir, imgOutNm=imgTelescNm, inOut=inOrOut)

    print('Saved Processed images to the following location:')
    print(imgTelescOutDir)
    print('\n')

    print('// *--------------------------------------------------------------* //')    


#def odmkImgTelescope(imgList, framesPerBeat, inOut=0, imgOutDir, imgOutNm='None'):
#    ''' outputs a sequence of telescoping images (zoom in or zoom out)
#        The period of telescoping sequence synched to framesPerBeat
#        assumes images in imgList are normalized (scaled/cropped) numpy arrays '''
#
#    if imgOutNm != 'None':
#        imgTelescNm = imgOutNm
#    else:
#        imgTelescNm = 'imgTelescope'
#        
#    # find number of source images in src dir
#    numSrcImg = len(imgList)
#    # initialize SzX, SzY to dimensions of src img
#    SzX = imgList[0].shape[1]
#    SzY = imgList[0].shape[0]
#
#    numFrames = numSrcImg * framesPerBeat                
#
#    hop_sz = ceil(np.log(SzX/framesPerBeat))  # number of pixels to scale img each iteration
#    
#    #imgTelescopeNmArray = []
#    #imgTelescopeArray = []
#    
#    imgCount = numFrames    # counter to increment output name
#    n_digits = int(ceil(np.log10(imgCount))) + 2
#    nextInc = 0
#    for i in range(numSrcImg):
#        newDimX = SzX
#        newDimY = SzY
#        imgClone = imgList[i]    # mod to rotate through src img
#        for t in range(fpb):
#            if newDimX > 2:
#                newDimX -= 2*hop_sz
#            if newDimY > 2:
#                newDimY -= 2*hop_sz
#            # scale image to new dimensions
#            imgItr = odmkEyeRescale(imgClone, newDimX, newDimY)
#            # region = (left, upper, right, lower)
#            # subbox = (i + 1, i + 1, newDimX, newDimY)
#            for j in range(SzY):
#                for k in range(SzX):
#                    #if ((j >= (t+1)*hop_sz) and (j < (newDimY+(SzY-newDimY)/2)) and (k >= (t+1)*hop_sz) and (k < (newDimX+(SzX-newDimX)/2))):
#                    if ((j >= (t+1)*hop_sz) and (j < newDimY+((t+1)*hop_sz)/2) and (k >= (t+1)*hop_sz) and (k < newDimX+((t+1)*hop_sz)/2)):
#                        #imgClone[j+(SzY-newDimY)/2, k+(SzX-newDimX)/2, :] = imgItr[j - t, k - t, :]
#                        imgClone[j, k, :] = imgItr[j - (SzY-newDimY)/2, k - (SzX-newDimX)/2, :]
#            nextInc += 1
#            zr = ''
#            if inOrOut == 1:
#                for j in range(n_digits - len(str(nextInc))):
#                    zr += '0'
#                strInc = zr+str(nextInc)
#            else:
#                for j in range(n_digits - len(str(imgCount - (nextInc)))):
#                    zr += '0'
#                strInc = zr+str(imgCount - (nextInc))
#            imgTelescFull = imgOutDir+imgTelescNm+strInc+'.jpg'
#            misc.imsave(imgTelescFull, imgClone)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

if cntrlEYE == 10:
# *****check v results*****

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Xfade telescope Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # function: odmkImgDivXfadeTelescope
    # iterate zoom out & overlay, zoom in & overlay for n frames

    # loaded images into the following arrays (defined above)

    # jpgSrcDir = eyeDir+'gorgulanScale/'    # pre-scaled img list
    # [processObjList, processSrcList] = importAllJpg(jpgSrcDir)

    # num frames per beat (quantized)
    fpb = 2*int(np.ceil(eyeClks.framesPerBeat))

    inOrOut = 1     # telescope direction: 0 = in, 1 = out

    #eyeOutFileName = 'myOutputFileName'    # defined above
    #imgXfadeTelNm = eyeOutFileName
    # dir where processed img files are stored:
    #eyeOutDirName = 'myOutputDirName'    # defined above
    imgXfadeTelOutDir = eyeDir+eyeOutDirName
    os.makedirs(imgXfadeTelOutDir, exist_ok=True)
    if inOrOut == 0:
        imgXfadeTelNm = eyeOutFileName+'_telIn'
    else:
        imgXfadeTelNm = eyeOutFileName+'_telOut'

    eye.odmkImgXfadeTelescope(processScaledArray, fpb, imgXfadeTelOutDir, inOut=inOrOut, imgOutNm=imgXfadeTelNm)

    print('Saved Processed images to the following location:')
    print(imgXfadeTelOutDir)
    print('\n')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

# *-----BYPASS BEGIN-----*
if cntrlEYE == 11:
# *****check v results*****

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Div Xfade telescope Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # function: odmkImgDivXfadeTelescope
    # iterate zoom out & overlay, zoom in & overlay for n frames

    # loaded images into the following arrays (defined above)

    # jpgSrcDir = eyeDir+'gorgulanScale/'    # pre-scaled img list
    # [processObjList, processSrcList] = importAllJpg(jpgSrcDir)

    # num frames per beat (quantized)
    fpb = 2*int(np.ceil(eyeClks.framesPerBeat))

    inOrOut = 1     # telescope direction: 0 = in, 1 = out

    # dir where processed img files are stored:
    imgDivXfadeTelOutDir = eyeDir+'odmkDivXfadeTelOut/'
    os.makedirs(imgDivXfadeTelOutDir, exist_ok=True)
    if inOrOut == 0:
        imgDivXfadeTelNm = 'divXfadeTelIn'
    else:
        imgDivXfadeTelNm = 'divXfadeTelOut'

    eye.odmkImgDivXfadeTelescope(processScaledArray, fpb, imgDivXfadeTelOutDir, inOut=inOrOut, imgOutNm=imgDivXfadeTelNm)

    print('Saved Processed images to the following location:')
    print(imgDivXfadeTelOutDir)
    print('\n')
    
    print('// *--------------------------------------------------------------* //')    

# *-----BYPASS END-----*

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

elif cntrlEYE == 12:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - odmkEyeEchoBpm Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames:
    # randomly select an image from the source dir, hold for n frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgRndBpmNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgRndBpmdir = eyeDir+eyeOutDirName
    # If Dir does not exist, makedir:
    os.makedirs(imgRndBpmdir, exist_ok=True)

    # generate the downFrames sequence:
    eyeDFrames = eyeClks.clkDownFrames()
    
    numECHOES = 8
    stepECHOES = 3
    decayECHOES = 'None'

    # final video length in seconds
    numFrames = int(np.ceil(xLength * framesPerSec))

    # [eyeECHO_Bpm_Out, eyeECHO_Bpm_Nm] =  odmkEyeEchoBpm(imgSrcArray, numFrames, eyeDFrames, numECHOES, stepECHOES, imgRndBpmdir, echoDecay = decayECHOES, imgOutNm=imgRndBpmNm)
    eye.odmkEyeEchoBpm(imgSrcArray, numFrames, eyeDFrames, numECHOES, stepECHOES, imgRndBpmdir, echoDecay = decayECHOES, imgOutNm=imgRndBpmNm)
    print('// *--------------------------------------------------------------* //')

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


if cntrlEYE == 20:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Image Color Enhance::---*')
    print('// *---Color Enhance a sequence of images, period = framesPerBeat---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outXfadeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outXfadeDir = eyeDir+eyeOutDirName
    os.makedirs(outXfadeDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames    
    sLength = xLength
    
    [xCEnOutA, xCEnOutNmA] = eye.odmkImgColorEnhance(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

    print('\nodmkImgColorEnhance function output => Created python lists:')
    print('<<<xCEnOutA>>>   (ndimage objects)')
    print('<<<xCEnOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(outXfadeDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


if cntrlEYE == 21:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMK Image Color Enhance::---*')
    print('// *---Color Enhance a sequence of images, period = framesPerBeat---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outXfadeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outXfadeDir = eyeDir+eyeOutDirName
    os.makedirs(outXfadeDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = xLength
    sLength = xLength

    inOrOut = 0     # telescope direction: 0 = in, 1 = out
    
    eye.odmkImgCEnTelescope(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm, inOut=inOrOut)

    print('\nodmkImgCEnTelescope function output => Created python lists:')
    print('<<<xCEnTelescOutA>>>   (ndimage objects)')
    print('<<<xCEnTelescOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(outXfadeDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')


if cntrlEYE == 22:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Pixel Random Replace::---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outXfadeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outXfadeDir = eyeDir+eyeOutDirName
    os.makedirs(outXfadeDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = xLength
    sLength = xLength

    inOrOut = 0     # telescope direction: 0 = in, 1 = out
    
    eye.odmkImgCEnTelescope(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm, inOut=inOrOut)

    print('\nodmkImgCEnTelescope function output => Created python lists:')
    print('<<<xCEnTelescOutA>>>   (ndimage objects)')
    print('<<<xCEnTelescOutNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(outXfadeDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')
    

if cntrlEYE == 23:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Dual Bipolar Oscillating telescope f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    bipolarTelescNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    bipolarTelescDir = eyeDir+eyeOutDirName
    os.makedirs(bipolarTelescDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    inOrOut = 0     # telescope direction: 0 = in, 1 = out
    
    eyeTb.odmkDualBipolarTelesc(imgSrcArray, sLength, framesPerSec, xfadeFrames, bipolarTelescDir, imgOutNm=bipolarTelescNm, inOrOut=inOrOut)

    print('\nProcessed images output to the following directory:')
    print(bipolarTelescDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')

if cntrlEYE == 24:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Bananasloth Recursive f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    bSlothRecursNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    bSlothRecursDir = eyeDir+eyeOutDirName
    os.makedirs(bSlothRecursDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    inOrOut = 0     # telescope direction: 0 = in, 1 = out

    eyeTb.odmkBSlothRecurs(imgSrcArray, sLength, framesPerSec, xfadeFrames, bipolarTelescDir, imgOutNm=bipolarTelescNm, inOrOut=inOrOut)

    print('\nProcessed images output to the following directory:')
    print(bSlothRecursDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : ODMK pixel banging
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : img post-processing
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


if postProcess == 1:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Repeat All images in SRC Directory::---*')
    print('// *--------------------------------------------------------------* //')
    # repeatAllImg - operates on directory
    # renames and repeats all images in the srcDir in alternating reverse order
    # n=1 -> reversed, n=2 -> repeat, n=3 -> reversed, ...

    # srcDir defined above - source image directory
    repeatSrcDir = eyeDir+srcDir

    # eyeOutFileName = 'myOutputFileName'    # defined above
    repeatNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    repeatDir = eyeDir+eyeOutDirName
    # If Dir does not exist, makedir:
    os.makedirs( repeatRevDir, exist_ok=True)

    eye.repeatAllImg(srcDir, n, w=1, repeatDir=repeatDir, repeatName=repeatNm)

elif postProcess == 2:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Repeat Reverse All images in SRC Directory::---*')
    print('// *--------------------------------------------------------------* //')
    # repeatReverseAllImg - operates on directory
    # renames and repeats all images in the srcDir in alternating reverse order
    # n=1 -> reversed, n=2 -> repeat, n=3 -> reversed, ...

    # srcDir defined above - source image directory
    repeatSrcDir = eyeDir+srcDir

    # eyeOutFileName = 'myOutputFileName'    # defined above
    repeatRevNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    repeatRevDir = eyeDir+eyeOutDirName
    # If Dir does not exist, makedir:
    os.makedirs( repeatRevDir, exist_ok=True)


    eye.repeatReverseAllImg(repeatSrcDir, n, w=1, repeatReverseDir=repeatRevDir, repeatReverseName=repeatRevNm)



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

# ***GOOD***

# *-----BYPASS BEGIN-----*

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::Concatenate all images in directory list::---*')
#print('// *--------------------------------------------------------------* //')
#
#
#loc1 = eyeDir+'uCache/imgTelescopeIn1_1436x888_56x/'
#loc2 = eyeDir+'uCache/imgTelescopeIn2_1436x888_56x/'
#loc3 = eyeDir+'uCache/imgTelescopeIn3_1436x888_56x/'
#loc4 = eyeDir+'uCache/imgTelescopeIn4_1436x888_56x/'
#
#dirList = [loc1, loc2, loc3, loc4]
#
#gorgulanConcatDir = eyeDir+'gorgulanConcatExp/'
#os.makedirs(concatDir, exist_ok=True)
#reName = 'gorgulanConcat'
#
## function:  concatAllDir(dirList, w=0, concatDir='None', concatName='None')
## [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList)
## [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=0, concatName=reName)
## [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=0, concatDir=gorgulanConcatDir)
## [gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=1, concatDir=gorgulanConcatDir)
#[gorgulanObjList, gorgulanSrcList] = concatAllDir(dirList, w=1, concatDir=gorgulanConcatDir, concatName=reName)
#
#print('\nFound '+str(len(gorgulanSrcList))+' images in the directory list\n')
#print('\nCreated python lists of ndimage objects:')
#print('<<gorgulanObjList>> (img data objects) and <<gorgulanSrcList>> (img names)\n')
#
#print('// *--------------------------------------------------------------* //')

# *-----BYPASS END-----*

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# *-----BYPASS BEGIN-----*

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::repeat all images in directory::---*')
#print('// *--------------------------------------------------------------* //')

#srcLenswarpDir = eyeDir+'odmkIdolx3/'
#outLenswarpDir = eyeDir+'eyeLenswarpOut/'
#os.makedirs(outLenswarpDir, exist_ok=True)
#
#[imgSrcObj, imgSrcObjNm] = importAllJpg(srcLenswarpDir)
#
#[xLenswarpRepeatA, xLenswarpRepeatNmA] = repeatDir(srcLenswarpDir, 3, w=1, repeatDir=outLenswarpDir, repeatName='eyeLenswarpR')


#print('// *--------------------------------------------------------------* //')

# *-----BYPASS END-----*

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : img post-processing
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# Notes:
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


## image division blend algorithm..
#c = a/((b.astype('float')+1)/256)
## saturating function - if c[m,n] > 255, set to 255:
#d = c*(c < 255)+255*np.ones(np.shape(c))*(c > 255)
#
#eyeDivMixFull = outXfadeDir+'eyeDivMix.jpg'
#misc.imsave(eyeDivMixFull, d)

# ***End: division blend ALT algorithm***


# ***
# convert from Image image to Numpy array:
# arr = array(img)

# convert from Numpy array to Image image:
# img = Image.fromarray(array)


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# Experimental:
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# *** Add sound clip synched to each pixelbang itr
# *** add complete soundtrack synched to frame-rate

# create util func to reverse numbering of img

# *** randomly select pixelbang algorithm for each itr
# ***define 'channels' interleaved pixelbang algorithms of varying duration


# randome pixel replacement from img1 to img2
# -replace x% of pixels randomly each frame, replaced pixels pulled out of pool
# -for next random selection to choose from until all pixels replaced

# 2D wave eqn modulation
# modulate colors/light or modulate between two images ( > 50% new image pixel)

# 2D wave modulation iterating zoom

# trails, zoom overlay interpolation...

# fractal pixel replace - mandelbrot set algorithm

# random pixel replace through rotating image set - image is rotated by
# randomly replacing pixels while skipping through bounded sliding window of
# incrementally rotated windows

# sliding probability window -> for vector of length n, randomly choose values
# bounded by a sliding window (goal is to evolve through a list of img)

# // *---------------------------------------------------------------------* //