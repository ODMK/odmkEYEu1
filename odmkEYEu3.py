# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((odmkEYEu3.py))::__
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
sys.path.insert(1, rootDir+'audio')
import native_wav as wavio


#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(2, rootDir+'DSP')
import odmkClocks as clks
import odmkSigGen1 as sigGen

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

def importAllJpg(srcDir):
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
    # print('// *--------------------------------------------------------------* //')

    return [imgObjList, imgSrcList]

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


# // *--------------------------------------------------------------* //
# // *---::ODMKEYE - EYE image array processing prototype::---*
# // *--------------------------------------------------------------* //

def odmkEYEprototype(imgArray, sLength, framesPerSec, FrameCtrl, imgOutDir, imgOutNm='None', usrCtrl=0):
    ''' basic prototype function '''

    if imgOutNm != 'None':
        imgPrototypeNm = imgOutNm
    else:
        imgPrototypeNm = 'imgXfadeRot'

    SzX = imgArray[0].shape[1]
    SzY = imgArray[0].shape[0]

    numFrames = int(ceil(sLength * framesPerSec))
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



# -----------------------------------------------------------------------------
# img Pre-processing func
# -----------------------------------------------------------------------------

def odmkEyePrint(img, title, idx):
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    figx = plt.figure(num=idx, facecolor='silver', edgecolor='k')
    plt.imshow(img)
    plt.xlabel('Src Width = '+str(imgWidth))
    plt.ylabel('Src Height = '+str(imgHeight))
    plt.title(title)


def odmkEyeRescale(img, SzX, SzY):
    # Rescale Image to SzW width and SzY height:
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    imgRescale_r = misc.imresize(img_r, [SzY, SzX], interp='cubic')
    imgRescale_g = misc.imresize(img_g, [SzY, SzX], interp='cubic')
    imgRescale_b = misc.imresize(img_b, [SzY, SzX], interp='cubic')
    imgRescale = np.dstack((imgRescale_r, imgRescale_g, imgRescale_b))
    return imgRescale


def odmkEyeZoom(img, Z):
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


def odmkEyeCrop(img, SzX, SzY, high):
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
                imgCrop_r[j, k] = img_r[j, k + wdiff_lhs]
                imgCrop_g[j, k] = img_g[j, k + wdiff_lhs]
                imgCrop_b[j, k] = img_b[j, k + wdiff_lhs]
            else:
                # pdb.set_trace()
                imgCrop_r[j, k] = img_r[j + hdiff_ths, k + wdiff_lhs]
                imgCrop_g[j, k] = img_g[j + hdiff_ths, k + wdiff_lhs]
                imgCrop_b[j, k] = img_b[j + hdiff_ths, k + wdiff_lhs]
        # if j == SzY - 1:
            # print('imgItr ++')
    imgCrop = np.dstack((imgCrop_r, imgCrop_g, imgCrop_b))
    return imgCrop


def odmkEyeDim(img, SzX, SzY, high):
    ''' Zoom and Crop image to match source dimensions '''

    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    #pdb.set_trace()

    if (imgWidth == SzX and imgHeight == SzY):
        # no processing
        imgScaled = img

    elif (imgWidth == SzX and imgHeight > SzY):
        # cropping only
        imgScaled = odmkEyeCrop(img, SzX, SzY, high)

    elif (imgWidth > SzX and imgHeight == SzY):
        # cropping only
        imgScaled = odmkEyeCrop(img, SzX, SzY, high)

    elif (imgWidth > SzX and imgHeight > SzY):
        # downscaling and cropping
        wRatio = SzX / imgWidth
        hRatio = SzY / imgHeight
        if wRatio >= hRatio:
            zoomFactor = wRatio
        else:
            zoomFactor = hRatio
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth < SzX and imgHeight == SzY):
        # upscaling and cropping
        wdiff = SzX - imgWidth
        zoomFactor = (imgWidth + wdiff) / imgWidth
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth == SzX and imgHeight < SzY):
        # upscaling and cropping
        hdiff = SzY - imgHeight
        zoomFactor = (imgHeight + hdiff) / imgHeight
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth < SzX and imgHeight < SzY):
        # upscaling and cropping (same aspect ratio -> upscaling only)
        wRatio = mstrSzX / imgWidth
        hRatio = mstrSzY / imgHeight
        if wRatio >= hRatio:
            zoomFactor = wRatio
        else:
            zoomFactor = hRatio
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth > SzX and imgHeight < SzY):
        # upscaling and cropping
        # wdiff = imgWidth - SzX
        hdiff = SzY - imgHeight
        zoomFactor = (imgHeight + hdiff) / imgHeight
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY, high)

    elif (imgWidth < SzX and imgHeight > SzY):
        # upscaling and cropping
        wdiff = SzX - imgWidth
        # hdiff = imgHeight - SzY
        zoomFactor = (imgWidth + wdiff) / imgWidth
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY, high)

    return imgScaled


def odmkScaleAll(srcObjList, SzX, SzY, w=0, high=0, outDir='None', outName='None'):
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
        imgScaled = odmkEyeDim(srcObjList[k], SzX, SzY, high)
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

def repeatAllImg(srcDir, n, w=0, repeatDir='None', repeatName='None'):
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

    [imgObjList, imgSrcList] = importAllJpg(srcDir)

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

def repeatReverseAllImg(srcDir, n, w=0, repeatReverseDir='None', repeatReverseName='None'):
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

    [imgObjList, imgSrcList] = importAllJpg(srcDir)

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


def concatAllDir(dirList, w=0, concatDir='None', concatName='None'):
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
        [imgObjList, imgSrcList] = importAllJpg(dirList[j])
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


# // *--------------------------------------------------------------* //
# // *---::ODMKEYE - Image Random Select Algorithm::---*
# // *--------------------------------------------------------------* //

def odmkImgRndSel(imgList, numFrames, imgOutDir, imgOutNm='None'):

    if imgOutNm != 'None':
        imgRndSelNm = imgOutNm
    else:
        imgRndSelNm = 'imgRndSelOut'

    imgRndSelNmArray = []
    imgRndSelArray = []
    
    rIdxArray = randomIdx(numFrames, len(imgList))
    
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

def odmkImgRndBpm(imgList, numFrames, frameHold, imgOutDir, imgOutNm='None'):

    if imgOutNm != 'None':
        imgRndSelNm = imgOutNm
    else:
        imgRndSelNm = 'imgRndSelOut'

    imgRndSelNmArray = []
    imgRndSelArray = []
    
    rIdxArray = randomIdx(numFrames, len(imgList))
    
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

def odmkImgRotateSeq(imgSrc, numFrames, imgOutDir, imgOutNm='None', rotDirection=0):
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
    
def odmkImgXfade(imgList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None'):
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
    
def odmkImgXfadeAll(imgList, xfadeFrames, imgOutDir, imgOutNm='None'):
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
    
def odmkImgSolXfade(imgList, numFrames, imgOutDir, imgOutNm='None'):
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

def odmkImgDivXfade(imgList, numFrames, imgOutDir, imgOutNm='None'):
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

def odmkImgXfadeRot(imgList, sLength, framesPerSec, xFrames, imgOutDir, imgOutNm='None', rotDirection=0):
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
    
    
    
    
    
## // *--------------------------------------------------------------* //
## // *---::ODMKEYE - Image CrossFade Rotate Sequence Algorithm::---*
## // *--------------------------------------------------------------* //
#
#def odmkImgXfadeRot(imgList, numFrames, imgOutDir, imgOutNm='None', rotDirection=0):
#    ''' outputs a sequence of images fading from img1 -> img2
#        Linearly alpha-blends images using PIL Image.blend
#        rotation direction: 0=counterclockwise ; 1=clockwise
#        assume images in imgList are numpy arrays '''
#
#    if imgOutNm != 'None':
#        imgXfadeRotNm = imgOutNm
#    else:
#        imgXfadeRotNm = 'imgXfadeRot'
#
#    zn = cyclicZn(numFrames-1)    # less one, then repeat zn[0] for full 360
#
#    numImg = len(imgList)
#    imgCount = numFrames * (numImg - 1)
#    n_digits = int(ceil(np.log10(imgCount))) + 2
#    nextInc = 0
#    alphaX = np.linspace(0.0, 1.0, numFrames)
#
#    imgXfadeRotNmArray = []
#    imgXfadeRotArray = []
#
#    for j in range(numImg - 1):
#        imgPIL1 = Image.fromarray(imgList[j])
#        imgPIL2 = Image.fromarray(imgList[j + 1])
#        for i in range(numFrames):
#            alphaB = Image.blend(imgPIL1, imgPIL2, alphaX[i])
#            ang = (atan2(zn[i % (numFrames-1)].imag, zn[i % (numFrames-1)].real))*180/np.pi
#            if rotDirection == 1:
#                ang = -ang
#            rotate_alphaB = ndimage.rotate(alphaB, ang, reshape=False)
#
#            nextInc += 1
#            zr = ''
#            for j in range(n_digits - len(str(nextInc))):
#                zr += '0'
#            strInc = zr+str(nextInc)
#            imgXfadeRotFull = imgOutDir+imgXfadeRotNm+strInc+'.jpg'
#            misc.imsave(imgXfadeRotFull, rotate_alphaB)
#            imgXfadeRotNmArray.append(imgXfadeRotFull)
#            imgXfadeRotArray.append(alphaB)
#
#    return [imgXfadeRotArray, imgXfadeRotNmArray]    

# // *--------------------------------------------------------------* //
# // *---::ODMKEYE - Image Telescope Sequence Algorithm::---*
# // *--------------------------------------------------------------* //

def odmkImgTelescope(imgList, xLength, framesPerSec, framesPerBeat, imgOutDir, imgOutNm='None', inOut=0):
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
            imgItr = odmkEyeRescale(imgClone, newDimX, newDimY)
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


# // *--------------------------------------------------------------* //
# // *---::ODMKEYE - Img Xfade Telescope Sequence Algorithm::---*
# // *--------------------------------------------------------------* //

def odmkImgXfadeTelescope(imgList, framesPerBeat, imgOutDir, inOut=0, imgOutNm='None'):
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
            imgItr1 = Image.fromarray(odmkEyeRescale(imgClone1, newDimX, newDimY))
            imgItr2 = Image.fromarray(odmkEyeRescale(imgClone2, newDimX, newDimY))
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

def odmkImgDivXfadeTelescope(imgList, framesPerBeat, imgOutDir, inOut=0, imgOutNm='None'):
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
            imgItr1 = odmkEyeRescale(imgClone1, newDimX, newDimY)
            imgItr2 = odmkEyeRescale(imgClone2, newDimX, newDimY)
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

def odmkEyeEchoBpm(imgList, numFrames, echoTrig, numEcho, echoStep, imgOutDir, echoDecay = 'None', imgOutNm='None'):
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
    
    rIdxArray = randomIdx(numFrames, len(imgList))
    rIdxEchoArray = randomIdx(numFrames, len(imgList))
    
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
    
def odmkImgColorEnhance(imgList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None'):
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
    
def odmkImgCEnTelescope(imgList, xLength, framesPerSec, xfadeFrames, imgOutDir, imgOutNm='None', inOut=0):
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
            imgItr = odmkEyeRescale(imgClone, newDimX, newDimY)
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
# // *---::ODMKEYE - EYE image array processing prototype::---*
# // *--------------------------------------------------------------* //

def odmkPxlRndReplace(imgArray, sLength, framesPerSec, FrameCtrl, imgOutDir, imgOutNm='None', numPxl=0):
    ''' swap n pixels per frame from two images'''

    if imgOutNm != 'None':
        imgPrototypeNm = imgOutNm
    else:
        imgPrototypeNm = 'imgXfadeRot'

    SzX = imgArray[0].shape[1]
    SzY = imgArray[0].shape[0]

    numFrames = int(ceil(sLength * framesPerSec))
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


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //

print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMK EYE IMG-2-VIDEO Experiments::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : Loading & Formatting img and sound files
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Set Program Master Parameters::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //
# // *-- <<<PRIMARY-PARAMETERS>>> --*
# // *---------------------------------------------------------------------* //

xLength = 9         # length of signal in seconds:
fs = 44100.0        # audio sample rate:
bWidth = 24         # audio sample bit width
framesPerSec = 30.0 # video frames per second:
#ub313 bpm = 178 , 267
bpm = 93    #glamourGoat14

timeSig = 4         # time signature: 4 = 4/4; 3 = 3/4

# eyeDir = 'C:/Users/djoto-odmk/odmk-sci/odmk_code/odmkPython/eye/odmkSrc/'
eyeDir = rootDir+'eye/eyeSrc/'

# cntrlLoadX - Image loading controls
# cntrlLoadX = <n>
# <0> Bypass  => No images loaded
# <1> Load    => choose raw image dir (a directory of random .jpg images)
cntrlLoadX = 1
if cntrlLoadX == 1:
    # jpgSrcDir: set the directory path used for processing .jpg files
    #jpgSrcDir = eyeDir+'totwnEyesSrc/'
    #jpgSrcDir = eyeDir+'odmkGorgulan56Scale1280x960/'    # pre-scaled or scaled img list
    #jpgSrcDir = eyeDir+'odmkGorgulan56Scale360x240/'
    jpgSrcDir = eyeDir+'odmkG1Src2/'
    #jpgSrcDir = eyeDir+'odmkGorgulanSrc/'

cntrlScaleAll = 0
# cntrlScaleAll - Image scale processing
# <0> Bypass  => No scaling
# <1> Scale
#cntrlScaleAll = 0
if cntrlScaleAll == 1:
    high = 0    # "head crop" cntrl - <0> => center crop, <1> crop from img top
    # *** Define Dir & File name for Scaling
    # scaleFileName: file name used for output processed .jpg (appended with 000X)
    scaleFileName = 'odmkGorgulan56'
    # scaleOutDirName: set the directory path used for output processed .jpg
    scaleOutDirName = 'odmkGorgulan56Scale360x240/'

# <<<selects pixel banging algorithm>>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 1  => Image Random Select Algorithm              ::odmkImgRndSel::
# 2  => Image Random BPM Algorithm                 ::odmkImgRndSel::
# 3  => Image Random Select Telescope Algorithm    ::**nofunc**::
# 4  => Image BPM rotating Sequence Algorithm      ::odmkImgRotateSeq::
# 5  => Image Rotate & alternate sequence          ::**nofunc**::
# 6  => Image CrossFade Sequencer (alpha-blend)    ::odmkImgXfade::
# 67 => Image CrossFade Sequencer (solarize)       ::odmkImgSolXfade::
# 7  => Image Divide CrossFade Sequencer           ::odmkImgDivXfade::
# 8  => Image CrossFade Rotate sequence            ::odmkImgXfadeRot::
# 9  => Image telescope sequence                   ::odmkImgTelescope::
# 10 => Image CrossFade telescope Sequence         ::odmkImgXfadeTelescope::
# 11 => Image Div Xfade telescope Sequence         ::odmkImgDivXfadeTelescope::
# 12 => Image Eye Echo BPM Sequencer               ::odmkEyeEchoBpm::
# 20 => Image Color Enhance BPM Algorithm          ::odmkImgColorEnhance::
# 21 => Image Color Enhance Telescope Algorithm    ::odmkImgCEnTelescope::



cntrlEYE = 21

if cntrlEYE != 0:
    # eyeOutFileName: file name used for output processed .jpg (appended with 000X)
    #eyeOutFileName = 'eyeEchoGorgulan1v'
    eyeOutFileName = 'odmkGSrc1CEnhanceTsc'
    # eyeOutDirName: set the directory path used for output processed .jpg
    #eyeOutDirName = 'eyeEchoGorgulan1vDir/'
    eyeOutDirName = 'odmkGSrc1CEnhanceTscDir/'


# <<<selects image post-processing function>>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 1  => Image Random Select Algorithm              ::odmkImgRndSel::
postProcess = 0

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




print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Set Master Dimensions::---*')
print('// *--------------------------------------------------------------* //')


mstrSzX = 360
mstrSzY = 240

#mstrSzX = 1280
#mstrSzY = 960

# ?D Standard video aspect ratio
#mstrSzX = 1280
#mstrSzY = 720

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

print('\nOutput frame width = '+str(mstrSzX))
print('Output frame Heigth = '+str(mstrSzY))

# // *---------------------------------------------------------------------* //

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Instantiate clocking/sequencing object::---*')
print('// *--------------------------------------------------------------* //')

eyeClks = clks.odmkClocks(xLength, fs, bpm, framesPerSec)

# // *---------------------------------------------------------------------* //

# example: num frames per beat (quantized)
# fpb = int(np.ceil(eyeClks.framesPerBeat))

# example: simple sequence -> downFrames (1 on downbeat frame, o elsewhere)
# eyeDFrames = eyeClks.clkDownFrames()




## print an image object from the list
## odmkEyePrint(imgObjList[3], 'MFKD', 1)
#
#gz3 = imgObjList[0]
#
#gz3_r = gz3[:, :, 0]
#gz3_g = gz3[:, :, 1]
#gz3_b = gz3[:, :, 2]
#
#gzTest = np.dstack((gz3_r, gz3_g, gz3_b))
## odmkEyePrint(gzTest, 'MFKD Clone test', 2)
#
## gzTest = gz3
#misc.imsave('imgSrc/exp1/testImgObjPrint.jpg', gzTest)


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Import all source img (.jpg) in a directory::---*')
print('// *--------------------------------------------------------------* //')

# Bypass <0> ; choose raw image dir <1> ; or pre-scaled images <2>
#cntrlLoadX = 2    #*** defined above


if cntrlLoadX == 1:    # process raw .jpg folder
    # jpgSrcDir = eyeDir+'process/'
    # jpgSrcDir = eyeDir+'kei777/'
    # jpgSrcDir = eyeDir+'totwnEyesSrc/'    #*** defined above
    [imgSrcArray, imgSrcNmArray] = importAllJpg(jpgSrcDir)

    print('\nCreated python lists:')
    print('<<imgSrcArray>> (image data object list)')
    print('<<imgSrcNmArray>> (image data object names)\n')

#elif cntrlLoadX == 2:    # use if scaled img already exist
#    # jpgSrcDir = eyeDir+'gorgulanScaled/'    # pre-scaled img list
#    # jpgSrcDir = eyeDir+'totwnEyesScale/'    # pre-scaled img list
#    [imgSrcArray, imgSrcNmArray] = importAllJpg(jpgSrcDir)
#
#    print('\nCreated python lists:')
#    print('<<processScaledArray>>   (img data objects)')
#    print('<<processScaledNmArray>> (img names)\n')

else:    # Bypass
    print('\nCTRL Image Loading Bypassed\n')


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
    

    [processScaledArray, processScaledNmArray] = odmkScaleAll(imgSrcArray, mstrSzX, mstrSzY, w=1, high=0, outDir=processScaleDir, outName=outNameRoot)

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
# begin : ODMK pixel banging
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# <<<selects pixel banging algorithm>>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 1  => Image Random Select Algorithm              ::odmkImgRndSel::
# 2  => Image Random BPM Algorithm                 ::odmkImgRndSel::
# 3  => Image Random Select Telescope Algorithm    ::**nofunc**::
# 4  => Image BPM rotating Sequence Algorithm      ::odmkImgRotateSeq::
# 5  => Image Rotate & alternate sequence          ::**nofunc**::
# 6  => Image CrossFade Sequencer (alpha-blend)    ::odmkImgXfade::
# 67 => Image CrossFade Sequencer (solarize)       ::odmkImgSolXfade::
# 7  => Image Divide CrossFade Sequencer           ::odmkImgDivXfade::
# 8  => Image CrossFade Rotate sequence            ::odmkImgXfadeRot::
# 9  => Image telescope sequence                   ::odmkImgTelescope::
# 9  => Image telescope sequence                   ::odmkImgTelescope::
# 10 => Image CrossFade telescope Sequence         ::odmkImgXfadeTelescope::
# 11 => Image Div Xfade telescope Sequence         ::odmkImgDivXfadeTelescope::
# 20 => Image Color Enhance BPM Algorithm          ::odmkImgColorEnhance::
# 21 => Image Color Enhance Telescope Algorithm    ::odmkImgCEnTelescope::


# cntrlEYE = 2    # ***DEFINED ABOVE***


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
    [imgRndSelOutA, imgRndSelNmA] = odmkImgRndSel(imgSrcArray, numFrames, imgRndSeldir, imgOutNm=imgRndSelNm)

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

    [imgRndBpmOutA, imgRndBpmNmA] =  odmkImgRndBpm(imgSrcArray, numFrames, eyeDFrames, imgRndBpmdir, imgOutNm=imgRndBpmNm)

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

    [imgRotArray, imgRotNmArray] = odmkImgRotateSeq(imgSrc, numFrames, outDir, imgOutNm=outNm, rotDirection=rotD)


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
    
    [xfadeOutA, xfadeOutNmA] = odmkImgXfade(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

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

    [xfadeOutA, xfadeOutNmA] = odmkImgXfadeAll(imgSrcArray, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

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

    [xSolOutA, xSolOutNmA] = odmkImgSolXfade(imgSrcArray, framesPerBeat, outSolXfadeDir, imgOutNm=outSolXfadeNm)

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

    [imgSrcObj, imgSrcObjNm] = importAllJpg(srcDivXfadeDir)

    framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))

    [divXfadeOutA, divXfadeOutNmA] = odmkImgDivXfade(imgSrcArray, framesPerBeat, outDivXfadeDir, imgOutNm='eyeDivXfade')

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

    [xfadeRotOutA, xfadeRotOutNmA] = odmkImgXfadeRot(imgSrcArray, xLength, framesPerSec, xFrames, outXfadeRotDir, imgOutNm=outXfadeRotNm, rotDirection=rotD)

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
        

    odmkImgTelescope(imgSrcArray, xLength, fps, fpb, imgTelescOutDir, imgOutNm=imgTelescNm, inOut=inOrOut)

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

    odmkImgXfadeTelescope(processScaledArray, fpb, imgXfadeTelOutDir, inOut=inOrOut, imgOutNm=imgXfadeTelNm)

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

    odmkImgDivXfadeTelescope(processScaledArray, fpb, imgDivXfadeTelOutDir, inOut=inOrOut, imgOutNm=imgDivXfadeTelNm)

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
    odmkEyeEchoBpm(imgSrcArray, numFrames, eyeDFrames, numECHOES, stepECHOES, imgRndBpmdir, echoDecay = decayECHOES, imgOutNm=imgRndBpmNm)
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
    
    [xCEnOutA, xCEnOutNmA] = odmkImgColorEnhance(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

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
    
    odmkImgCEnTelescope(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm, inOut=inOrOut)

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

    repeatAllImg(srcDir, n, w=1, repeatDir=repeatDir, repeatName=repeatNm)

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


    repeatReverseAllImg(repeatSrcDir, n, w=1, repeatReverseDir=repeatRevDir, repeatReverseName=repeatRevNm)



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
