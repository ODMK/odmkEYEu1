# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((odmkEYEu2.py))::__

# Python ODMK img processing research
# ffmpeg experimental

# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
from math import atan2, floor, ceil
import random
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt


from odmkClear import *

# temp python debugger - use >>>pdb.set_trace() to set break
import pdb


# // *---------------------------------------------------------------------* //

clear_all()

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
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Import all source img (.jpg) in a directory::---*')
    print('// *--------------------------------------------------------------* //')

    # generate list all files in a directory
    imgSrcList = []
    imgObjList = []

    print('\nLoading source images from the following path:')
    print(srcDir)

    for filename in os.listdir(srcDir):
        imgSrcList.append(filename)
        imgPath = srcDir+filename
        imgObjTemp = misc.imread(imgPath)
        imgObjList.append(imgObjTemp)
    imgCount = len(imgSrcList)
    print('\nFound '+str(imgCount)+' images in the folder:\n')
    print(imgSrcList)
    print('\nCreated python lists:')
    print('<<imgObjList>> (ndimage objects)')
    print('<<imgSrcList>> (img names)\n')
    print('// *--------------------------------------------------------------* //')

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


#def reverseIdx(imgArray):

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


def odmkCenterPxl(img):
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    cPxl_x = floor(imgWidth / 2)
    cPxl_y = floor(imgHeight / 2)
    centerPxl = [cPxl_x, cPxl_y]
    return centerPxl


def odmkHelixPxl(vLength, theta):
    helixPxlArray = []
    # generate a vector of 'center pixels' moving around unit circle
    # quantized cyclicZn coordinates!
    # theta = angle of rotation per increment
    return helixPxlArray


def odmkFiboPxl(vLength, theta):
    fiboPxlArray = []
    # generate a vector of 'center pixels' tracing a fibonnaci spiral
    # quantized fibonnaci coordinates!
    # theta = angle of rotation per increment
    return fiboPxlArray


def odmkEyeCrop(img, SzX, SzY):
    # crop img to SzX width x SzY height

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
            imgCrop_r[j, k] = img_r[j + hdiff_ths, k + wdiff_lhs]
            imgCrop_g[j, k] = img_g[j + hdiff_ths, k + wdiff_lhs]
            imgCrop_b[j, k] = img_b[j + hdiff_ths, k + wdiff_lhs]
    imgCrop = np.dstack((imgCrop_r, imgCrop_g, imgCrop_b))
    return imgCrop


def odmkEyeDim(img, SzX, SzY):
    # Zoom and Crop image to match source dimensions
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    if (imgWidth == SzX and imgHeight == SzY):
        # no processing
        imgScaled = img

    elif (imgWidth == SzX and imgHeight > SzY):
        # cropping only
        imgScaled = odmkEyeCrop(img, SzX, SzY)

    elif (imgWidth > SzX and imgHeight == SzY):
        # cropping only
        imgScaled = odmkEyeCrop(img, SzX, SzY)

    elif (imgWidth > SzX and imgHeight > SzY):
        # downscaling and cropping
        wdiff = imgWidth - SzX
        hdiff = imgHeight - SzY
        minDiff = min(wdiff, hdiff)
        if wdiff <= hdiff:
            zoomFactor = (imgWidth - minDiff) / imgWidth
        else:
            zoomFactor = (imgHeight - minDiff) / imgHeight
        # print('zoomFactor = '+str(zoomFactor))
        imgScaled = odmkEyeZoom(img, zoomFactor)
        # misc.imsave('imgSrc/exp1/myFirstKikDrumZOOM00001.jpg', imgScaled)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY)
        # misc.imsave('imgSrc/exp1/myFirstKikDrumZCrop00001.jpg', imgScaled)

    elif (imgWidth < SzX and imgHeight == SzY):
        # upscaling and cropping
        wdiff = SzX - imgWidth
        zoomFactor = (imgWidth + wdiff) / imgWidth
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY)

    elif (imgWidth == SzX and imgHeight < SzY):
        # upscaling and cropping
        hdiff = SzY - imgHeight
        zoomFactor = (imgHeight + hdiff) / imgHeight
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY)

    elif (imgWidth < SzX and imgHeight < SzY):
        # upscaling and cropping (same aspect ratio -> upscaling only)
        wdiff = mstrSzX - imgWidth
        hdiff = mstrSzY - imgHeight
        maxDiff = max(wdiff, hdiff)
        if wdiff >= hdiff:
            zoomFactor = (imgWidth + maxDiff) / imgWidth
        else:
            zoomFactor = (imgHeight + maxDiff) / imgHeight
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY)

    elif (imgWidth > SzX and imgHeight < SzY):
        # upscaling and cropping
        wdiff = imgWidth - SzX
        hdiff = SzY - imgHeight
        zoomFactor = (imgHeight + hdiff) / imgHeight
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY)

    elif (imgWidth < SzX and imgHeight > SzY):
        # upscaling and cropping
        wdiff = SzX - imgWidth
        hdiff = imgHeight - SzY
        zoomFactor = (imgWidth + wdiff) / imgWidth
        imgScaled = odmkEyeZoom(img, zoomFactor)
        imgScaled = odmkEyeCrop(imgScaled, SzX, SzY)

    return imgScaled


def odmkScaleAll(srcObjList, SzX, SzY, w, outDir='None', outName='None'):
    ''' rescales and normalizes all .jpg files in image object list.
        scales, zooms, & crops images to dimensions SzX, SzY
        Assumes srcObjList of ndimage objects exists in program memory
        Typically importAllJpg is run first.
        if w == 1, write processed images to output directory'''
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Rescale and Normalize All imgages in folder::---*')
    print('// *--------------------------------------------------------------* //')

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
        imgScaled = odmkEyeDim(srcObjList[k], SzX, SzY)
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


# -----------------------------------------------------------------------------
# img Post-processing func
# -----------------------------------------------------------------------------

def concatAllDir(dirList, w=0, concatDir='None', concatName='None'):
    ''' Takes a list of directories containing .jpg images,
        renames and concatenates all files into new output directory.
        The new file names are formatted for processing with ffmpeg'''

    # check write condition, if w=0 ignore outDir
    if w == 0 and concatDir != 'None':
        print('Warning: write = 0, outDir ignored')
    if w == 1:
        if concatDir == 'None':
            print('Error: outDir must be specified; processed img will not be saved')
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
    #pdb.set_trace()
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
            imgScaledFull = concatDir+imgNormalizeNm
            misc.imsave(imgScaledFull, imgConcatObjList[i])

    return [imgConcatObjList, imgConcatSrcList]


# -----------------------------------------------------------------------------
# ODMK img Pixel-Banging Algorithms
# -----------------------------------------------------------------------------



# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //

print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMK EYE ffmpeg Experiments::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : Loading & Formatting img and sound files
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::Load input image files::---*')
#print('// *--------------------------------------------------------------* //')
#
## gzFile = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/odmkChebychevPolyBlk1.jpg'
## gz1File = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/myFirstKikDrum.jpg'
## gz2File = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/odmkYoukaiHasseiRed1.jpg'
## gzFile = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/barutanBreaks1.jpg'
#
#gz1File = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/odmkTriFoldKnt001.png'
#gz2File = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/odmkCHYpolyx.jpg'
#gz3File = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/myFirstKikDrum.jpg'
#
## // *---------------------------------------------------------------------* //
#
#gz1 = misc.imread(gz1File)
#
#gz1_name = os.path.split(gz1File)[1]
#gz1_name = gz1_name.split('.')[0]
#
#gz1Width = gz1.shape[1]
#gz1Height = gz1.shape[0]
#
## Use as master eye dimensions
##mstrSzX = gz1Width
##mstrSzY = gz1Height
#
#print('\nImage1 Title = '+gz1_name)
#print('\ngz1 Width = '+str(gz1Width))
#print('gz1 Height = \n'+str(gz1Height))
#
## // *---------------------------------------------------------------------* //
#
#gz2 = misc.imread(gz2File)
#
#gz2_name = os.path.split(gz2File)[1]
#gz2_name = gz2_name.split('.')[0]
#
#gz2Width = gz2.shape[1]
#gz2Height = gz2.shape[0]
#
#print('\nImage2 Title = '+gz2_name)
#print('\ngz2 Width = '+str(gz2Width))
#print('gz2 Height = \n'+str(gz2Height))

# // *---------------------------------------------------------------------* //


rootDir = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/'

# If Dir does not exist, makedir:
# os.makedirs(path, exist_ok=True)

# // *---------------------------------------------------------------------* //

# // *---------------------------------------------------------------------* //
# // *--Set Master Dimensions--*
# // *---------------------------------------------------------------------* //

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Set Master Dimensions for output::---*')
print('// *--------------------------------------------------------------* //')

# Golden ratio frames:
mstrSzX = 1076
mstrSzY = 666

#mstrSzX = 1257
#mstrSzY = 777

# likely good size for final output
#mstrSzX = 1436
#mstrSzY = 888
print('\nOutput frame width = '+str(mstrSzX))
print('Output frame Heigth = '+str(mstrSzY))

# // *---------------------------------------------------------------------* //

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::Rescale and Normalize All imgages in folder::---*')
#print('// *--------------------------------------------------------------* //')
#
## generate list all files in a directory
#imgSrcList = []
#imgObjList = []
#
## path = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/eyeSrcExp1/'
#path = rootDir+'eyeSrcExp1/'
#print('\nLoading source images from the following path:')
#print(path)
#
#for filename in os.listdir(path):
#    imgSrcList.append(filename)
#    imgPath = path+filename
#    imgObjTemp = misc.imread(imgPath)
#    imgObjList.append(imgObjTemp)
#imgCount = len(imgSrcList)
#print('\nFound '+str(imgCount)+' images in the folder:\n')
#print(imgSrcList)
#print('\nCreated numpy arrays:')
#print('<<imgObjList>> (img data objects) and <<imgSrcList>> (img names)\n')
#
#print('// *--------------------------------------------------------------* //')

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

## // *---------------------------------------------------------------------* //
#
## scale all img objects in imgObjList
#
## Optional!: save all scaled images to dir
#gzScaledir = rootDir+'expScaled/'
#
#gzScaledNmArray = []
#gzScaledArray = []
#
## Find num digits required to represent max index
#n_digits = int(ceil(np.log10(imgCount))) + 2
#nextInc = 0
#for k in range(imgCount):
#    gzSTransc = odmkEyeDim(imgObjList[k], mstrSzX, mstrSzY)
#    # auto increment output file name
#    nextInc += 1
#    zr = ''    # reset lead-zero count to zero each itr
#    for j in range(n_digits - len(str(nextInc))):
#        zr += '0'
#    strInc = zr+str(nextInc)
#    gzScaledNm = 'gzScaled'+strInc+'.jpg'
#    gzScaledFull = gzScaledir+gzScaledNm
#    misc.imsave(gzScaledFull, gzSTransc)
#    gzScaledNmArray.append(gzScaledNm)
#    gzScaledArray.append(gzSTransc)
#
#print('\nScaled all images in the source directory\n')
#print('Renamed files: "gzScaled00X.jpg"')
#
#print('\nCreated numpy arrays:')
#print('<<gzScaledArray>> (img data objects) and <<gzScaledNmArray>> (img names)\n')
#
#print('Saved Scaled images to the following location:')
#print(gzScaledir)
#print('\n')
#
#print('// *--------------------------------------------------------------* //')

## Crop image to master dimensions
#eyeCrop1 = odmkEyeCrop(imgObjList[0], mstrSzX, mstrSzY)
## odmkEyePrint(eyeCrop1, 'MFKD Cropped', 3)
#misc.imsave('imgSrc/exp1/myFirstKikDrumCrop00001.jpg', eyeCrop1)
#
## Zoom and Crop image to master dimensions
#
#img3Width = gz3.shape[1]
#img3Height = gz3.shape[0]

#wdiff = img3Width - mstrSzX
#hdiff = img3Height - mstrSzY
#minDiff = min(wdiff, hdiff)
#print('minDiff = '+str(minDiff))
#
#if wdiff <= hdiff:
#    zoomFactor = (img3Width - minDiff) / img3Width
#else:
#    zoomFactor = (img3Height - minDiff) / img3Height
#
#print('zoomFactor = '+str(zoomFactor))
#
#gz3ZOOM = odmkEyeZoom(gz3, zoomFactor)
#misc.imsave('imgSrc/exp1/myFirstKikDrumZOOM00001.jpg', gz3ZOOM)
#
#gz3ZCrop = odmkEyeCrop(gz3ZOOM, mstrSzX, mstrSzY )
#misc.imsave('imgSrc/exp1/myFirstKikDrumZCrop00001.jpg', gz3ZCrop)

#gz3Scaled = odmkEyeDim(gz3, mstrSzX, mstrSzY)
#misc.imsave('imgSrc/exp1/myFirstKikDrumScaled00001.jpg', gz3Scaled)


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

# ***GOOD***

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::ODMKEYE - Image Random Select Algorithm::---*')
#print('// *--------------------------------------------------------------* //')
#
## for n frames: 
## randomly select an image from the source dir, hold for h frames
#
##print('\n')
##print('// *--------------------------------------------------------------* //')
##print('// *---::Import all source img (.jpg) in a directory::---*')
##print('// *--------------------------------------------------------------* //')
#
#jpgSrcDir = rootDir+'process/'
#
#[processObjList, processSrcList] = importAllJpg(jpgSrcDir)
#
#gorgulanDir = rootDir+'gorgulanScale/'
#
##[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 0)
##[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 0, outName='gorgulan')
##[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 1, outDir=gorgulanDir)
#[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 1, outDir=gorgulanDir, outName='gorgulan')
#
#
#print('\nCreated python lists:')
#print('<<processScaledArray>> (img data objects) and <<processScaledNmArray>> (img names)\n')
#
##print('Saved Scaled images to the following location:')
##print(processScaledir)
#print('\n')
#
#print('// *--------------------------------------------------------------* //')
#
## output dir where processed img files are stored:
#imgRndSeldir = rootDir+'exp5/'
## If Dir does not exist, makedir:
#os.makedirs(imgRndSeldir, exist_ok=True)
#
#imgRndSelNm = 'imgRndSelOut'
#
## final video length in seconds
#eyeLength = 46
#frameRate = 30
#numFrames = eyeLength * frameRate
#
##num_process = 393
#
#imgRndSelNmArray = []
#imgRndSelArray = []
#
#rIdxArray = randomIdx(numFrames, len(processSrcList))
#
#imgCount = numFrames
#n_digits = int(ceil(np.log10(imgCount))) + 2
#nextInc = 0
#for i in range(numFrames):
#    nextInc += 1
#    zr = ''
#    for j in range(n_digits - len(str(nextInc))):
#        zr += '0'
#    strInc = zr+str(nextInc)
#    imgNormalizeNm = imgRndSelNm+strInc+'.jpg'
#    imgRndSelFull = imgRndSeldir+imgNormalizeNm
#    misc.imsave(imgRndSelFull, processScaledArray[rIdxArray[i]])
#
## // *--------------------------------------------------------------* //
## // *---::ODMKEYE - END Image Random Select Algorithm::---*')
## // *--------------------------------------------------------------* //

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMKEYE - Image Random Select Telescope Algorithm::---*')
print('// *--------------------------------------------------------------* //')

# for n frames: 
# randomly select an image from the source dir, hold for h frames

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::Import all source img (.jpg) in a directory::---*')
#print('// *--------------------------------------------------------------* //')

jpgSrcDir = rootDir+'process/'

[processObjList, processSrcList] = importAllJpg(jpgSrcDir)

gorgulanDir = rootDir+'gorgulanScale/'

#[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 0)
#[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 0, outName='gorgulan')
#[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 1, outDir=gorgulanDir)
[processScaledArray, processScaledNmArray] = odmkScaleAll(processObjList, mstrSzX, mstrSzY, 1, outDir=gorgulanDir, outName='gorgulan')


print('\nCreated python lists:')
print('<<processScaledArray>> (img data objects) and <<processScaledNmArray>> (img names)\n')

#print('Saved Scaled images to the following location:')
#print(processScaledir)
print('\n')

print('// *--------------------------------------------------------------* //')

# output dir where processed img files are stored:
imgTelescRnddir = rootDir+'exp6/'
# If Dir does not exist, makedir:
os.makedirs(imgTelescRnddir, exist_ok=True)

imgTelescRndNm = 'imgRndSelOut'

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
rIdxArray = randomIdx(numFrames+tlItr, len(processSrcList))

imgCount = numFrames
n_digits = int(ceil(np.log10(imgCount))) + 2
nextInc = 0
for i in range(tlItr):
    newDimX = SzX
    newDimY = SzY
    imgClone = processScaledArray[rIdxArray[i]]
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


# // *---------------------------------------------------------------------* //
# // *--Incremental Rotate--*
# // *---------------------------------------------------------------------* //

## ex, rotate image by 45 deg:
## rotate_gz1 = ndimage.rotate(gz1, 45, reshape=False)
#
## dir where processed img files are stored:
#gzdir = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/exp1/'
#
## num_gz = 192
#num_gz = 111
#zn = cyclicZn(num_gz)
#
#
#gz1NmArray = []
#gz1Array = []
#
#nextInc = 0
#for i in range(num_gz):
#    zr = ''
#    ang = (atan2(zn[i].imag, zn[i].real))*180/np.pi
#    rotate_gz1 = ndimage.rotate(gz1, ang, reshape=False)
#    nextInc += 1
#    # Find num digits required to represent max index
#    n_digits = int(ceil(np.log10(num_gz))) + 2
#    for j in range(n_digits - len(str(nextInc))):
#        zr += '0'
#    strInc = zr+str(nextInc)
#    gz1Nm = gzdir+gz1_name+strInc+'.jpg'
#    misc.imsave(gz1Nm, rotate_gz1)
#    gz1NmArray.append(gz1Nm)
#    gz1Array.append(rotate_gz1)

# // *---------------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
# // *--Incremental Rotate & alternate--*
# // *---------------------------------------------------------------------* //

## ex, rotate image by 45 deg:
## rotate_gz1 = ndimage.rotate(gz1, 45, reshape=False)
#
## dir where processed img files are stored:
#gzdir = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/exp1/'
#
#eye_name = 'odmkRotFlash1'
#
## num_gz = 192
#num_gz = 256
#zn = cyclicZn(num_gz)
#
#
#gz1NmArray = []
#gz1Array = []
#
#nextInc = 0
#for i in range(num_gz):
#    zr = ''
#    ang = (atan2(zn[i].imag, zn[i].real))*180/np.pi
#    if ((i % 4) == 1 or (i % 4) == 2):
#        rotate_gz1 = ndimage.rotate(gz1, ang, reshape=False)
#    else:
#        rotate_gz1 = ndimage.rotate(gzRESIZEx02, ang, reshape=False)
#    nextInc += 1
#    # Find num digits required to represent max index
#    n_digits = int(ceil(np.log10(num_gz))) + 2
#    for j in range(n_digits - len(str(nextInc))):
#        zr += '0'
#    strInc = zr+str(nextInc)
#    gz1Nm = gzdir+eye_name+strInc+'.jpg'
#    misc.imsave(gz1Nm, rotate_gz1)
#    gz1NmArray.append(gz1Nm)
#    gz1Array.append(rotate_gz1)

# // *---------------------------------------------------------------------* //


# // *---------------------------------------------------------------------* //
# // *--Eye COS Crossfader--*
# // *---------------------------------------------------------------------* //

## crossfade between two images using cos windows for constant power out:
## rotate_gz1 = ndimage.rotate(gz1, 45, reshape=False)
#
## dir where processed img files are stored:
#gzdir = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/exp1/'
#
#eye_name = 'odmkCosXfade'
#
## num_gz = 192
#num_gz = 256
##zn = cyclicZn(num_gz)
#
## initialize a 1/4 period Cos window
#winCosXfade = quarterCos(num_gz)
#
#gzCosxA_r = gzScaledArray[0][:, :, 0]
#gzCosxA_g = gzScaledArray[0][:, :, 1]
#gzCosxA_b = gzScaledArray[0][:, :, 2]
#
#gzCosxB_r = gzScaledArray[1][:, :, 0]
#gzCosxB_g = gzScaledArray[1][:, :, 1]
#gzCosxB_b = gzScaledArray[1][:, :, 2]
#
##imgZoom = np.dstack((imgZoom_r, imgZoom_g, imgZoom_b))
#
#
#gzAHalf_r = 1.5 * gzCosxA_r
#gzAHalf_g = 1.5 * gzCosxA_g
#gzAHalf_b = 1.5 * gzCosxA_b
#gzAHalf_recon = np.dstack((gzAHalf_r, gzAHalf_g, gzAHalf_b))
#
#gzAHalf_reconFull = gzdir+'gzAHalf_recon.jpg'
#misc.imsave(gzAHalf_reconFull, gzAHalf_recon)
#
#gzBHalf_r = 0.5 * gzCosxB_r
#gzBHalf_g = 0.5 * gzCosxB_g
#gzBHalf_b = 0.5 * gzCosxB_b
#gzBHalf_recon = np.dstack((gzBHalf_r, gzBHalf_g, gzBHalf_b))
#
#gzBHalf_reconFull = gzdir+'gzBHalf_recon.jpg'
#misc.imsave(gzBHalf_reconFull, gzBHalf_recon)
#
#gzMix_r = gzAHalf_r + gzBHalf_r
#gzMix_g = gzAHalf_g + gzBHalf_g
#gzMix_b = gzAHalf_g + gzBHalf_b
#gzMix_recon = np.dstack((gzMix_r, gzMix_g, gzMix_b))
#
#gzMix_reconFull = gzdir+'gzMix_recon.jpg'
#misc.imsave(gzMix_reconFull, gzMix_recon)
#
#
#
#
#gzCosxNmArray = []
#gzCosxArray = []
#
#nextInc = 0
#for i in range(num_gz):
#    # interpolate from one image to another using COS windows
#
#    
#        
#    zr = ''        
#    nextInc += 1
#    # Find num digits required to represent max index
#    n_digits = int(ceil(np.log10(num_gz))) + 2
#    for j in range(n_digits - len(str(nextInc))):
#        zr += '0'
#    strInc = zr+str(nextInc)
#    gz1Nm = gzdir+eye_name+strInc+'.jpg'
#    misc.imsave(gz1Nm, rotate_gz1)
#    gz1NmArray.append(gz1Nm)
#    gz1Array.append(rotate_gz1)

# // *---------------------------------------------------------------------* //

# ***GOOD***

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::ODMKEYE - Image telescope Algorithm::---*')
#print('// *--------------------------------------------------------------* //')
#
## iterate zoom out & overlay, zoom in & overlay
#
## defined above!
#SzX = mstrSzX
#SzY = mstrSzY
#
## num_img = 56
#num_img = 46
#inOrOut = 0     # telescope direction: 0 = in, 1 = out
#hop_sz = 5      # number of pixels to scale img each iteration
#
## dir where processed img files are stored:
#imgTelescdir = rootDir+'exp1/'
#if inOrOut == 0:
#    imgTelescNm = 'telescopeOut'
#else:
#    imgTelescNm = 'telescopeIn'
#
## **temp? - selects a specific image from array, should be more generic***
#gzIdx = 2
#imgClone = gzScaledArray[gzIdx]
#imgCloneNm = gzScaledNmArray[gzIdx]
#
#
#imgTelescopeNmArray = []
#imgTelescopeArray = []
#
#newDimX = SzX
#newDimY = SzY
#for i in range(num_img):
#    if newDimX > 2:
#        newDimX -= hop_sz
#    if newDimY > 2:
#        newDimY -= hop_sz
#    # scale image to new dimensions
#    imgItr = odmkEyeRescale(imgClone, newDimX, newDimY)
#    # region = (left, upper, right, lower)
#    # subbox = (i + 1, i + 1, newDimX, newDimY)
#
#    for j in range(SzY):
#        for k in range(SzX):
#            if ((j > i) and (j < newDimY) and (k > i) and (k < newDimX)):
#                imgClone[j, k, :] = imgItr[j - i, k - i, :]
#    # Find num digits required to represent max index
#    n_digits = int(ceil(np.log10(num_img))) + 2
#    zr = ''
#    if inOrOut == 1:
#        for j in range(n_digits - len(str(i + 1))):
#            zr += '0'
#        strInc = zr+str(i + 1)
#    else:
#        for j in range(n_digits - len(str(num_img - (i + 1)))):
#            zr += '0'
#        strInc = zr+str(num_img - (i + 1))
#    imgTelescFull = imgTelescdir+imgTelescNm+strInc+'.jpg'
#    misc.imsave(imgTelescFull, imgClone)
#    imgTelescopeNmArray.append(imgTelescNm+strInc+'.jpg')
#    imgTelescopeArray.append(imgClone)
#
#print('\nProcessed source img: \n'+imgCloneNm)
#print('Renamed files: "'+imgTelescNm+'00X.jpg"')
#
#print('\nCreated numpy arrays:')
#print('<<imgTelescopeArray>> (img data objects) and <<imgTelescopeNmArray>> (img names)\n')
#
#print('Saved Scaled images to the following location:')
#print(imgTelescdir)
#print('\n')
#
#print('// *--------------------------------------------------------------* //')    
#
# // *---------------------------------------------------------------------* //
# // *--end: Image telescope Out--*
# // *---------------------------------------------------------------------* //


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

# ***GOOD***

#print('\n')
#print('// *--------------------------------------------------------------* //')
#print('// *---::Concatenate all images in directory list::---*')
#print('// *--------------------------------------------------------------* //')
#
#
#loc1 = rootDir+'uCache/imgTelescopeIn1_1436x888_56x/'
#loc2 = rootDir+'uCache/imgTelescopeIn2_1436x888_56x/'
#loc3 = rootDir+'uCache/imgTelescopeIn3_1436x888_56x/'
#loc4 = rootDir+'uCache/imgTelescopeIn4_1436x888_56x/'
#
#dirList = [loc1, loc2, loc3, loc4]
#
#gorgulanConcatDir = rootDir+'gorgulanConcatExp/'
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



# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : img post-processing
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //

# plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //

# // *---------------------------------------------------------------------* //
# experimental algorithms

# *** Figure out frame-rate <=> sample-rate synchronization

# *** Add sound clip synched to each pixelbang itr
# *** add complete soundtrack synched to frame-rate

# create util func to generate random list of img in dir
# create util func to reverse numbering of img

# *** randomly select pixelbang algorithm for each itr
# ***define 'channels' interleaved pixelbang algorithms of varying duration


# randome pixel replacement from img1 to img2
# -replace x% of pixels randomly each frame, replaced pixels pulled out of pool
# -for next random selection to choose from until all pixels replaced

# 2D wave eqn modulation
# -modulate colors/light, or modulate between two images ( > 50% new image pixel)

# 2D wave modulation iterating zoom

# trails, zoom overlay interpolation...

# fractal pixel replace - mandelbrot set algorithm

# random pixel replace through rotating image set - image is rotated by
# randomly replacing pixels while skipping through bounded sliding window of
# incrementally rotated windows

# sliding probability window -> for vector of length n, randomly choose values
# bounded by a sliding window (goal is to evolve through a list of img)

# // *---------------------------------------------------------------------* //
