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
import glob, shutil
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
import odmkVIDEOu as odmkv

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(2, rootDir+'DSP')
import odmkClocks as clks
import odmkWavGen1 as wavGen

#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(3, rootDir+'audio')
import native_wav as wavio

# temp python debugger - use >>>pdb.set_trace() to set break
import pdb



# // *---------------------------------------------------------------------* //
# plt.close('all')
clear_all()

# // *---------------------------------------------------------------------* //

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# /////////////////////////////////////////////////////////////////////////////
# *---utility func--*
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


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



print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Initializing EYE Object Parameters::---*')
print('// *--------------------------------------------------------------* //')

# /////////////////////////////////////////////////////////////////////////////
# *---Begin INITIALIZE: Set top-level parameters--*
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

eyeDir = rootDir+'eye/eyeSrc/'

# 10 21 21 16 16 16

# 180/26,  90/1, 90/1, 90/26, 180/222, 45/24

xLength = 113
#xLength = 5

fs = 48000.0        # audio sample rate:
framesPerSec = 30 # video frames per second:
numFrames = int(ceil(xLength * framesPerSec))
# ub313 bpm = 178 , 267  
# 39 - 78 - 156
#bpm = 186            # 93 - 186 - 372
bpm = 128            # 64.5 - 129 - 258
timeSig = 4         # time signature: 4 = 4/4; 3 = 3/4

# set format of source img { fjpg, fbmp }
imgFormat = 'fjpg'


# HD Standard video aspect ratio
#SzX = 1920
#SzX = 1080
#SzY = 1080

#SzX = 1560
#SzY = 1560

#SzX = 1080
#SzY = 1080

SzX = 1280
SzY = 720

# 1920x1080 quarter frame
#SzX = 960
#SzY = 540

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


# <<< Select Top-Level processing >>>
# *---------------------------------------------------------------------------*
# 1  => Main Algorithm Pixel-Banging
# 2  => Pre-Processing
# 3  => Post-Processing
topSwitch = 1


print('\nEye Directory = '+eyeDir+'\n')
print('Sequence Length (seconds) = '+str(xLength)+'\n')
print('Audio Sample Rate (cycles/sec) = '+str(fs)+'\n')
print('Video Frames per Second = '+str(framesPerSec)+'\n')
print('Sequence BPM = '+str(bpm)+'\n')
print('Sequence Time Signature = '+str(timeSig)+'\n')
print('Video Frame Dimensions [X, Y] = '+str(SzX)+' x '+str(SzY)+'\n')




# *---------------------------------------------------------------------------*

        

# /////////////////////////////////////////////////////////////////////////////
# *---Begin MAIN: Select top-level switches--*
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

if topSwitch == 1:

# <<< img array sequencing function Select >>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 1  => 1 directory of frames

# <<< Pixel Banging Algorithm Select >>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 100  => EYE Linear Select Algorithm            ::odmkImgLinSel::
# 101  => EYE Rotate Linear Select Algorithm     ::odmkImgRotLinSel::
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
# 222 => EYE Pixel Random Rotate Algorithm       ::odmkPxlRndRotate::
# 23 => EYE LFO Modulated Parascope Algorithm    ::odmkLfoParascope::
# 24 => EYE Bananasloth Recursive Algorithm      ::odmkBSlothRecurs::
# 26 => EYE Bananasloth Glitch 1                 ::odmkBSlothGlitch1::
# 27 => EYE Bananasloth Glitch 3                 ::odmkBSlothGlitch3::
# 28 => EYE LFO Horizontal Trails Algorithm      ::odmkLfoHtrails::

# 202  => EYE RanSel LFO Algorithm                 ::odmkImgRndSel::

# 623 => EYE FxSlothCultLife Glitch 1            ::odmkFxSlothCult1::

# 626 => EYE FxSlothCultLife Glitch 1            ::odmkFxSlothCult1::

# 626 => EYE Video VideoGoreXXZ01                ::odmkVideoGoreXXZ01::

# *--------------------------...


    cntrlLoadX = 1
    #cntrlScaleAll = 0
    srcSeqProcess = 0       # 0 = single directory, 1 = multiple directries  FIXIT!!
    cntrlEYE = 12
    preProcess = 0
    postProcess = 0         # 0 = bypass, 1 = ???, 2 = ???, 3 = concatAllDir(...)
    
  
elif topSwitch == 2:  
    
    # <<< Pre Processing Function Select >>>
    # *---------------------------------------------------------------------------*
    # 0  => Bypass
    # 1  => ?                                           ::?::
    # 2  => scale all img in dir                        ::odmkScaleAll::
    # 3  => convert .jpg img to .bmp                    ::convertJPGtoBMP::
    # 4  => ?                                           ::?::
    
    
    # Pre Processing
    cntrlLoadX = 0
    #cntrlScaleAll = 0
    srcSeqProcess = 0
    cntrlEYE = 0
    preProcess = 2
    postProcess = 0
  

elif topSwitch ==3:  
    
    # <<< Post Processing Function Select >>>
    # *---------------------------------------------------------------------------*
    # *---------------------------------------------------------------------------*
    # 0  => Bypass
    # 1  => ?                                           ::?::
    # 2  => ?                                           ::?::
    # 3  => Concatenate all img in directory list       ::concatAllDir::
    # 4  => 4-quadrant img mirror                       ::mirrorHV4AllImg::
    # 5  => 4-quadrant temporal dly img mirror          ::mirrorTemporalHV4AllImg::
    # 6  => interlace 2 directories
    
    # post processing
    cntrlLoadX = 0
    #cntrlScaleAll = 0
    srcSeqProcess = 0
    cntrlEYE = 0
    preProcess = 0
    postProcess = 6


else:
    
    print('All processing Bypassed!')


if cntrlLoadX == 1:
    # jpgSrcDir: set the directory path used for processing .jpg files
    # jpgSrcDir = eyeDir+'odmk17_1280x720/'
    jpgSrcDir = eyeDir+'totwnMCkeisolomon1280x720/'
    #jpgSrcDir = eyeDir+'bloodWizardMeefeep1080/'


#if cntrlScaleAll == 1:
#    high = 0    # "head crop" cntrl - <0> => center crop, <1> crop from img top
#    # *** Define Dir & File name for Scaling
#
#    scaleFileName = 'odmk17_1280x720'
#
#    scaleOutDirName = 'odmk17_1280x720_convert/'    
    
    
if srcSeqProcess == 1:
    srcDir = eyeDir+'bloodWizardMeefeepFX/'
    
    
    
if srcSeqProcess == 2:
    srcDir1 = eyeDir+'bloodWizardMSol_111/'
    srcDir2 = eyeDir+'bloodWizardMSol_111/'
    
    srcDirList = [srcDir1, srcDir2]

   
if srcSeqProcess == 5:
    srcDir1 = eyeDir+'greenInfernoSrc/'
    srcDir2 = eyeDir+'bollywoodGnk1/'
    srcDir3 = eyeDir+'osamaBinTrap/'
    srcDir4 = eyeDir+'demiLevato/'
    srcDir5 = eyeDir+'pulgasari/'
    srcDir6 = eyeDir+'spiceIndicatorGnk/'
    srcDir7 = eyeDir+'streetTrash/' 

    srcDirList = [srcDir1, srcDir2, srcDir3, srcDir4, srcDir5, srcDir6, srcDir7]

#    srcDir1 = eyeDir+'keisolomon1280x720/'
#    srcDir2 = eyeDir+'keisolomon1280x720/'
#    srcDirList = [srcDir1, srcDir2]
   
   
   
if cntrlEYE != 0:  
    
    # eyeOutFileName: file name used for output processed .jpg (appended with 000X)
    #eyeOutFileName = 'eyeEchoGorgulan1v'
    #eyeOutFileName = 'missiledeathcultTelescope'
    #eyeOutFileName = 'bananaslothParascope'
    #eyeOutFileName = 'heavyMetalFX_1560x_'
    eyeOutFileName = 'totwnSurvival'    
    
    # eyeOutDirName: set the directory path used for output processed .jpg
    #eyeOutDirName = 'eyeEchoGorgulan1vDir/'
    #eyeOutDirName = 'missiledeathcultTelescopeDir/'
    #eyeOutDirName = 'bananaslothParascopeDir/'
    #eyeOutDirName = 'heavyMetalFX_1560x1560_X3/'
    eyeOutDirName = 'totwnSurvival_136/'    



# *---------------------------------------------------------------------------*
# *---------------------------------------------------------------------------*


# <<<selects image pre-processing function>>>
# *---------------------------------------------------------------------------*
# 0  => Bypass
# 1  => Image Random Select Algorithm              ::odmkImgRndSel::
# postProcess = 0

if preProcess == 1:
    # repeatReverseAllImg:
    n = 2                         # number of repetitions (including 1st copy)
    srcDir = 'odmkExternal_Y3_129bpm\\'       # source image directory 
    eyeOutDirName = 'baliMaskYoukaiDirRpt/'    # output image directory 
    eyeOutFileName = 'baliMaskYoukai'         # output image name 



elif preProcess == 2:
    high = 1    # "head crop" cntrl - <0> => center crop, <1> crop from img top
    # *** Define Dir & File name for Scaling
    
    scaleSrcDir = 'odmkSeijinNextSrc\\'       # source image directory
    
    scaleSzX = 1560
    scaleSzY = 1560

    scaleFileName = 'odmkSeijinNext_x'
    scaleOutDirName = 'odmkSeijinNextSrc1560x1560\\'


elif preProcess == 3:
    # ***convert all images from jpg to bmp
    
    jpgSrcDir = 'bloodWizardMeefeep1080z\\'       # source image directory

    bmpOutFileName = 'bloodWizardMeefeep1080z_bmp'
    bmpOutDirName = 'bloodWizardMeefeep1080z_bmp\\'

# *---------------------------------------------------------------------------*
# *---------------------------------------------------------------------------*


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


elif postProcess == 3:
    # concatAllDir
    loc1 = eyeDir+'bloodWizardMFXx\\'
    loc2 = eyeDir+'bloodWizardMFX\\'
#    loc3 = eyeDir+'fxMoshBeast_X3_MIR1\\'
#    loc4 = eyeDir+'fxMoshBeast_X4_MIR1\\'
#    loc5 = eyeDir+'astroCratorJuice_Y5/'
#    loc6 = eyeDir+'astroCratorJuice_Y6/'
#    loc7 = eyeDir+'bSlothCultMirrorHV4_Z7_135bpm/'
#    loc8 = eyeDir+'bSlothCultMirrorHV4_Z8_135bpm/'
#    loc9 = eyeDir+'bSlothCultMirrorHV4_Z9_135bpm/'
#    loc10 = eyeDir+'bSlothCultMirrorHV4_Z10_135bpm/'    
    
    #loc6 = eyeDir+'odmkFxionDir_Z6/'
#    loc6 = eyeDir+'bananaslothParascopeDir_6/'
#    loc7 = eyeDir+'bananaslothParascopeDir_7/'
#    loc8 = eyeDir+'bananaslothParascopeDir_8/'    
    
    #concatDirList = [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8, loc9, loc10]
    #concatDirList = [loc1, loc2]
    concatDirList = [loc1, loc2]
   
    concatReName = 'bloodWizardMeefeepFX'
    concatOutDir = 'bloodWizardMeefeepFX\\'


elif postProcess == 4:

    ctrlQ = 'LR'    # defines the root quadrant to mirror from: {'UL', 'LL', 'UR', 'LR'}
    mirrorHV4SrcDir = 'heavyMetalFX_1560x1560_X3/'
    mirrorHV4ReName = 'heavyMetalFX_X3_MIR'
    mirrorHV4OutDir = 'heavyMetalFX_X3_MIR/'


elif postProcess == 5:

    ctrlQ = 'LR'    # defines the root quadrant to mirror from: {'UL', 'LL', 'UR', 'LR'}
    ctrlDly = 42
    mirrorTemporalHV4SrcDir = 'fxMoshBeast_II/'
    mirrorTemporalHV4ReName = 'fxMoshBeast_X3_MIR1'
    mirrorTemporalHV4OutDir = 'fxMoshBeast_X3_MIR1/'
    
    
elif postProcess == 6:
    imgInterlaceSrcDir1 = 'totwnMissileCult/'
    imgInterlaceSrcDir2 = 'keisolomon1280x720/'
    
    imgInterlaceOutDir = 'totwnMCkeisolomon1280x720/'
    
    imgInterlaceReName = 'totwnMCkeisolomon'
    

elif postProcess == 7:
    imgXfadeSrcDir1 = 'vogueGnkDncX01/'
    imgXfadeSrcDir2 = 'gnkdButterfly/'
    
    imgXfadeOutDir = 'gnkdVogueMetaB/'
    
    imgXfadeReName = 'gnkdVogueMetaB'
    
    
elif postProcess == 8:
    imgScanSrcDir1 = 'moshOlly2017_src/'
    imgScanSrcDir2 = 'vogueGnkDncX01/'
    imgScanSrcDir3 = 'gnkdButterfly/'
    imgScanSrcDir3 = 'odmkExternal_Y3_129bpm/'
    
    
    imgScanOutDir = 'odmkScanDir_test/'
    
    imgScanReName = 'odmkScanDir_test'    
    

    
# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Instantiate objects::---*')
print('// *--------------------------------------------------------------* //') 


eyeTb = eye.odmkEYEu(xLength, bpm, timeSig, SzX, SzY, imgFormat, framesPerSec=framesPerSec, fs=fs, cntrlEYE=cntrlEYE, rootDir='None')
print('\nAn odmkEYEu object has been instanced with:')
print('xLength = '+str(xLength)+'; bpm = '+str(bpm)+' ; timeSig = '+str(timeSig)+' ; framesPerSec = '+str(framesPerSec)+' ; fs = '+str(fs))


odmkvTb = odmkv.odmkVIDEOu(xLength, bpm, timeSig, SzX, SzY, imgFormat, framesPerSec=framesPerSec, fs=fs, cntrlEYE=cntrlEYE, rootDir='None')
print('\nAn odmkVIDEOu object has been instanced with:')
print('xLength = '+str(xLength)+'; bpm = '+str(bpm)+' ; timeSig = '+str(timeSig)+' ; framesPerSec = '+str(framesPerSec)+' ; fs = '+str(fs))

eyeClks = clks.odmkClocks(xLength, fs, bpm, framesPerSec)
print('\nAn odmkEYEu object has been instanced with:')
print('xLength = '+str(xLength)+'; fs = '+str(fs)+'; bpm = '+str(bpm)+' ; framesPerSec = '+str(framesPerSec))


framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



# <<< :: Begin Function Calls :: >>>


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


if srcSeqProcess == 1:

    imgSeqArray = sorted(glob.glob(srcDir+'*'))



if srcSeqProcess == 2:
    
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKVIDEO - Image source sequencer - 1 Dir::---*')
    print('// *--------------------------------------------------------------* //')

    # generate control signal
    secondsPerBeat = eyeClks.spb

    fsLfo = 500.0
    tbLFOGen = wavGen.odmkWavGen1(xLength, fsLfo)


    shape = 1
    freqCtrl = framesPerBeat / (framesPerSec/2)
    phaseCtrl = 0
    
    # TEMP for 6 ( +/-3 ) directories -> need to adapt to function
    #xCtrl = tbLFOGen.odmkWTOsc1(numFrames, shape, freqCtrl, phaseCtrl, quant=3)+3
    xCtrl = tbLFOGen.odmkWTOsc1(numFrames, shape, freqCtrl, phaseCtrl, quant=1)

    #pdb.set_trace()

    imgSeqArray = eyeTb.imgScanDir(srcDirList, numFrames, xCtrl)

    print('\nCreated python lists:')
    print('<<imgSrcArray>> (image data object list)')
    print('<<imgSrcNmArray>> (image data object names)\n')


print('// *--------------------------------------------------------------* //')


if srcSeqProcess == 5:
    
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image source sequencer - imgScanDir::---*')
    print('// *--------------------------------------------------------------* //')

    # generate control signal
    secondsPerBeat = eyeClks.spb

    fsLfo = 500.0
    tbLFOGen = wavGen.odmkWavGen1(xLength, fsLfo)


    shape = 1
    freqCtrl = framesPerBeat / (framesPerSec/2)
    phaseCtrl = 0
    
    xCtrl = tbLFOGen.odmkWTOsc1(numFrames, shape, freqCtrl, phaseCtrl, quant=3)+3
    #xCtrl = tbLFOGen.odmkWTOsc1(numFrames, shape, freqCtrl, phaseCtrl, quant=1)

    #pdb.set_trace()

    imgSeqArray = eyeTb.imgScanDir(srcDirList, numFrames, xCtrl)

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

#if cntrlScaleAll == 1:
#
#    print('\n')
#    print('// *--------------------------------------------------------------* //')
#    print('// *---::Scale all imgages in img obj array::---*')
#    print('// *--------------------------------------------------------------* //')
#
#    # define output directory for scaled img
#    # Dir path must end with backslash /
#    processScaleDir = eyeDir+scaleOutDirName
#    os.makedirs(processScaleDir, exist_ok=True)
#    
#    # individual file root name
#    outNameRoot = scaleFileName    
#    
#
#    [processScaledArray, processScaledNmArray] = eyeTb.odmkScaleAll(imgSrcArray, SzX, SzY, w=1, high=0, outDir=processScaleDir, outName=outNameRoot)
#
#    print('\nCreated python lists:')
#    print('<<processScaledArray>> (img data objects) and <<processScaledNmArray>> (img names)\n')
#
#    print('Saved Scaled images to the following location:')
#    print(processScaleDir)
#    print('\n')
#
#else:    # Bypass
#    pass

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


if cntrlEYE == 100:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Linear Select Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames:
    # randomly select an image from the source dir, hold for h frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgLinSelNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgLinSeldir = eyeDir+eyeOutDirName
    os.makedirs(imgLinSeldir, exist_ok=True)  # If Dir does not exist, makedir

    eyeTb.odmkImgLinSel(imgSrcArray, numFrames, imgLinSeldir, imgOutNm=imgLinSelNm)

    print('\nodmkImgRndSel function output => Created python lists:')
    print('<<<imgRndSelOutA>>>   (ndimage objects)')
    print('<<<imgRndSelNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(imgLinSeldir)

    print('// *--------------------------------------------------------------* //')

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


if cntrlEYE == 101:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Linear Solarize Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames:
    # randomly select an image from the source dir, hold for h frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgLinSolNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgLinSoldir = eyeDir+eyeOutDirName
    os.makedirs(imgLinSoldir, exist_ok=True)  # If Dir does not exist, makedir
    
    xfadeFrames = int(np.ceil(eyeClks.framesPerBeat))

    eyeTb.odmkImgLinSol(imgSeqArray, numFrames, xfadeFrames, imgLinSoldir, imgOutNm=imgLinSolNm)

    print('\nodmkImgRndSel function output => Created python lists:')
    print('<<<imgRndSelOutA>>>   (ndimage objects)')
    print('<<<imgRndSelNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(imgLinSoldir)

    print('// *--------------------------------------------------------------* //')

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //



if cntrlEYE == 102:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Rotate Linear Select Algorithm::---*')
    print('// *--------------------------------------------------------------* //')

    # for n frames:
    # randomly select an image from the source dir, hold for h frames

    # eyeOutFileName = 'myOutputFileName'    # defined above
    imgLinSelNm = eyeOutFileName
    # output dir where processed img files are stored:
    # eyeOutDirName = 'myOutputDirName'    # defined above
    imgLinSeldir = eyeDir+eyeOutDirName
    os.makedirs(imgLinSeldir, exist_ok=True)  # If Dir does not exist, makedir

    xfadeFrames = int(np.ceil(eyeClks.framesPerBeat))

    eyeTb.odmkImgRotLinSel(imgSeqArray, numFrames, xfadeFrames, imgLinSeldir, imgOutNm=imgLinSelNm)

    print('\nodmkImgRndSel function output => Created python lists:')
    print('<<<imgRndSelOutA>>>   (ndimage objects)')
    print('<<<imgRndSelNmA>>> (img names)')

    print('\noutput processed img to the following directory:')
    print(imgLinSeldir)

    print('// *--------------------------------------------------------------* //')

# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //

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

    eyeTb.odmkImgRndSel(imgSrcArray, numFrames, imgRndSeldir, imgOutNm=imgRndSelNm)

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


    eyeTb.odmkImgRndBpm(imgSrcArray, numFrames, eyeDFrames, imgRndBpmdir, imgOutNm=imgRndBpmNm)

    print('// *--------------------------------------------------------------* //')


# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


elif cntrlEYE == 202:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - Image Random Select LFO Algorithm::---*')
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


    eyeTb.odmkImgRndLfo(imgSrcArray, ctrlLfo, eyeDFrames, imgRndBpmdir, imgOutNm=imgRndBpmNm)

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
    #SzX = SzX
    #SzY = SzY

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
            
            newDimX -= hop_sz
            if newDimX < 2:
                newDimX = 2
            newDimY -= hop_sz   
            if newDimY < 2:
                newDimY = 2
                
#            if newDimX > 2:
#                newDimX -= hop_sz
#            if newDimY > 2:
#                newDimY -= hop_sz
            
            # scale image to new dimensions
            imgItr = eyeTb.odmkEyeRescale(imgClone, newDimX, newDimY)
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


elif cntrlEYE == 5:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE odmkImgRotateAlt::---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    imgRotateAltNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    imgRotateAltDir = eyeDir+eyeOutDirName
    os.makedirs(imgRotateAltDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength
    
    #pdb.set_trace()

    imgSrcArray = []
    imgSrcArray.extend(sorted(glob.glob(imgSeqArray+'*')))

    eyeTb.odmkImgRotateAlt(imgSrcArray, sLength, framesPerSec, xfadeFrames, imgRotateAltDir, imgOutNm=imgRotateAltNm)

    print('\nProcessed images output to the following directory:')
    print(imgRotateAltDir)


# *****FUNCTIONALIZEIT*****

#elif cntrlEYE == 5:
#
#    # dir where processed img files are stored:
#    gzdir = 'C:/usr/eschei/odmkPython/odmk/eye/imgSrc/exp1/'
#
#    eye_name = 'odmkRotFlash1'
#
#    # num_gz = 192
#    num_gz = 256
#    zn = cyclicZn(num_gz)
#
#    gz1NmArray = []
#    gz1Array = []
#
#    nextInc = 0
#    for i in range(num_gz):
#        zr = ''
#        ang = (atan2(zn[i].imag, zn[i].real))*180/np.pi
#        if ((i % 4) == 1 or (i % 4) == 2):
#            rotate_gz1 = ndimage.rotate(gz1, ang, reshape=False)
#        else:
#            rotate_gz1 = ndimage.rotate(gzRESIZEx02, ang, reshape=False)
#        nextInc += 1
#        # Find num digits required to represent max index
#        n_digits = int(ceil(np.log10(num_gz))) + 2
#        for j in range(n_digits - len(str(nextInc))):
#            zr += '0'
#        strInc = zr+str(nextInc)
#        gz1Nm = gzdir+eye_name+strInc+'.jpg'
#        misc.imsave(gz1Nm, rotate_gz1)
#        gz1NmArray.append(gz1Nm)
#        gz1Array.append(rotate_gz1)


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

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 6)) #2 images for 2 frames
    xfadeFrames = int(np.ceil(eyeClks.framesPerBeat))
    sLength = xLength
    
    eyeTb.odmkImgXfade(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

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

    eyeTb.odmkImgXfadeAll(imgSrcArray, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

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
    
    sLength = xLength

    #framesPerBeat = int(np.ceil(eyeClks.framesPerBeat))

    eyeTb.odmkImgSolXfade(imgSrcArray, sLength, framesPerSec, framesPerBeat, outSolXfadeDir, imgOutNm=outSolXfadeNm)

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

    xFrames = int(ceil(xLength * framesPerSec) / 4)   # 1 fade whole length

    eyeTb.odmkImgXfadeRot(imgSrcArray, xLength, framesPerSec, xFrames, outXfadeRotDir, imgOutNm=outXfadeRotNm, rotDirection=rotD)

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
        

    eyeTb.odmkImgTelescope(imgSrcArray, xLength, fps, fpb, imgTelescOutDir, imgOutNm=imgTelescNm, inOut=inOrOut)

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

    eyeTb.odmkImgXfadeTelescope(processScaledArray, fpb, imgXfadeTelOutDir, inOut=inOrOut, imgOutNm=imgXfadeTelNm)

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

    eyeTb.odmkImgDivXfadeTelescope(processScaledArray, fpb, imgDivXfadeTelOutDir, inOut=inOrOut, imgOutNm=imgDivXfadeTelNm)

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
    
    numECHOES = 16
    stepECHOES = 5          # 1 or greater
    decayECHOES = 'None'

    # final video length in seconds
    numFrames = int(np.ceil(xLength * framesPerSec))

    # [eyeECHO_Bpm_Out, eyeECHO_Bpm_Nm] =  odmkEyeEchoBpm(imgSrcArray, numFrames, eyeDFrames, numECHOES, stepECHOES, imgRndBpmdir, echoDecay = decayECHOES, imgOutNm=imgRndBpmNm)
    eyeTb.odmkEyeEchoBpm(imgSrcArray, numFrames, eyeDFrames, numECHOES, stepECHOES, imgRndBpmdir, echoDecay = decayECHOES, imgOutNm=imgRndBpmNm)
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
    
    eyeTb.odmkImgColorEnhance(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm)

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
    
    eyeTb.odmkImgCEnTelescope(imgSrcArray, sLength, framesPerSec, xfadeFrames, outXfadeDir, imgOutNm=outXfadeNm, inOut=inOrOut)

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
    outPxlRndRepNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outPxlRndRepDir = eyeDir+eyeOutDirName
    os.makedirs(outPxlRndRepDir, exist_ok=True)


    sLength = xLength

    
    eyeTb.odmkPxlRndReplace(imgSrcArray, sLength, framesPerSec, outPxlRndRepDir, imgOutNm=outPxlRndRepNm)


    print('\noutput processed img to the following directory:')
    print(outPxlRndRepDir)


    print('// *--------------------------------------------------------------* //')
    

if cntrlEYE == 222:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Pixel Random Rotate::---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    outPxlRndRotNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    outPxlRndRotDir = eyeDir+eyeOutDirName
    os.makedirs(outPxlRndRotDir, exist_ok=True)


    sLength = xLength

    
    eyeTb.odmkPxlRndRotate(imgSrcArray, sLength, framesPerSec, framesPerBeat, outPxlRndRotDir, imgOutNm=outPxlRndRotNm)


    print('\noutput processed img to the following directory:')
    print(outPxlRndRotDir)


    print('// *--------------------------------------------------------------* //')


if cntrlEYE == 23:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE LFO Modulated Parascope f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    lfoParascopeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    lfoParascopeDir = eyeDir+eyeOutDirName
    os.makedirs(lfoParascopeDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    inOrOut = 0     # telescope direction: 0 = in, 1 = out
    
    eyeTb.odmkLfoParascope(imgSrcArray, sLength, framesPerSec, xfadeFrames, lfoParascopeDir, imgOutNm=lfoParascopeNm, inOrOut=inOrOut)

    print('\nProcessed images output to the following directory:')
    print(lfoParascopeDir)

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

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    eyeTb.odmkBSlothRecurs(imgSrcArray, sLength, framesPerSec, xfadeFrames, bSlothRecursDir, imgOutNm=bSlothRecursNm)

    print('\nProcessed images output to the following directory:')
    print(bSlothRecursDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')
  

if cntrlEYE == 25:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Bananasloth Glitch 2 f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    bSlothGlitch1Nm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    bSlothGlitch1Dir = eyeDir+eyeOutDirName
    os.makedirs(bSlothGlitch1Dir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    eyeTb.odmkBSlothGlitch2(imgSrcArray, sLength, framesPerSec, xfadeFrames, bSlothGlitch1Dir, imgOutNm=bSlothGlitch1Nm)

    print('\nProcessed images output to the following directory:')
    print(bSlothGlitch1Dir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')
  
    
if cntrlEYE == 26:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Bananasloth Glitch 1 f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    bSlothGlitch1Nm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    bSlothGlitch1Dir = eyeDir+eyeOutDirName
    os.makedirs(bSlothGlitch1Dir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    eyeTb.odmkBSlothGlitch1(imgSrcArray, sLength, framesPerSec, xfadeFrames, bSlothGlitch1Dir, imgOutNm=bSlothGlitch1Nm)

    print('\nProcessed images output to the following directory:')
    print(bSlothGlitch1Dir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')   
 

if cntrlEYE == 27:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE Bananasloth Glitch 3 f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    bSlothGlitch1Nm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    bSlothGlitch1Dir = eyeDir+eyeOutDirName
    os.makedirs(bSlothGlitch1Dir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    eyeTb.odmkBSlothGlitch3(imgSrcArray, sLength, framesPerSec, xfadeFrames, bSlothGlitch1Dir, imgOutNm=bSlothGlitch1Nm)

    print('\nProcessed images output to the following directory:')
    print(bSlothGlitch1Dir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')
   

if cntrlEYE == 623:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE VideoGoreXXZ01::---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    fxVLfoParascopeNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    fxVLfoParascopeDir = eyeDir+eyeOutDirName
    os.makedirs(fxVLfoParascopeDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength
    
    #pdb.set_trace()


    odmkvTb.odmkVLfoParascope(imgSeqArray, sLength, framesPerSec, xfadeFrames, fxVLfoParascopeDir, imgOutNm=fxVLfoParascopeNm)

    print('\nProcessed images output to the following directory:')
    print(fxVLfoParascopeDir)


    print('// *--------------------------------------------------------------* //')


    

if cntrlEYE == 626:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE FxSlothCultLife Glitch::---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    fxSlothCultNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    fxSlothCultDir = eyeDir+eyeOutDirName
    os.makedirs(fxSlothCultDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength
    
    #pdb.set_trace()

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    odmkvTb.odmkFxSlothCult(imgSeqArray, sLength, framesPerSec, xfadeFrames, fxSlothCultDir, imgOutNm=fxSlothCultNm)

    print('\nProcessed images output to the following directory:')
    print(fxSlothCultDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')
    
   
   
if cntrlEYE == 627:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE VideoGoreXXZ01::---*')
    print('// *--------------------------------------------------------------* //')


    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    fxVideoGoreNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    fxVideoGoreDir = eyeDir+eyeOutDirName
    os.makedirs(fxVideoGoreDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength
    
    #pdb.set_trace()

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    odmkvTb.odmkVideoGoreXXZ01(imgSeqArray, sLength, framesPerSec, xfadeFrames, fxVideoGoreDir, imgOutNm=fxVideoGoreNm)

    print('\nProcessed images output to the following directory:')
    print(fxVideoGoreDir)

    # print('\nrepeatDir function output => Created python lists:')
    # print('<<<xfadeRepeatA>>>   (ndimage objects)')
    # print('<<<xfadeRepeatNmA>>> (img names)')

    # print('\noutput processed img to the following directory:')
    # print(eyeDir+'eyeXfadeRepeat')

    print('// *--------------------------------------------------------------* //')
   
   
if cntrlEYE == 28:
    
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - EYE LFO Horizontal Trails f::---*')
    print('// *--------------------------------------------------------------* //')

    # srcXfadeDir = eyeDir+'eyeSrcExp23/'    # defined above
    #eyeOutFileName = 'myOutputFileName'    # defined above
    lfoHtrailsNm = eyeOutFileName
    #eyeOutDirName = 'myOutputDirName'    # defined above
    lfoHtrailsDir = eyeDir+eyeOutDirName
    os.makedirs(lfoHtrailsDir, exist_ok=True)

    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat / 3)) #2 images for 2 frames
    #xfadeFrames = int(np.ceil(eyeClks.framesPerBeat)) #2 images for 2 frames
    xfadeFrames = framesPerBeat
    #xfadeFrames = xLength

    sLength = xLength

    #inOrOut = 1     # telescope direction: 0 = in, 1 = out

    eyeTb.odmkLfoHtrails(imgSrcArray, sLength, framesPerSec, xfadeFrames, lfoHtrailsDir, imgOutNm=lfoHtrailsNm)
    print('\nProcessed images output to the following directory:')
    print(lfoHtrailsDir)

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
# begin : img pre-processing
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


if preProcess == 2:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Scale All images in SRC Directory::---*')
    print('// *--------------------------------------------------------------* //')

    ctrl = high

    imgSrcArray = eyeDir+scaleSrcDir
    
    sclSzX = scaleSzX
    sclSzY = scaleSzY
    
    # define output directory for scaled img
    # Dir path must end with backslash /
    processScaleDir = eyeDir+scaleOutDirName
    os.makedirs(processScaleDir, exist_ok=True)
    
    # individual file root name
    outNameRoot = scaleFileName    
    

    eyeTb.odmkScaleAll(imgSrcArray, sclSzX, sclSzY, processScaleDir, high=ctrl, outName=outNameRoot)


    print('Saved Scaled images to the following location:')
    print(processScaleDir)
    print('\n')



elif preProcess == 3:

    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Convert All images in SRC Directory to bmp::---*')
    print('// *--------------------------------------------------------------* //')


    imgSrcDir = eyeDir+jpgSrcDir
    
    
    # define output directory for scaled img
    # Dir path must end with backslash /
    imgOutDir = eyeDir+bmpOutDirName
    os.makedirs(imgOutDir, exist_ok=True)
    
    # individual file root name
    outNameRoot = bmpOutFileName    
    

    eye.convertJPGtoBMP(imgSrcDir, imgOutDir, reName=outNameRoot)


    print('Saved converted images to the following location:')
    print(imgOutDir)
    print('\n')



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


elif postProcess == 3:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::Concatenate all images in directory list::---*')
    print('// *--------------------------------------------------------------* //')

    # ex. defined above: concatDirList = [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8]
    dirList = concatDirList
    reName = concatReName
    
    eyeConcatDir = eyeDir+concatOutDir
    os.makedirs(eyeConcatDir, exist_ok=True)
    
    
    eyeTb.concatAllDir(dirList, eyeConcatDir, reName)
    
    print('\nOutput all images to: '+eyeConcatDir)
    
    print('// *--------------------------------------------------------------* //')



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


elif postProcess == 4:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - mirrorHV4 list of files in directory::---*')
    print('// *--------------------------------------------------------------* //')


    srcDir = eyeDir+mirrorHV4SrcDir   
    
    mirrorHV4Nm = mirrorHV4ReName
    
    mirrorHV4Dir = eyeDir+mirrorHV4OutDir
    os.makedirs(mirrorHV4Dir, exist_ok=True)
    
    outSzX = 1920
    outSzY = 1080
    
    ctrlQuadrant = ctrlQ

    
    eyeTb.mirrorHV4AllImg(srcDir, outSzX, outSzY, mirrorHV4Dir, mirrorHV4Nm, ctrl=ctrlQuadrant)    
    print('\nOutput all images to: '+mirrorHV4Dir)
    
    print('// *--------------------------------------------------------------* //')




# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


elif postProcess == 5:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - mirrorTemporalHV4 list of files in directory::---*')
    print('// *--------------------------------------------------------------* //')


    srcDir = eyeDir+mirrorTemporalHV4SrcDir   
    
    mirrorTemporalHV4Nm = mirrorTemporalHV4ReName
    
    mirrorTemporalHV4Dir = eyeDir+mirrorTemporalHV4OutDir
    os.makedirs(mirrorTemporalHV4Dir, exist_ok=True)
    
    outSzX = 1920
    outSzY = 1080
    
    ctrlQuadrant = ctrlQ
    frameDly = ctrlDly

    
    eyeTb.mirrorTemporalHV4AllImg(srcDir, outSzX, outSzY, mirrorTemporalHV4Dir, mirrorTemporalHV4Nm, frameDly, ctrl=ctrlQuadrant)    
    print('\nOutput all images to: '+mirrorTemporalHV4Dir)
    
    print('// *--------------------------------------------------------------* //')



# // *********************************************************************** //
# // *********************************************************************** //
# // *********************************************************************** //


elif postProcess == 6:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - imgInterlaceDir interlace files in directories::---*')
    print('// *--------------------------------------------------------------* //')


    srcDir1 = eyeDir+imgInterlaceSrcDir1
    srcDir2 = eyeDir+imgInterlaceSrcDir2
    
    interlaceNm = imgInterlaceReName
    
    imgInterlaceDir = eyeDir+imgInterlaceOutDir
    os.makedirs(imgInterlaceDir, exist_ok=True)

    

    eyeTb.imgInterlaceDir(srcDir1, srcDir2, imgInterlaceDir, reName=interlaceNm)    
    print('\nOutput all images to: '+imgInterlaceDir)
    
    print('// *--------------------------------------------------------------* //')



elif postProcess == 7:
    print('\n')
    print('// *--------------------------------------------------------------* //')
    print('// *---::ODMKEYE - imgXfadeDir interlace files in directories::---*')
    print('// *--------------------------------------------------------------* //')


    srcDir1 = eyeDir+imgXfadeSrcDir1
    srcDir2 = eyeDir+imgXfadeSrcDir2
    
    XfadeNm = imgXfadeReName
    
    imgXfadeDir = eyeDir+imgXfadeOutDir
    os.makedirs(imgXfadeDir, exist_ok=True)


    # get length from shortest img dir
    imgFileList1 = []
    imgFileList2 = []
    
    if imgFormat=='fjpg':
        imgFileList1.extend(sorted(glob.glob(srcDir1+'*.jpg')))
        imgFileList2.extend(sorted(glob.glob(srcDir2+'*.jpg')))    
    elif imgFormat=='fbmp':
        imgFileList1.extend(sorted(glob.glob(srcDir1+'*.bmp')))
        imgFileList2.extend(sorted(glob.glob(srcDir2+'*.bmp')))     

    imgCount = 2*min(len(imgFileList1), len(imgFileList2))

    tbWavGen = wavGen.odmkWavGen1(imgCount, fs)


    secondsPerBeat = eyeClks.spb


    pwmOutFreq = 20000*secondsPerBeat
    pwmCtrlFreq = 1000*secondsPerBeat
    phaseInc = pwmOutFreq/fs

    pulseWidthCtrl = 0.5*tbWavGen.monosin(pwmCtrlFreq) + 0.5

    pwmOut = np.zeros([imgCount])

    for i in range(imgCount):
        pwmOut[i] = tbWavGen.pulseWidthMod(phaseInc, pulseWidthCtrl[i])

    eyeTb.imgXfadeDir(srcDir1, srcDir2, pwmOut, imgXfadeDir, reName=XfadeNm)    
    print('\nOutput all images to: '+imgXfadeDir)
    
    print('// *--------------------------------------------------------------* //')


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