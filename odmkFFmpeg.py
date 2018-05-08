# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((odmkFFmpeg.py))::__

# Python ODMK img processing research
# ffmpeg experimental

# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os, sys
from math import ceil
import numpy as np

# from subprocess import check_output, check_call
from subprocess import check_call

rootDir = 'C:\\odmkDev\\odmkCode\\odmkPython\\'
audioScrDir = 'C:\\odmkDev\\odmkCode\\odmkPython\\audio\\wavsrc\\'
audioOutDir = 'C:\\odmkDev\\odmkCode\\odmkPython\\audio\\wavout\\'

#sys.path.insert(0, 'C:/odmkDev/odmkCode/odmkPython/util')
sys.path.insert(0, rootDir+'util')
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

# eyeDir = 'C:/Users/djoto-odmk/odmk-sci/odmk_code/odmkPython/eye/odmkSrc/'

# ***** SET DIR *****
#eyeSrcDirNm = 'bananaslothParascopeDir\\'
#eyeSrcDirNm = 'pxlRndRotMirrorHV4_186bpm\\'
#eyeSrcDirNm = 'odmkPxlRndRot_N17_186\\'
eyeSrcDirNm = 'totwnSurvival_136\\'


# set format of source img { fjpg, fbmp }

eyeFormat = 'fjpg'




# *** don't change ***
eyeDir = rootDir+'eye\\eyeSrc\\'
eyeSrcDir = eyeDir+eyeSrcDirNm

#earSrcNm = 'dsvco.wav'
#earSrcNm = 'detectiveOctoSpace_one.wav'
#earSrcNm = 'noxiousTrollGas.wav'
earSrcNm = 'centipedeMeat_dmmx01.wav'

earSrcDir = audioScrDir
earSrc = earSrcDir+earSrcNm

#pdb.set_trace()


print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Search for .jpg images in eyeSrcDir directory::---*')
print('// *--------------------------------------------------------------* //')

# generate list all files in a directory
imgSrcList = []
try:
    for filename in os.listdir(eyeSrcDir):
        if eyeFormat=='fjpg':        
            if filename.endswith('.jpg'):
                eyeFormat = "fjpg"
                imgSrcList.append(filename)
        elif eyeFormat=='fbmp':
            if filename.endswith('.bmp'):
                eyeFormat = "fbmp"
                imgSrcList.append(filename)


    if imgSrcList == []:
        print('\nError: No .jpg files found in directory\n')
    else:
        imgCount = len(imgSrcList)
        print('\nFound '+str(imgCount)+' images in the eyeSrcDir directory!')
        eye_name = imgSrcList[0]
        eye_name = eye_name.split("00")[0]
        print('\nSet eye_name = "'+eye_name+'"')
        print('\nCreated numpy array: <<imgSrcList>>\n')
except OSError:
    print('\nError: directory:\n'+eyeSrcDir+'\ncannot be found\n')

print('// *--------------------------------------------------------------* //')

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Set earSrcDir directory::---*')
print('// *--------------------------------------------------------------* //')


# earSrc = 'C:/usr/eschei/odmkPython/odmk/audio/wavSrc/tonzuraTBE01.wav'


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : eye generator - ffmpeg function calls
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#movie_name = eye_name+'.mpg'
movie_name = eye_name+'.mp4'
print('\nSet movie_name = '+movie_name)
print('\nRunning ffmpeg... '+eye_name)

overwrite = True
n_interpolation_frames = 1
frame_rate = 30
bitrate = '5000k'
codec = 'h264'

# Find num digits required to represent max index
n_digits = int(ceil(np.log10(imgCount))) + 2

print('\nn_digits... '+str(n_digits))


if eyeFormat=='fjpg':
    fmt_str = eye_name+'%'+str(n_digits)+'d.jpg'
elif eyeFormat=='fbmp':   
    fmt_str = eye_name+'%'+str(n_digits)+'d.bmp'


print('\nfmt_str... '+fmt_str)

#movie_cmd = ["ffmpeg",
#             '-an',  # no sound!
#             '-r',  '%d' % frame_rate,
#             '-i', os.path.join(eyeSrcDir, fmt_str),
#             '-y' if overwrite else '-n',
#             # '-vcodec', codec,
#             '-b:v', bitrate,
#             movie_name]
             
#movie_cmd = ["ffmpeg",
#             '-r',  '%d' % frame_rate,
#             '-i', os.path.join(eyeSrcDir, fmt_str),
#             '-i', earSrc,
#             '-shortest',
#             '-y' if overwrite else '-n',
#             # '-vcodec', codec,
#             '-b:v', bitrate,
#             movie_name]


#ffmpeg -i input -c:v libx264 -preset slow -crf 22 -c:a copy output.mkv
#ffmpeg -i input -c:v libx265 -preset medium -crf 28 -c:a aac -b:a 128k output.mp4
movie_cmd = ["ffmpeg",
             '-i', os.path.join(eyeSrcDir, fmt_str),
             '-i', earSrc,
             '-c:v', 'libx265',
             '-preset', 'medium',
             '-crf', '28',
             '-y' if overwrite else '-n',
             movie_name]


check_call(movie_cmd)


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : eye generator - ffmpeg function calls
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //


# video frame-grabbing
# ffmpeg -i file.mp4 -r 1/1 $filename%06d.bmp
#ffmpeg -i test.mov -ss 00:00:09 -t 00:00:03 $filename%06d.bmp


#ffmpeg -i inputMov.whatever -ss starttime -t duration outputMov.whatever
#ffmpeg -i test.mov -ss 00:00:09 -t 00:00:03 test-out.mov

# example:
# ffmpeg -i olly2017.mp4 -ss 00:02:03 -t 00:00:27 moshOlly%06d.bmp



# mp3 encoding
# ffmpeg -i oto.wav -ab 320k -f mp3 newfile.mp3

# flac encoding
# ffmpeg -i oto.wav newfile.flac



#Caption length text: 2,200 characters Max
#Video aspect ratio: Landscape (1.91:1), Square (1:1), Vertical (4:5)
#Minimum resolution: 600 x 315 pixels (1.91:1 landscape) / 600 x 600 pixels (1:1 square) / 600 x 750 pixels (4:5 vertical)
#Minimum length: No minimum
#Maximum length: 60 seconds
#File type: Full list of supported file formats
#Supported video codecs: H.264, VP8
#Supported audio codecs: AAC, Vorbis
#Maximum size: 4GB
#Frame rate: 30fps max
