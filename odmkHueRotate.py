# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((odmkEYEu4.py))::__

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


def rgb_to_hsv(rgb):
    ''' Translated from source of colorsys.rgb_to_hsv
      r,g,b should be a numpy arrays with values between 0 and 255
      rgb_to_hsv returns an array of floats between 0.0 and 1.0. '''
      
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    ''' Translated from source of colorsys.hsv_to_rgb
      h,s should be a numpy arrays with values between 0.0 and 1.0
      v should be a numpy array with values between 0.0 and 255.0
      hsv_to_rgb returns an array of uints between 0 and 255. '''
      
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(img,hout):
    hsv = rgb_to_hsv(img)
    hsv[...,0] = hout
    rgb = hsv_to_rgb(hsv)
    return rgb


rootDir = 'C:/Users/djoto-odmk/odmk-sci/odmk_code/odmkPython/eye/odmkSrc/'
srcDir = rootDir+'gorgulanScaled/'
outDir = rootDir+'gorgulanHueRot/'
os.makedirs(outDir, exist_ok=True)

eyeHueNm = 'eyeRotateHue'

# Load input image:
gorgulan11 = misc.imread(srcDir+'keiLHC.jpeg')

numImg = 146

hRotVal = np.linspace(0, 6*np.pi, numImg)
hRotVal = np.sin(hRotVal)


#green_hue = (180-78)/360.0
#red_hue = (180-180)/360.0
#rot_hue = (50)/360.0



#gorgulanRed = shift_hue(gorgulan11, rot_hue)

#eyeHueFull = outDir+'eyeHue.jpg'
#misc.imsave(eyeHueFull, gorgulanRed)




n_digits = int(ceil(np.log10(numImg))) + 2
nextInc = 0
for i in range(numImg):
    
    gorgulanShift = shift_hue(gorgulan11, hRotVal[i])    
    
    nextInc += 1
    zr = ''
    for j in range(n_digits - len(str(nextInc))):
        zr += '0'
    strInc = zr+str(nextInc)
    eyeHueIdxNm = eyeHueNm+strInc+'.jpg'
    eyeHueFull = outDir+eyeHueIdxNm
    misc.imsave(eyeHueFull, gorgulanShift)

