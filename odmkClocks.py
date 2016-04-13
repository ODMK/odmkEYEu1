# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((odmkClocks.py))::__

# Python ODMK timing sequencer module

# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************


import os
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


# // *********************************************************************** //
# util functions
# // *********************************************************************** //

import os
import numpy as np


def cyclicZn(n):
    ''' calculates the Zn roots of unity '''
    cZn = np.zeros((n, 1))*(0+0j)    # column vector of zero complex values
    for k in range(n):
        # z(k) = e^(((k)*2*pi*1j)/n)                         # Define cyclic group Zn points
        cZn[k] = cos(((k)*2*pi)/n) + sin(((k)*2*pi)/n)*1j    # Euler's identity

    return cZn


def randomIdx(n, k):
    '''for an array of k elements, returns a list of random indexes
       of length n (n integers rangin from 0:k-1)'''
    randIdx = []
    for i in range(n):
        randIdx.append(round(random.random()*(k-1)))
    return randIdx


# // *********************************************************************** //
# sequence generators
# // *********************************************************************** //

# // *---------------------------------------------------------------------* //
# gen downbeat sequence
# // *---------------------------------------------------------------------* //

def odmkDownBeats(totalSamples, samplesPerBeat):
    ''' generates an output array of 1s at downbeat, 0s elsewhere '''
    
    xClockDown = np.zeros([totalSamples, 1])
    for i in range(totalSamples):
        if i % np.ceil(samplesPerBeat) == 0:
            xClockDown[i] = 1
        else:
            xClockDown[i] = 0
    return xClockDown


def odmkDownFrames(totalSamples, framesPerBeat):
    ''' generates an output array of 1s at frames corresponding to
        downbeats, 0s elsewhere '''

    xFramesDown = np.zeros([totalSamples, 1])
    for i in range(totalSamples):
        if i % np.ceil(framesPerBeat) == 0:
            xFramesDown[i] = 1
        else:
            xFramesDown[i] = 0
    return xFramesDown

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : parameter definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //
# Primary parameters
# // *---------------------------------------------------------------------* //


# length of x in seconds:
xLength = 1

# audio sample rate:
fs = 44100.0

# audio sample bit width
bWidth = 24

# video frames per second:
framesPerSec = 30.0

bpm = 133.0

# time signature: 0 = 4/4; 1 = 3/4
timeSig = 0




# // *---------------------------------------------------------------------* //
# 2nd parameters
# // *---------------------------------------------------------------------* //

# beats per second
bps = bpm / 60
# seconds per beat
spb = 60.0 / bpm

# Num of samples per beat
samplesPerBeat = fs * spb 

# Num of audio samples per frame:
samplesPerFrame = framesPerSec * fs

# Num of frames per beat
framesPerBeat = spb * framesPerSec

# Total audio samples in x:
totalSamples = int(np.ceil(xLength * fs))

# Total video frames in x:
totalFrames = int(np.ceil(xLength * framesPerSec))

# Total beats in x:
totalBeats = totalSamples / samplesPerBeat


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : parameter definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : generate sequence
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //
# gen downbeat sequence
# // *---------------------------------------------------------------------* //


downBeats = odmkDownBeats(totalSamples, samplesPerBeat)

downFrames = odmkDownFrames(totalSamples, framesPerBeat)

