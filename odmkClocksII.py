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
# import matplotlib.pyplot as plt


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *********************************************************************** //
# util functions
# // *********************************************************************** //


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

#def odmkDownBeats(totalSamples, samplesPerBeat):
#    ''' generates an output array of 1s at downbeat, 0s elsewhere '''
#    
#    xClockDown = np.zeros([totalSamples, 1])
#    for i in range(totalSamples):
#        if i % np.ceil(samplesPerBeat) == 0:
#            xClockDown[i] = 1
#        else:
#            xClockDown[i] = 0
#    return xClockDown
#
#
#def odmkDownFrames(totalSamples, framesPerBeat):
#    ''' generates an output array of 1s at frames corresponding to
#        downbeats, 0s elsewhere '''
#
#    xFramesDown = np.zeros([totalSamples, 1])
#    for i in range(totalSamples):
#        if i % np.ceil(framesPerBeat) == 0:
#            xFramesDown[i] = 1
#        else:
#            xFramesDown[i] = 0
#    return xFramesDown

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


class odmkClocks:
    ''' odmk audio/video clocking modules '''

    def __init__(self, xLength, fs, bpm, framesPerSec):
        # set primary parameters from inputs:
        self.xLength = xLength
        self.fs = fs
        self.bpm = bpm
        self.framesPerSec = framesPerSec

    # set secondary parameters

    def getBps(self, bpm):
        ''' beats per second '''
        bps = bpm / 60
        return bps

    def getSpb(self, bpm):
        '''seconds per beat '''
        spb = 60.0 / bpm
        return spb

#    def getSamplesPerBeat(self, fs, bpm):
#        ''' Num of samples per beat '''
#        samplesPerBeat = fs * self.getSpb(bpm)
#        return samplesPerBeat

    def getSamplesPerFrame(self, fs, framesPerSec):
        ''' Num of audio samples per frame '''
        samplesPerFrame = framesPerSec * fs
        return samplesPerFrame

#    def getFramesPerBeat(self, bpm, framesPerSec):
#        ''' Num of frames per beat '''
#        framesPerBeat = self.getSpb(bpm) * framesPerSec
#        return framesPerBeat

    def getTotalSamples(self, xLength, fs):
        ''' Total audio samples in x '''
        totalSamples = int(np.ceil(xLength * fs))
        return totalSamples

    def getTotalFrames(self, xLength, framesPerSec):
        ''' Total video frames in x '''
        totalFrames = int(np.ceil(xLength * framesPerSec))
        return totalFrames

#    def getTotalBeats(self, xLength, fs, bpm):
#        ''' Total beats in x '''
#        totalBeats = getTotalSamples(xLength, fs) / getSamplesPerBeat(fs, bpm)
#        return totalBeats


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


def getTotalBeats(xLength, fs, bpm):
    ''' Total beats in x '''
    totalBeats = odmkClocks.getTotalSamples(xLength, fs) / odmkClocks.getSamplesPerBeat(fs, bpm)
    return totalBeats


def clkDownBeats(xLength, fs, bpm):
    ''' generates an output array of 1s at downbeat, 0s elsewhere '''

    tSamples = odmkClocks.getTotalSamples(odmkClocks, xLength, fs)
    xClockDown = np.zeros([tSamples, 1])
    for i in range(tSamples):
        if i % np.ceil(odmkClocks.getSamplesPerBeat(odmkClocks, fs, bpm)) == 0:
            xClockDown[i] = 1
        else:
            xClockDown[i] = 0
    return xClockDown


def clkDownFrames(xLength, fs, bpm, framesPerSec):
    ''' generates an output array of 1s at frames corresponding to
        downbeats, 0s elsewhere '''

    tSamples = odmkClocks.getTotalSamples(xLength, fs)
    xFramesDown = np.zeros([tSamples, 1])
    for i in range(tSamples):
        if i % np.ceil(odmkClocks.getFramesPerBeat(bpm, framesPerSec)) == 0:
            xFramesDown[i] = 1
        else:
            xFramesDown[i] = 0
    return xFramesDown
