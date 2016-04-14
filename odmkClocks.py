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
# import scipy as sp


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

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : object definition
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

class odmkClocks:
    ''' odmk audio/video clocking modules 
        usage: myOdmkClks = odmkClocks(outLength, fs, bpm, framesPerSec)    
        ex: xDownFrames = myOdmkClks.clkDownFrames() ->
        returns an np.array of 1's at 1st frame of downbeat, 0's elsewhere '''

    def __init__(self, xLength, fs, bpm, framesPerSec):

        # *---set primary parameters from inputs---*

        self.xLength = xLength
        self.fs = fs
        self.bpm = bpm
        self.framesPerSec = framesPerSec
        print('\nAn odmkClocks object has been instanced with:')
        print('xLength = '+str(self.xLength)+'; fs = '+str(fs)+'; bpm = '+str(bpm)+'; framesPerSec = '+str(framesPerSec))

        # *---set secondary parameters---*

        # set bps - beats per second
        self.bps = bpm / 60
        # set spb - Seconds per beat
        self.spb = 60.0 / bpm
        # set samplesPerBeat
        self.samplesPerBeat = fs * self.spb
        # set samplesPerFrame - Num audio samples per video frame
        self.samplesPerFrame = framesPerSec * fs
        # set framesPerBeat - Num video frames per beat
        self.framesPerBeat = self.spb * framesPerSec
        # set totalSamples - Total audio samples in x
        self.totalSamples = int(np.ceil(xLength * fs))
        # set totalFrames - Total video frames in x
        self.totalFrames = int(np.ceil(xLength * framesPerSec))
        # set totalBeats - total beats in x
        self.totalBeats = self.totalSamples / self.samplesPerBeat

    # // ******************************************************************* //
    # sequence generators
    # // ******************************************************************* //

    # // *-----------------------------------------------------------------* //
    # gen downbeat sequence
    # // *-----------------------------------------------------------------* //

    def clkDownBeats(self):
        ''' generates an output array of 1s at downbeat, 0s elsewhere '''

        xClockDown = np.zeros([self.totalSamples, 1])
        for i in range(self.totalSamples):
            if i % np.ceil(self.samplesPerBeat) == 0:
                xClockDown[i] = 1
            else:
                xClockDown[i] = 0
        return xClockDown

    def clkDownFrames(self):
        ''' generates an output array of 1s at frames corresponding to
            downbeats, 0s elsewhere '''

        xFramesDown = np.zeros([self.totalSamples, 1])
        for i in range(self.totalSamples):
            if i % np.ceil(self.framesPerBeat) == 0:
                xFramesDown[i] = 1
            else:
                xFramesDown[i] = 0
        return xFramesDown

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : object definition
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
