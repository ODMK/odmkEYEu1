# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((odmkRndtest.py))::__
#
# odmk tests using pseudo-random number generation, weighted probabilities, etc
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
from math import atan2, floor, ceil
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt

from odmkClear import *


# // *---------------------------------------------------------------------* //

clear_all()

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def quarterCos(n):
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

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //

print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMK Random Number Exp::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')


print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Check function odmkWeightRnd::---*')
print('// *--------------------------------------------------------------* //')

# w = {'A': 1.0, 'B': 1.0, 'C': 18.0}
w = {'A': 0.5, 'B': 0.5}

wr = odmkWeightRnd(w)

# results = {'A': 0, 'B': 0, 'C': 0}
results = {'A': 0, 'B': 0}
for i in range(10000):
    results[wr.random()] += 1

print('\nafter 10000 iterations the distribution is:')
print(results)

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::Generate numpy array of weighted random numbers::---*')
print('// *--------------------------------------------------------------* //')

N = 777

probArray = quarterCos(N)

fig1 = plt.figure(num=1, facecolor='silver', edgecolor='k')
plt.plot(probArray)


results = {0: 0, 1: 0}
xfadeWeight = np.array([])
for j in range(N):
    xsel = {0: probArray[j], 1: 1-probArray[j]}
    rndWeights = odmkWeightRnd(xsel)
    results[rndWeights.random()] += 1
    xfadeWeight = np.append(xfadeWeight, rndWeights.random())

print('\nafter '+str(N)+' iterations the distribution is:')
print(results)

fig2 = plt.figure(num=2, facecolor='silver', edgecolor='k')
plt.plot(xfadeWeight)

# // *---------------------------------------------------------------------* //

#plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //




























    