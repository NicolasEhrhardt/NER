# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:38:58 2014

@author: benoit
"""
import numpy as np
import pylab as pl
import sys
import tsne

if len(sys.argv) < 4:
    print 'Usage:'
    print '  python %s <saved-vocab.csv> <output.png> <nb of points>' % sys.argv[0]
    exit()

X = np.loadtxt(sys.argv[1], skiprows=1)
X = X[:int(sys.argv[3]),:]
Y = tsne.tsne(X, 2, 50, 20.0);
pl.scatter(Y[:,0], Y[:,1], 20);
pl.savefig(sys.argv[2], bbox_inches='tight')