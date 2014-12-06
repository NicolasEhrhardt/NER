# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:38:58 2014

@author: benoit
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
import tsne

if len(sys.argv) < 4:
    print 'Usage:'
    print '  python %s <saved-vocab.csv> <labels.txt> <output.png> <nb of points>' % sys.argv[0]
    exit()

_, vocab, labels, output, npoints = sys.argv
npoints = int(npoints)

X = np.loadtxt(vocab, skiprows=1)
X = X[:npoints,:]

alllabels = []
with open(labels, 'r') as l:
    for line in l:
        alllabels.append(line.strip())
cutlabels = alllabels[0:npoints]

Y = tsne.tsne(X, 2, 50, 20.0);
plt.scatter(Y[:, 0], Y[:, 1], 20);

for label, x, y in zip(cutlabels, Y[:, 0], Y[:, 1]):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.show()

plt.savefig(sys.argv[2], bbox_inches='tight')
