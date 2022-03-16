# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:56:42 2021

@author: J
"""

import numpy as np

# Labels Comparison
f1 = np.fromfile('labelsBindsnetIn.csv', dtype = np.int8 ,sep='\n')
f2 = np.fromfile('labelsBindsnetOut.csv', dtype = np.int8 ,sep='\n')
f3 = np.fromfile('labelsQt.csv', dtype = np.int8 ,sep='\n')
cmp1 = np.count_nonzero(f1 == f2) / len(f1)
cmp2 = np.count_nonzero(f1 == f3) / len(f1)

print( 'Bindsnet In vs Bindsnet Out: {0:2.1f} %'.format(cmp1*100) )
print( 'Bindsnet In vs Qt: {0:2.1f} %'.format(cmp2*100) )