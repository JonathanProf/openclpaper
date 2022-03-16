# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:56:42 2021

@author: J
"""

import numpy as np

_path = './classification/'

n_neurons = 400
time = 64

# Labels Comparison
f1 = np.fromfile( _path + 'labelsBindsnetOut' + str(n_neurons) +'N_64ms.csv', dtype = np.int8 ,sep='\n')
f2 = np.fromfile( _path + 'labelsVIM3openCL' + str(n_neurons) +'N_64ms.csv', dtype = np.int8 ,sep='\n')
f3 = np.fromfile( _path + 'labelsBindsnetIn' + str(400) +'N_64ms.csv', dtype = np.int8 ,sep='\n')
cmp2 = 0
cmp1 = 0
tam = len(f2) if len(f2) < len(f1) else len(f1)
for i in range(tam):
	if(f3[i] == f2[i]):
		cmp1+=1
		
	if(f3[i] == f1[i]):
		cmp2+=1
		
	#else:
	#	print("INDEX[{0}] - SERIAL[{1}] - PARALLEL [{2}] ".format( i, f1[i], f2[i]))
	
print("Experiments for parameters (neurons, time)= ({0},{1})".format(n_neurons,time))
print( 'labelsBindsnetIn vs. labelsVIM3Paralelo: {0:2.1f} % for {1} samples'.format(cmp1/tam*100, tam) )
print( 'labelsBindsnetIn vs. labelsBindsnetOut: {0:2.1f} % for {1} samples'.format(cmp2/tam*100, tam) )