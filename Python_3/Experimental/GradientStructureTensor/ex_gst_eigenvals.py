#!/usr/bin/python
#
# Eigenvalues of the Gradient Structure Tensor
#
# Calculates the eigenvalues of the gradient structure tensor.
# 
# The eigenvalues are numbered in decreasing order of their magnitude.
#
import sys,os
import numpy as np
from scipy.ndimage import gaussian_filter
from numba import jit,double

#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa

#
# These are the attribute parameters
#
xa.params = {
	'Inputs': ['In-line gradient', 'Cross-line gradient', 'Z gradient'],
	'Output': ['e1', 'e2', 'e3'],
	'ZSampMargin' : {'Value':[-3,3], 'Symmetric': True},
	'StepOut' : {'Value': [3,3], 'Symmetric': True},
	'Help': 'http://waynegm.github.io/OpendTect-Plugin-Docs/External_Attributes/GST_Attributes/'
}
#
# Define the compute function
#
def doCompute():
	xs = xa.SI['nrinl']
	ys = xa.SI['nrcrl']
	zs = xa.params['ZSampMargin']['Value'][1] - xa.params['ZSampMargin']['Value'][0] + 1
	kernel = getGaussian(xs, ys, zs)
	while True:
		xa.doInput()

		gx = xa.Input['In-line gradient']
		gy = xa.Input['Cross-line gradient']
		gz = xa.Input['Z gradient']
#
#	Inner product of  gradients
		gx2 = gx * gx
		gy2 = gy * gy
		gz2 = gz * gz
		gxgy = gx * gy
		gxgz = gx * gz
		gygz = gy * gz
#
#	Outer gaussian smoothing
		rgx2 = sconvolve(gx2, kernel)
		rgy2 = sconvolve(gy2, kernel)
		rgz2 = sconvolve(gz2, kernel)
		rgxgy = sconvolve(gxgy, kernel)
		rgxgz = sconvolve(gxgz, kernel)
		rgygz = sconvolve(gygz, kernel)
#
#	Form the structure tensor
		T = np.rollaxis(np.array([	[rgx2,  rgxgy, rgxgz],
									[rgxgy, rgy2,  rgygz],
									[rgxgz, rgygz, rgz2 ]]), 2)
#
#	Get the eigenvalues
		w = np.linalg.eigvalsh(T)
		v = np.rollaxis(np.sort(w),1)
		xa.Output['e1'] = v[2,:]
		xa.Output['e2'] = v[1,:]
		xa.Output['e3'] = v[0,:]
		xa.doOutput()
	
#
# Gaussian filter kernel
#
def getGaussian( xs, ys, zs ):
	tmp = np.zeros((xs,ys,zs))
	tmp[xs//2, ys//2, zs//2] = 1.0
	return gaussian_filter(tmp, (xs/6,ys/6,zs/6), mode='constant')
#
# Convolution of 3D filter with 3D data - only calulates the output for the centre trace
# Numba JIT used to accelerate the calculations

@jit(double(double[:,:,:], double[:,:,:]))
def sconvolve(arr, filt):
	X,Y,Z = arr.shape
	Xf,Yf,Zf = filt.shape
	X2 = X//2
	Y2 = Y//2
	Xf2 = Xf//2
	Yf2 = Yf//2
	Zf2 = Zf//2
	result = np.zeros(Z)
	for i in range(Zf2, Z-Zf2):
		num = 0.0
		for ii in range(Xf):
			for jj in range(Yf):
				for kk in range(Zf):
					num += (filt[Xf-1-ii, Yf-1-jj, Zf-1-kk] * arr[X2-Xf2+ii, Y2-Yf2+jj, i-Zf2+kk])
		result[i] = num
	return result

#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
