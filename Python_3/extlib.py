#
# Python External Attribute Library
#
import numpy as np
import scipy.ndimage as ndi
from numba import autojit, jit, double

# Scharr 3 point derivative filter
#
def scharr3( input, axis=-1, output=None, mode="reflect", cval=0.0):
	"""Calculate a size 3 Scharr derivative filter.
	Parameters
	----------
	%(input)s
	%(axis)s
	%(output)s
	%(mode)s
	%(cval)s
	"""
	input = np.asarray(input)
	axis = ndi._ni_support._check_axis(axis, input.ndim)
	output, return_value = ndi._ni_support._get_output(output, input)
	ndi.correlate1d(input, [-0.5, 0, 0.5], axis, output, mode, cval, 0)
	axes = [ii for ii in range(input.ndim) if ii != axis]
	for ii in axes:
		ndi.correlate1d(output, [0.12026,0.75948,0.12026], ii, output, mode, cval, 0,)
	return return_value

# Kroon 3 point derivative filter
#
def kroon3( input, axis=-1, output=None, mode="reflect", cval=0.0):
	"""Calculate a size 3 Kroon derivative filter.
	Parameters
	----------
	%(input)s
	%(axis)s
	%(output)s
	%(mode)s
	%(cval)s
	"""
	input = np.asarray(input)
	axis = ndi._ni_support._check_axis(axis, input.ndim)
	output, return_value = ndi._ni_support._get_output(output, input)
	ndi.correlate1d(input, [-0.5, 0, 0.5], axis, output, mode, cval, 0)
	axes = [ii for ii in range(input.ndim) if ii != axis]
	for ii in axes:
		ndi.correlate1d(output, [0.178947,0.642105,0.178947], ii, output, mode, cval, 0,)
	return return_value

# Farid 5 point derivative filter
#
def farid5( input, axis=-1, output=None, mode="reflect", cval=0.0):
	"""Calculate a size 3 Kroon derivative filter.
	Parameters
	----------
	%(input)s
	%(axis)s
	%(output)s
	%(mode)s
	%(cval)s
	"""
	input = np.asarray(input)
	axis = ndi._ni_support._check_axis(axis, input.ndim)
	output, return_value = ndi._ni_support._get_output(output, input)
	ndi.correlate1d(input, [-0.109604, -0.276691,  0.000000, 0.276691, 0.109604], axis, output, mode, cval, 0)
	axes = [ii for ii in range(input.ndim) if ii != axis]
	for ii in axes:
		ndi.correlate1d(output, [0.037659,  0.249153,  0.426375, 0.249153, 0.037659], ii, output, mode, cval, 0,)
	return return_value
# Gaussian filter kernel
#
def getGaussian( xs, ys, zs ):
	"""Return a gaussian filter kernel of the specified size
	"""
	tmp = np.zeros((xs,ys,zs))
	tmp[xs//2, ys//2, zs//2] = 1.0
	return ndi.gaussian_filter(tmp, (xs/6,ys/6,zs/6), mode='constant')


# Convolution of 3D filter with 3D data - only calulates the output for the centre trace
# Numba JIT used to accelerate the calculations
#
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
#	General vector filtering function
#	indata contains the vector components
#	window is the window length in the Z direction the size in the X and Y directions is determined from the data
#	filtFunc is a Python function that takes an array of vector coordinates and applies the filter
#	outdata is an array that holds the filtered output vectors
def vecFilter(indata, window, filtFunc, outdata ):
    nz = indata.shape[3]
    half_win = window//2
    outdata.fill(0.0) 
    for z in range(half_win,nz-half_win):
        pts = indata[:,:,:,z-half_win:z+half_win+1].reshape(3,-1)
        outdata[:,z] = filtFunc(pts)
    for z in range(half_win):
        outdata[:,z] = outdata[:,half_win]
    for z in range(nz-half_win, nz):
        outdata[:,z] = outdata[:,nz-half_win-1]

#
#	Calculate the mean vector of the contents of the pts array 
def vecmean(pts):
	n = pts.shape[-1]
	dist = np.zeros((n))
	X=Y=Z=0.0
	for i in range(n):
		X += pts[0,i]
		Y += pts[1,i]
		Z += pts[2,i]
	return np.array([X,Y,Z])/n
#
#	Calculate the vector median of the contents of the pts array - this uses absolute distance
def vmf_l1(pts):
    n = pts.shape[-1]
    dist = np.zeros((n))
    for i in range(n):
        for j in range(i+1,n):
            tmp = abs(pts[0,j]-pts[0,i]) + abs(pts[1,j]-pts[1,i]) + abs(pts[2,j]-pts[2,i])
            dist[i] += tmp
            dist[j] += tmp
    return pts[:,np.argmin(dist)]

#
#	Calculate the vector median of the contents of the pts array - this uses squared distance
def vmf_l2(pts):
    n = pts.shape[-1]
    dist = np.zeros((n))
    for i in range(n):
        for j in range(i+1,n):
            tmp = (pts[0,j]-pts[0,i])**2 + (pts[1,j]-pts[1,i])**2 + (pts[2,j]-pts[2,i])**2
            dist[i] += tmp
            dist[j] += tmp
    return pts[:,np.argmin(dist)]
#
#	Stride trickery for rolling windows
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
