#!/usr/bin/python
#
# Vector Mean Filter
#
# Expects stick tensor input outputs dip
#
import sys,os
import numpy as np
from scipy.ndimage import gaussian_filter
from numba import autojit

#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa

#
# These are the attribute parameters
#
xa.params = {
	'Inputs': ['In-line Stick', 'Cross-line Stick', 'Z Stick'],
	'Output': ['Crl_dip', 'Inl_dip', 'True Dip', 'Dip Azimuth'],
	'ZSampMargin' : {'Value':[-3,3], 'Symmetric': True},
	'StepOut' : {'Value': [3,3], 'Symmetric': True},
	'Help': 'http://waynegm.github.io/OpendTect-Plugin-Docs/External_Attributes/Vector_Filters/'
}
#
# Define the compute function
#
def doCompute():
	inlFactor = xa.SI['zstep']/xa.SI['inldist'] * xa.SI['dipFactor']
	crlFactor = xa.SI['zstep']/xa.SI['crldist'] * xa.SI['dipFactor']
	zw = xa.params['ZSampMargin']['Value'][1] - xa.params['ZSampMargin']['Value'][0] + 1
	while True:
		xa.doInput()

		sx = xa.Input['In-line Stick']
		sy = xa.Input['Cross-line Stick']
		sz = xa.Input['Z Stick']
#
#	Apply the Vector Median Filter - here a Numba accelerated version is used
		out = np.empty((3,xa.TI['nrsamp']))
		vecFilter(sx, sy, sz, zw, vecmean_numba, out)
#
#	Get the output
		xa.Output['Crl_dip'] = -out[1,:]/out[2,:]*crlFactor
		xa.Output['Inl_dip'] = -out[0,:]/out[2,:]*inlFactor
		xa.Output['True Dip'] = np.sqrt(xa.Output['Crl_dip']*xa.Output['Crl_dip']+xa.Output['Inl_dip']*xa.Output['Inl_dip'])
		xa.Output['Dip Azimuth'] = np.degrees(np.arctan2(xa.Output['Inl_dip'],xa.Output['Crl_dip']))
		xa.doOutput()
#
#	General filtering function
#	inX, inY and inZ are the vector components
#	window is the window lenght in the Z direction the size in the X and Y directions is determined from the data
#	filtFunc is a Python function that takes an array of vector coordinates and applies the filter
#	outdata is an array that holds the filtered output vectors
def vecFilter(inX, inY, inZ, window, filtFunc, outdata ):
    nx, ny, nz = inX.shape
    half_win = window//2
    outdata.fill(0.0) 
    for z in range(half_win,nz-half_win):
        wX= inX[:,:,z-half_win:z+half_win+1].flatten()
        wY= inY[:,:,z-half_win:z+half_win+1].flatten()
        wZ= inZ[:,:,z-half_win:z+half_win+1].flatten()
        pts = np.array([wX,wY,wZ])
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

vecmean_numba = autojit(vecmean)  
#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
