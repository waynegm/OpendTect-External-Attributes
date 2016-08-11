#!/usr/bin/python
#
# Local cross-correlation of two inputs 
#
#
import sys,os
import numpy as np
from scipy import signal
from numba import autojit, jit, double, int64
#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa
import extlib as xl
#
# These are the attribute parameters
#
xa.params = {
	'Inputs': ['Reference', 'Match'],
	'Output': ['Shift', 'Quality'],
	'ZSampMargin' : {'Value':[-10,10], 'Symmetric': True},
	'Par_0': {'Name': 'Max Lag (samples)', 'Value': 5},
	'Help': 'http://waynegm.github.io/OpendTect-Plugin-Docs/External_Attributes/Miscellaneous/'
}
#
# Define the compute function
#
def doCompute():
	zw = xa.params['ZSampMargin']['Value'][1] - xa.params['ZSampMargin']['Value'][0] + 1
	nlag = int(xa.params['Par_0']['Value'])

	while True:
		xa.doInput()

		ref = xa.Input['Reference'][0,0,:]
		match = xa.Input['Match'][0,0,:]
		qual = np.zeros(ref.shape)

		lag = localCorr(ref,match,zw,nlag)
#
#	Get the output
		xa.Output['Shift'] = lag*xa.SI['zstep']
		xa.Output['Quality'] = qual
		xa.doOutput()
#
# Local correlation - naive implementation
#
@jit(double(double[:], double[:],int64, int64))
def localCorr( reference, match, zw, nlag ):
	window = signal.blackman(zw)
	lags = 2*nlag + 1
	hzw = zw//2
	ns = reference.shape[0]
	refpad = np.pad(reference, hzw, 'edge')
	matpad = np.pad(match, hzw+nlag, 'edge')
	lag = np.zeros(ns)
	for ir in range(0,ns):
		cor = np.zeros(lags)
		rbeg = ir
		rend = rbeg + zw
		for il in range(-nlag, nlag):
			lbeg = ir + nlag +il
			lend = lbeg + zw
			cor[il+nlag] = np.mean(refpad[rbeg:rend]*matpad[lbeg:lend]*window)
		pos = np.argmax(cor)
		if pos>=1 and pos<lags:
			cp = (cor[pos-1]-cor[pos+1])/(2.*cor[pos-1]-4.*cor[pos]+2.*cor[pos+1])
			lag[ir] = (pos-nlag+cp)
		else:
			lag[ir]=0.0
	return lag
#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
