#!/usr/bin/python
#
# Response attribute - changes input to square wave with min and max matching input
#
#
import sys,os
import numpy as np
from numba import jit
#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa
#
# These are the attribute parameters
#
xa.params = {
	'Inputs': ['Input'],
	'ZSampMargin' : {'Value':[-10,10], 'Hidden': True},
	'Help': 'http://waynegm.github.io/OpendTect-Plugin-Docs/External_Attributes/Response/'
}
#
# Define the compute function
#
def doCompute():
	while True:
		xa.doInput()

		inp = xa.Input['Input'][0,0,:]
		outp = np.zeros(inp.shape)

		response( inp, outp)
#
#	Get the output
		xa.Output = outp
		xa.doOutput()
#
# Square wave a trace
#
@jit(nopython=True)
def response(inp, outp):
    ns = inp.shape[0]
    start = 0
    pos = inp[0]>0
    for i in range(ns):
        if inp[i]<0 and pos:
            if start<i-1:
                outp[start:i-1] = np.max(inp[start:i-1])
            else:
                outp[start] = inp[start]
            pos = False
            start = i
        if inp[i]>0 and not pos:
            if start<i-1:
                outp[start:i-1] = np.min(inp[start:i-1])
            else:
                outp[start] = inp[start]
            pos=True
            start = i
#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
