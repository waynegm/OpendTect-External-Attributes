#!/usr/bin/python
#
# Simple test of single trace multi-attribute input
#
#
import sys,os
import numpy as np
#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa

#
# These are the attribute parameters
#
xa.params = {
	'Inputs': ['A','B','C'],
	'Output': ['A','B','C','A+B-2C']
}
#
# Define the compute function
#
def doCompute():
	while True:
		xa.doInput()
		xa.Output['A'] = xa.Input['A']
		xa.Output['B'] = xa.Input['B']
		xa.Output['C'] = xa.Input['C']
		xa.Output['A+B-2C'] = xa.Input['A'] + xa.Input['B'] - 2.0 * xa.Input['C']
		xa.doOutput()
	

#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
