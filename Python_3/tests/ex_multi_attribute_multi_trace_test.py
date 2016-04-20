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
	'Output': ['A','B','C','A+B-2C'],
	'ZSampMargin' : {'Value': [-1,1], 'Symmetric': True},
	'StepOut' : {'Value': [1,1]}
}
#
# Define the compute function
#
def doCompute():
	hxs = xa.SI['nrinl']//2
	hys = xa.SI['nrcrl']//2
	while True:
		xa.doInput()
		xa.Output['A'] = np.mean(xa.Input['A'],axis=(0,1))
		xa.Output['B'] = np.mean(xa.Input['B'],axis=(0,1))
		xa.Output['C'] = np.mean(xa.Input['C'],axis=(0,1))
		xa.Output['A+B-2C'] = xa.Input['A'][hxs,hys,:] + xa.Input['B'][hxs,hys,:] - 2.0 * xa.Input['C'][hxs,hys,:]
		xa.doOutput()
	

#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
