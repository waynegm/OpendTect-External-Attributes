#!/usr/bin/python
#
# Do nothing - multi-trace
#
import sys, os
import numpy as np
from scipy.ndimage import prewitt
#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa

#
# These are the attribute parameters
#
xa.params = {
	'Input': 'Input',
	'ZSampMargin' : {'Value': [-1,1], 'Symmetric': True},
	'StepOut' : {'Value': [1,1]}
}
#
# Define the compute function
#
def doCompute():
#
# index of current trace position in Input numpy array
#
	ilndx = xa.SI['nrinl']//2
	crldx = xa.SI['nrcrl']//2
	while True:
		xa.doInput()
		xa.Output = xa.Input[ilndx,crldx,:]
		xa.doOutput()
	
#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
