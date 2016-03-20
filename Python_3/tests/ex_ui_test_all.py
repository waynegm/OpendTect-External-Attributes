#!/usr/bin/python
#
# Test availability of all UI Elements
#
import sys,os
import numpy as np
#
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa 
#
# These are the attribute parametersas xa
#
xa.params = {
	'Inputs': ['Input_1','Input_2','Input_3','Input_4','Input_5','Input_6'],
	'Output': ['Output_1', 'Output_2', 'Output_3'],
	'ZSampMargin' : {'Value':[-1,1]},
	'StepOut' : {'Value': [1,1]},
	'Select' : {'Name': 'Selection', 'Values': ['Item 0', 'Item 1', 'Item 2', 'Item 3'], 'Selection': 2},
	'Par_0': {'Name': 'Parameter 0', 'Value': 0},
	'Par_1': {'Name': 'Parameter 1', 'Value': 1},
	'Par_2': {'Name': 'Parameter 2', 'Value': 2},
	'Par_3': {'Name': 'Parameter 3', 'Value': 3},
	'Par_4': {'Name': 'Parameter 4', 'Value': 4},
	'Par_5': {'Name': 'Parameter 5', 'Value': 5},
	'Parallel': False,
	'Help': 'https://github.com/waynegm'
}
#
# Define the compute function
#
def doCompute():
	json.dump(xa.params, sys.stderr)
	sys.stderr.flush()
	while True:
		xa.doInput()
		xa.doOutput()
	


#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])

