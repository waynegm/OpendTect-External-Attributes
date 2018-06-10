# External Attribute Debug Test
#
import sys,os
import numpy as np
import web_pdb
#
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import extattrib as xa
#
xa.params = {
	'Inputs': ['Input1'],
	'Parallel' : False
}
#
def doCompute():
#
#   Start debugging before computation starts
#
	web_pdb.set_trace()
#
	while True:
		xa.doInput()
		inp = xa.Input['Input1'][0,0,:]
#
#   Add some more local variables
#
		inline = xa.TI['inl']
		crossline = xa.TI['crl']
#
		xa.Output = inp
		xa.doOutput()
#
xa.doCompute = doCompute
#
xa.run(sys.argv[1:])
  

