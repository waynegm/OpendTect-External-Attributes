# External Attribute Logging Test
#
import sys,os
import numpy as np
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
#   Set the  logging level to INFO
#
	xa.logH.setLevel(xa.logging.INFO)
#
#   Write the name of the Python interpreter to the log
#
	xa.logH.info('Executing using: %s', sys.executable)
#
	while True:
		xa.doInput()
		inp = xa.Input['Input1'][0,0,:]
#
#   Write some information about the current trace to the log
#
		inline = xa.TI['inl']
		crossline = xa.TI['crl']
		xa.logH.info('Processing Inline: %s Crossline: %s Amplitude Range: %s to %s', inline, crossline, np.amin(inp), np.amax(inp))
#
		xa.Output = inp
		xa.doOutput()
#
xa.doCompute = doCompute
#
xa.run(sys.argv[1:])
  

