# Maximum gradient along dip
#
import sys,os
import numpy as np
# Import the module with the I/O scaffolding of the External Attribute
#
sys.path.insert(0, os.path.join(sys.path[0], '..', '..'))
import extattrib as xa
import extlib as xl

#
# These are the attribute parameters
#
xa.params = {
	'Inputs': ['Input Volume', 'In-line dip','Cross-line dip'],
	'Output': ['Projected Gradient'],
	'ZSampMargin' : {'Value':[-1,1], 'Hidden': True},
	'StepOut' : {'Value': [1,1], 'Hidden': True},
	'Parallel': True,
	'Help': 'http://waynegm.github.io/OpendTect-Plugin-Docs/External_Attributes/Faults/'
}
#
# Define the compute function
#
def doCompute():
	hxs = xa.SI['nrinl']//2
	hys = xa.SI['nrcrl']//2
	inlFactor = xa.SI['zstep']/xa.SI['inldist'] * xa.SI['dipFactor']
	crlFactor = xa.SI['zstep']/xa.SI['crldist'] * xa.SI['dipFactor']
	while True:
		xa.doInput()
		
		g  = xa.Input['Input Volume']
		dx = -xa.Input['In-line dip'][hxs,hys,:]/inlFactor
		dy = -xa.Input['Cross-line dip'][hxs,hys,:]/crlFactor
#
# Compute surface normal unit vector
		dz = np.ones(dx.shape)
		d = np.sqrt(dx*dx + dy*dy + dz*dz )
		dx /= d
		dy /= d
		dz /= d
#
# Compute gradients
		gx = xl.kroon3( g, axis=0 )[hxs,hys,:]
		gy = xl.kroon3( g, axis=1 )[hxs,hys,:]
		gz = xl.kroon3( g, axis=2 )[hxs,hys,:]
		g2 = gx*gx + gy*gy + gz*gz
#
#	Compute projection of gradient onto surface normal unit vector
		p = dx*gx + dy*gy + dz*gz
#
# Compute gradient projection on surface and output
		xa.Output['Projected Gradient'] = np.sqrt(g2 - p*p)
		xa.doOutput()
	
#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
