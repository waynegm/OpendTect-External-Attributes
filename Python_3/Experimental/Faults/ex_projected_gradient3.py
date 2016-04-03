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
	'Inputs': ['Input', 'Crl_dip','Inl_dip'],
	'ZSampMargin' : {'Value':[-1,1], 'Minimum': [-1,1], 'Symmetric': True},
	'StepOut' : {'Value': [1,1], 'Minimum':[1,1], 'Symmetric': True},
	'Help': 'http://waynegm.github.io/OpendTect-Plugin-Docs/External_Attributes/Faults/'
}
#
# Define the compute function
#
def doCompute():
	ilndx = xa.SI['nrinl']//2
	crldx = xa.SI['nrcrl']//2
	print("Hello", file=sys.stderr)
	inlFactor = xa.SI['zstep']/xa.SI['inldist'] * xa.SI['dipFactor']
	crlFactor = xa.SI['zstep']/xa.SI['crldist'] * xa.SI['dipFactor']
	while True:
		xa.doInput()

		d = xa.Input['Input'][ilndx,crldx,:]
#		gx = -xa.Input['Inl_dip'][xs,ys,:]/inlFactor
#		gy = -xa.Input['Crl_dip'][xs,ys,:]/crlFactor
#		gz = np.ones(gx.shape)
#		s = np.sqrt(gx*gx+gy*gy+gz*gz)
#		print(d.shape,file=sys.stderr)
#		print(s.shape,file=sys.stderr)
#
# Compute gradients
#		dx = xl.kroon3( d, axis=0 )[xs,ys,:]
#		dy = xl.kroon3( d, axis=1 )[xs,ys,:]
#		dz = xl.kroon3( d, axis=2 )[xs,ys,:]
#
#	Compute maximum gradient tangential to orientation normal
#		px = dx*(1.0 - gx/s)
#		py = dy*(1.0 - gy/s)
#		px = dz*(1.0 - gz/s)
#
#	Output
		xa.Output = d
		xa.doOutput()
	
#
# Assign the compute function to the attribute
#
xa.doCompute = doCompute
#
# Do it
#
xa.run(sys.argv[1:])
  
