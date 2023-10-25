import pandas as pd
import numpy as np
import os
import struct
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

def TurbSimData(f):
	data = {}
	scl = np.zeros(3, np.float32); off = np.zeros(3, np.float32)
		  
	# Reading header info
	f.seek(0)
	ID, nz, ny, nTwr, nt                      = struct.unpack('<h4l', f.read(2+4*4))
	dz, dy, dt, uHub, zHub, zBottom           = struct.unpack('<6f' , f.read(6*4))
	scl[0],off[0],scl[1],off[1],scl[2],off[2] = struct.unpack('<6f' , f.read(6*4))
	nChar, = struct.unpack('<l',  f.read(4))
	info = (f.read(nChar)).decode()
	
	# Reading turbulence field
	u    = np.zeros((3,nt,ny,nz))
	uTwr = np.zeros((3,nt,nTwr))
	# For loop on time (acts as buffer reading, and only possible way when nTwr>0)
	for it in range(nt):
		Buffer = np.frombuffer(f.read(2*3*ny*nz), dtype=np.int16).astype(np.float32).reshape([3, ny, nz], order='F')
		u[:,it,:,:]=Buffer
		Buffer = np.frombuffer(f.read(2*3*nTwr), dtype=np.int16).astype(np.float32).reshape([3, nTwr], order='F')
		uTwr[:,it,:]=Buffer
	u -= off[:, None, None, None]
	u /= scl[:, None, None, None]
	data['u']    = u
	uTwr -= off[:, None, None]
	uTwr /= scl[:, None, None]
	data['uTwr'] = uTwr

	data['info'] = info
	data['ID']   = ID
	tdecimals=8
	data['dt']   = np.round(dt, tdecimals) # dt is stored in single precision in the TurbSim output
	data['y']    = np.arange(ny)*dy 
	data['y']   -= np.mean(data['y']) # y always centered on 0
	data['z']    = np.arange(nz)*dz +zBottom
	data['t']    = np.round(np.arange(nt)*dt, tdecimals)
	data['zTwr'] =-np.arange(nTwr)*dz + zBottom
	data['zRef'] = zHub
	data['uRef'] = uHub

	return data

@st.cache_data()
def FullFieldPlot(f,_placeholder,nstep=1000):
	fdata = TurbSimFile(f)
	data = fdata['u'][0,::nstep,:,:]

	tmin,tmax = int(np.round(fdata['t'][0],0)),int(np.round(fdata['t'][-1],0))
	ymin,ymax = int(np.round(fdata['y'][0],0)),int(np.round(fdata['y'][-1],0))
	zmin,zmax = int(np.round(fdata['z'][0],0)),int(np.round(fdata['z'][-1],0))

	x,y,z = np.mgrid[tmin:tmax:(len(data)+1)*1j,ymin:ymax:(fdata.ny+1)*1j,zmin:zmax:(fdata.nz+1)*1j]

	colors = plt.cm.jet((data-np.min(data))/(np.max(data)-np.min(data)))
	norm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))

	visiblebox = np.random.choice([True,True],fdata['u'][0,::nstep,:,:].shape)

	fig = plt.figure()

	ax = plt.subplot(111,projection ='3d')

	vox = ax.voxels(x,y,z,visiblebox,facecolors=colors,alpha = 0.5)

	m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
	m.set_array([])
   
	plt.colorbar(m,fraction=0.025,cax=ax)

	ax.view_init(30, 210)
	#ax.axis('off')
	#st.write(np.shape(data))

	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')

	ax.set_xlim(tmin,tmax)
	ax.set_ylim(ymin,ymax)
	ax.set_zlim(zmin,zmax)

	ax.grid(False)
	st.pyplot(fig)

	return fig

EmptyFileError = type('EmptyFileError', (Exception,),{})
File=dict
class TurbSimFile(File):
	""" 
	Read/write a TurbSim turbulence file (.bts). The object behaves as a dictionary.

	Main keys
	---------
	- 'u': velocity field, shape (3 x nt x ny x nz)
	- 'y', 'z', 't': space and time coordinates 
	- 'dt', 'ID', 'info'
	- 'zTwr', 'uTwr': tower coordinates and field if present (3 x nt x nTwr)
	- 'zRef', 'uRef': height and velocity at a reference point (usually not hub)

	Main methods
	------------
	- read, write, toDataFrame, keys
	- valuesAt, vertProfile, horizontalPlane, verticalPlane, closestPoint
	- fitPowerLaw
	- makePeriodic, checkPeriodic

	Examples
	--------

		ts = TurbSimFile('Turb.bts')
		print(ts.keys())
		print(ts['u'].shape)  
		u,v,w = ts.valuesAt(y=10.5, z=90)


	"""

	@staticmethod
	def defaultExtensions():
		return ['.bts']

	@staticmethod
	def formatName():
		return 'TurbSim binary'

	def __init__(self, filename=None, **kwargs):
		self.filename = None
		if filename:
			self.filename = filename
			self.read(filename, **kwargs)

	def read(self, filename=None, header_only=False, tdecimals=8):
		""" read BTS file, with field: 
					 u    (3 x nt x ny x nz)
					 uTwr (3 x nt x nTwr)
		"""
		scl = np.zeros(3, np.float32); off = np.zeros(3, np.float32)
		
		# Reading header info
		self.filename.seek(0)
		ID, nz, ny, nTwr, nt                      = struct.unpack('<h4l', self.filename.read(2+4*4))
		dz, dy, dt, uHub, zHub, zBottom           = struct.unpack('<6f' , self.filename.read(6*4)  )
		scl[0],off[0],scl[1],off[1],scl[2],off[2] = struct.unpack('<6f' , self.filename.read(6*4))
		nChar, = struct.unpack('<l',  self.filename.read(4))
		info = (self.filename.read(nChar)).decode()
		# Reading turbulence field
		if not header_only: 
			u    = np.zeros((3,nt,ny,nz))
			uTwr = np.zeros((3,nt,nTwr))
			# For loop on time (acts as buffer reading, and only possible way when nTwr>0)
			for it in range(nt):
				Buffer = np.frombuffer(self.filename.read(2*3*ny*nz), dtype=np.int16).astype(np.float32).reshape([3, ny, nz], order='F')
				u[:,it,:,:]=Buffer
				Buffer = np.frombuffer(self.filename.read(2*3*nTwr), dtype=np.int16).astype(np.float32).reshape([3, nTwr], order='F')
				uTwr[:,it,:]=Buffer
			u -= off[:, None, None, None]
			u /= scl[:, None, None, None]
			self['u']    = u
			uTwr -= off[:, None, None]
			uTwr /= scl[:, None, None]
			self['uTwr'] = uTwr
		self['info'] = info
		self['ID']   = ID
		self['dt']   = np.round(dt, tdecimals) # dt is stored in single precision in the TurbSim output
		self['y']    = np.arange(ny)*dy 
		self['y']   -= np.mean(self['y']) # y always centered on 0
		self['z']    = np.arange(nz)*dz +zBottom
		self['t']    = np.round(np.arange(nt)*dt, tdecimals)
		self['zTwr'] =-np.arange(nTwr)*dz + zBottom
		self['zRef'] = zHub
		self['uRef'] = uHub

	# --------------------------------------------------------------------------------}
	# --- Convenient properties (matching Mann Box interface as well)
	# --------------------------------------------------------------------------------{
	@property
	def z(self): return self['z'] # np.arange(nz)*dz +zBottom

	@property
	def y(self): return self['y'] # np.arange(ny)*dy  - np.mean( np.arange(ny)*dy )
 
	@property
	def t(self): return self['t'] # np.arange(nt)*dt

	# NOTE: it would be best to use dz and dy as given in the file to avoid numerical issues
	@property
	def dz(self): return self['z'][1]-self['z'][0]

	@property
	def dy(self): return self['y'][1]-self['y'][0]

	@property
	def dt(self): return self['t'][1]-self['t'][0]

	@property
	def nz(self): return len(self.z)

	@property
	def ny(self): return len(self.y)

	@property
	def nt(self): return len(self.t)

	# --------------------------------------------------------------------------------}
	# --- Extracting relevant "Line" data at one point
	# --------------------------------------------------------------------------------{
	def valuesAt(self, y, z, method='nearest'):
		""" return wind speed time series at a point """
		if method == 'nearest':
			iy, iz = self.closestPoint(y, z)
			u = self['u'][0,:,iy,iz]
			v = self['u'][1,:,iy,iz]
			w = self['u'][2,:,iy,iz]
		else:
			raise NotImplementedError()
		return u, v, w

	def closestPoint(self, y, z):
		iy = np.argmin(np.abs(self['y']-y))
		iz = np.argmin(np.abs(self['z']-z))
		return iy,iz

	def hubValues(self, zHub=None):
		if zHub is None:
			try:
				zHub=float(self['zRef'])
				bHub=True
			except:
				bHub=False
				iz = np.argmin(np.abs(self['z']-(self['z'][0]+self['z'][-1])/2))
				zHub = self['z'][iz]
		else:
			bHub=True
		try:
			uHub=float(self['uRef'])
		except:
			iz = np.argmin(np.abs(self['z']-zHub))
			iy = np.argmin(np.abs(self['y']-(self['y'][0]+self['y'][-1])/2))
			uHub = np.mean(self['u'][0,:,iy,iz])
		return zHub, uHub, bHub

	def midValues(self):
		iy,iz = self.iMid
		zMid = self['z'][iz]
		#yMid = self['y'][iy] # always 0
		uMid = np.mean(self['u'][0,:,iy,iz])
		return zMid, uMid

	@property
	def iMid(self):
		iy = np.argmin(np.abs(self['y']-(self['y'][0]+self['y'][-1])/2))
		iz = np.argmin(np.abs(self['z']-(self['z'][0]+self['z'][-1])/2))
		return iy,iz

	def closestPoint(self, y, z):
		iy = np.argmin(np.abs(self['y']-y))
		iz = np.argmin(np.abs(self['z']-z))
		return iy,iz

	def _longiline(ts, iy0=None, iz0=None, removeMean=False):
		""" return velocity components on a longitudinal line
		If no index is provided, computed at mid box 
		"""
		if iy0 is None:
			iy0,iz0 = ts.iMid
		u = ts['u'][0,:,iy0,iz0]
		v = ts['u'][1,:,iy0,iz0]
		w = ts['u'][2,:,iy0,iz0]
		if removeMean:
			u -= np.mean(u)
			v -= np.mean(v)
			w -= np.mean(w)
		return u, v, w

	def _latline(ts, ix0=None, iz0=None, removeMean=False):
		""" return velocity components on a lateral line
		If no index is provided, computed at mid box 
		"""
		if ix0 is None:
			iy0,iz0 = ts.iMid
			ix0=int(len(ts['t'])/2)
		u = ts['u'][0,ix0,:,iz0]
		v = ts['u'][1,ix0,:,iz0]
		w = ts['u'][2,ix0,:,iz0]
		if removeMean:
			u -= np.mean(u)
			v -= np.mean(v)
			w -= np.mean(w)
		return u, v, w

	def _vertline(ts, ix0=None, iy0=None, removeMean=False):
		""" return velocity components on a vertical line
		If no index is provided, computed at mid box 
		"""
		if ix0 is None:
			iy0,iz0 = ts.iMid
			ix0=int(len(ts['t'])/2)
		u = ts['u'][0,ix0,iy0,:]
		v = ts['u'][1,ix0,iy0,:]
		w = ts['u'][2,ix0,iy0,:]
		if removeMean:
			u -= np.mean(u)
			v -= np.mean(v)
			w -= np.mean(w)
		return u, v, w

	# --------------------------------------------------------------------------------}
	# --- Extracting plane data at one point
	# --------------------------------------------------------------------------------{
	def horizontalPlane(ts, z=None, iz0=None, removeMean=False):
		""" return velocity components on a horizontal plane
		If no z value is provided, returned at mid box 
		"""
		if z is None and iz0 is None:
			_,iz0 = ts.iMid
		elif z is not None:
			_, iz0 = ts.closestPoint(ts.y[0], z) 

		u = ts['u'][0,:,:,iz0]
		v = ts['u'][1,:,:,iz0]
		w = ts['u'][2,:,:,iz0]
		if removeMean:
			u -= np.mean(u)
			v -= np.mean(v)
			w -= np.mean(w)
		return u, v, w

	def verticalPlane(ts, y=None, iy0=None, removeMean=False):
		""" return velocity components on a vertical plane
		If no y value is provided, returned at mid box 
		"""
		if y is None and iy0 is None:
			iy0,_ = ts.iMid
		elif y is not None:
			iy0, _ = ts.closestPoint(y, ts.z[0]) 

		u = ts['u'][0,:,iy0,:]
		v = ts['u'][1,:,iy0,:]
		w = ts['u'][2,:,iy0,:]
		if removeMean:
			u -= np.mean(u)
			v -= np.mean(v)
			w -= np.mean(w)
		return u, v, w

	# --------------------------------------------------------------------------------}
	# --- Extracting average data
	# --------------------------------------------------------------------------------{
	def vertProfile(self, y_span='full'):
		""" Vertical profile of the box
		INPUTS:
		 - y_span: if 'full', average the vertical profile accross all y-values
				   if 'mid', average the vertical profile at the middle y value
		"""
		if y_span=='full':
			m = np.mean(np.mean(self['u'][:,:,:,:], axis=1), axis=1)
			s = np.std( np.std( self['u'][:,:,:,:], axis=1), axis=1)
		elif y_span=='mid':
			iy, iz = self.iMid
			m = np.mean(self['u'][:,:,iy,:], axis=1)
			s = np.std( self['u'][:,:,iy,:], axis=1)
		else:
			raise NotImplementedError()
		return self.z, m, s


	# --------------------------------------------------------------------------------}
	# --- Computation of useful quantities
	# --------------------------------------------------------------------------------{
	def crosscorr_y(ts, iy0=None, iz0=None):
		""" Cross correlation along y
		If no index is provided, computed at mid box 
		"""
		y = ts['y']
		if iy0 is None:
			iy0,iz0 = ts.iMid
		u, v, w = ts._longiline(iy0=iy0, iz0=iz0, removeMean=True)
		rho_uu_y=np.zeros(len(y))
		rho_vv_y=np.zeros(len(y))
		rho_ww_y=np.zeros(len(y))
		for iy,_ in enumerate(y):
			ud, vd, wd = ts._longiline(iy0=iy, iz0=iz0, removeMean=True)
			rho_uu_y[iy] = np.mean(u*ud)/(np.std(u)*np.std(ud))
			rho_vv_y[iy] = np.mean(v*vd)/(np.std(v)*np.std(vd))
			rho_ww_y[iy] = np.mean(w*wd)/(np.std(w)*np.std(wd))
		return y, rho_uu_y, rho_vv_y, rho_ww_y

	def crosscorr_z(ts, iy0=None, iz0=None):
		""" 
		Cross correlation along z, mid box
		If no index is provided, computed at mid box 
		"""
		z = ts['z']
		if iy0 is None:
			iy0,iz0 = ts.iMid
		u, v, w = ts._longiline(iy0=iy0, iz0=iz0, removeMean=True)
		rho_uu_z = np.zeros(len(z))
		rho_vv_z = np.zeros(len(z))
		rho_ww_z = np.zeros(len(z))
		for iz,_ in enumerate(z):
			ud, vd, wd = ts._longiline(iy0=iy0, iz0=iz, removeMean=True)
			rho_uu_z[iz] = np.mean(u*ud)/(np.std(u)*np.std(ud))
			rho_vv_z[iz] = np.mean(v*vd)/(np.std(v)*np.std(vd))
			rho_ww_z[iz] = np.mean(w*wd)/(np.std(w)*np.std(wd))
		return z, rho_uu_z, rho_vv_z, rho_ww_z


	def csd_longi(ts, iy0=None, iz0=None):
		""" Compute cross spectral density
		If no index is provided, computed at mid box 
		"""
		import scipy.signal as sig
		u, v, w = ts._longiline(iy0=iy0, iz0=iz0, removeMean=True)
		t       = ts['t']
		dt      = t[1]-t[0]
		fs      = 1/dt
		fc, chi_uu = sig.csd(u, u, fs=fs, scaling='density') #nperseg=4096, noverlap=2048, detrend='constant')
		fc, chi_vv = sig.csd(v, v, fs=fs, scaling='density') #nperseg=4096, noverlap=2048, detrend='constant')
		fc, chi_ww = sig.csd(w, w, fs=fs, scaling='density') #nperseg=4096, noverlap=2048, detrend='constant')
		return fc, chi_uu, chi_vv, chi_ww