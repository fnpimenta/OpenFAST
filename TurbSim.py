"""Read/Write TurbSim File

Part of weio library: https://github.com/ebranlard/weio

"""
import pandas as pd
import numpy as np
import os
import struct

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