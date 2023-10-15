import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ipywidgets import *
from scipy import signal , integrate, linalg
from scipy.signal import hilbert
import ipywidgets as widgets
from IPython.display import display, HTML
import os
import random
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import least_squares , curve_fit
from matplotlib import cm

pi = np.pi

def ModeFit(h,a2,a3,a4,a5):
    a6 = (1-a2-a3-a4-a5)
    return a2*h**2 + a3*h**3 + a4*h**4 + a5*h**5 + a6*h**6
    
def ModesPlot(h,mass,stiff,N=2,n_plot=2,L=100,mtop=0):
    seg = np.linspace(0,1,N+1)*L

    f_m = interp1d(h*L,mass)
    f_s = interp1d(h*L,stiff)    
     
    points = np.linspace(0,1,N+1)*L

    mi = f_m(points)
    si = f_s(points)
    
    mi = mi[1:]/2 + mi[:-1]/2
    si = si[1:]/2 + si[:-1]/2

    rs_h = np.linspace(0,h[-1],1000)*L
    rs_m = f_m(rs_h)
    rs_s = f_s(rs_h)
    
    MassMatrix = np.zeros((N,N))
    for i in range(N-1):
        MassMatrix[i,i] = mi[i]*(points[i+1]-points[i])/2 + mi[i+1]*(points[i+2]-points[i+1])/2
    MassMatrix[-1,-1] = mi[-1]*(points[-1]-points[-2])/2 + mtop
    
    StiffMatrix = np.zeros_like(MassMatrix)
    
    KMatrix = np.zeros((2*N,2*N))
    for i in range(N-1):
        L1 = points[i+1]-points[i]
        EI1 = si[i]

        L2 = points[i+2]-points[i+1]
        EI2 = si[i+1]
       
        KMatrix[2*i,2*i] += 12*EI1/L1**3 + 12*EI2/L2**3
        KMatrix[2*i,2*i+1] += 6*EI2/L2**2 - 6*EI1/L1**2
        
        KMatrix[2*i+1,2*i] += 6*EI2/L2**2 - 6*EI1/L1**2
        KMatrix[2*i+1,2*i+1] += 4*EI1/L1 + 4*EI2/L2
        
        KMatrix[2*i,2*i+2] += -12*EI2/L2**3
        KMatrix[2*i+2,2*i] += -12*EI2/L2**3

        KMatrix[2*i,2*i+3] += 6*EI2/L2**2
        KMatrix[2*i+3,2*i] += 6*EI2/L2**2        
        
        KMatrix[2*i+1,2*i+2] += -6*EI2/L2**2
        KMatrix[2*i+2,2*i+1] += -6*EI2/L2**2

        KMatrix[2*i+1,2*i+3] += 2*EI2/L2
        KMatrix[2*i+3,2*i+1] += 2*EI2/L2          
    
    KMatrix[-2,-2] += +12*EI2/L2**3
    KMatrix[-2,-1] +=  -6*EI2/L2**2    

    KMatrix[-1,-2] += -6*EI2/L2**2
    KMatrix[-1,-1] += +4*EI2/L2          
    Kinv = np.linalg.inv(KMatrix) 

    CMatrix = np.zeros((N,N))
    for i in range(N):
        CMatrix[i,:] = Kinv[2*i,0::2]
        
    KMatrix = np.linalg.inv(CMatrix)

    for i in range(N):
        fv = np.zeros(N)
        fv[i] = 1
        StiffMatrix[:,i] = KMatrix@fv
        
    w,v = np.linalg.eig(np.linalg.inv(MassMatrix)@StiffMatrix)
    freq = np.sqrt(w)/(2*pi)
    
    plt.figure(figsize = (16,10))
    gs = gridspec.GridSpec(2,2)
    gs.update(hspace=0)
    ax1 = plt.subplot(gs[0,0])
    ax3 = plt.subplot(gs[:,1])
    
    ax1.plot(rs_h,rs_m,'darkblue',linewidth=1)
    for i in range(N):
        ax1.plot([points[i],points[i+1]],[mi[i],mi[i]],'b')
        if i<(N-1):
            ax1.plot([points[i+1],points[i+1]],[mi[i],mi[i+1]],'b')
    ax1.set_xlim(0,L)
    ax1.set_ylim(0)
    ax1.grid()
    
    ax2 = ax1.twinx()

    ax2.plot(rs_h,rs_s*1e-9,'darkred',linewidth=1)
    for i in range(N):
        ax2.plot([points[i],points[i+1]],[si[i]*1e-9,si[i]*1e-9],'r')
        if i<(N-1):
            ax2.plot([points[i+1],points[i+1]],[si[i]*1e-9,si[i+1]*1e-9],'r')

    ax2.set_xlim(0,L)
    ax2.set_ylim(0)
    ax2.grid()

    ax1.set_yticks(ax1.get_yticks()[ax1.get_yticks()<=int(max(mass))]/ax1.get_ylim()[1]*ax1.get_ylim()[1])
    ax2.set_yticks(ax1.get_yticks()/ax1.get_ylim()[1]*ax2.get_ylim()[1])

    ax3.plot(h*0,h*L,'k')
    ax3.plot(points*0,points,'ok')
    ax3.set_ylim(0,L)

    ax1.set_xlabel('Tower height (m)')
    ax1.set_title('Tower distributed properties')
    ax3.set_title('Modes')

    ax3.set_ylabel('Tower height (m)')
    ax3.set_xticklabels('')

    ax1.set_ylabel(u'Mass density (kg/m)',color='red')
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')

    ax2.set_ylabel('Stiffness (GN m$^2$)',color='blue')
    ax2.spines['right'].set_color('blue')
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['left'].set_color('red')

    colors = cm.rainbow(np.linspace(0, 1, n_plot))
    
    for i in range(1,1+n_plot):
        mode = np.zeros(len(v)+1)
        mode[1:] = v[:,-i]
        popt, pcov = curve_fit(ModeFit, points/L , mode/mode[-1])
        phi = ' %.2f$x^2$'%popt[0]
        for j in range(3):
            if popt[j+1]>0:
                phi += ' + %.2f$x^%d$'%(popt[j+1],j+3)
            else:
                phi += ' - %.2f$x^%d$'%(abs(popt[j+1]),j+3)
        if (1-np.sum(popt))>0:
            phi += ' + %.2f$x^%d$'%(1-np.sum(popt),6)
        else:
            phi += ' - %.2f$x^%d$'%(abs(1-np.sum(popt)),6)
            
        ax3.plot(mode,points,'o',color=colors[i-1])
        
        fitmode = ModeFit(rs_h/L,*popt)

        ax3.plot(fitmode/fitmode[-1]*mode[-1],rs_h,'--',color=colors[i-1])
        ax3.plot(0,0,'--o',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi)) 

    ax3.legend(loc='lower left',
               bbox_to_anchor=(-1.2,0),
               ncol=1,
               fancybox=False,
               framealpha=1,
               frameon=False)
    return 

def ModesEstimator(filename):
    '''
    Interactive estimator of the mode shapes for a wind turbine tower. 

    Parameters
    ----------
    filename : string
        File containing the mass and stiffness evolution of the tower over its height.
    '''
    if not(type(filename)==str):
        print('Error loading data. Input is not a string.')
        return
    else:
        if filename[-4:]=='.csv':
            data = pd.read_csv(filename, sep=',',header=0,on_bad_lines='skip',skiprows=[1],encoding_errors='replace')
        else:
            data = pd.read_csv(filename + '.csv', sep=',',header=0,on_bad_lines='skip',skiprows=[1],encoding_errors='replace')
        n_calc = IntSlider(value=20,min=0,max=100,description='N points:',continuous_update=False,readout=True)   
        n_plot = IntSlider(value=4,min=0,max=4,description='N plot:',continuous_update=False,readout=True)   
        m_top = FloatText(value=50000,description='Top m. (kg):',disabled=False)
        L = FloatText(value=100,description='H (m):',disabled=False)
        def n_calc_change(change):
            if n_calc.value<4:
                n_plot.max = n_calc.value

        n_calc.observe(n_calc_change, names='value')

        diagrams = widgets.interactive_output(ModesPlot, {'h':fixed(np.array(data.iloc[:,0])),
                                                          'mass':fixed(np.array(data.iloc[:,1])),
                                                          'stiff':fixed(np.array(data.iloc[:,2])),
                                                          'N':n_calc,'n_plot':n_plot,'L':L,'mtop':m_top})
        display(VBox([HBox([n_calc,m_top,L]),n_plot,diagrams]))   

        return

class OpenFAST:
    '''
    Set of fucntions prepared to help the analysis of OpenFAST output files.
    '''
    def __init__(self,FileName,tmin=0,label=''):
        if FileName[-4:] == '.out':
            self.name = FileName[:-4]
        else:
            self.name = FileName
        if label == '':
            self.label = self.name
        else:
            self.label = label
        try:
            self.get_results(description=0)
            self.clean_data(tmin)
            print('Automatically loaded OpenFAST simulation results.')
            print('File description: ' + self.description)
        except:
            print('No data file available. Upload the .out file and try get_results() again.')

    def clean_data(self,tmin):
        self.data = self.data[self.data['Time']>=tmin].reset_index()
        self.data.drop('index',inplace=True,axis=1)

        self.time = self.data['Time']
        
        if 'Wind1VelX' in self.data.columns:
            self.wind = self.ExploreData(self.data[['Time','Wind1VelX']],'Wind1VelX','m/s',self.fs)
        if 'RotSpeed' in self.data.columns:
            self.rpm = self.ExploreData(self.data[['Time','RotSpeed']],'RotSpeed','RPM',self.fs)
        if 'RotTorq' in self.data.columns:
            self.rt = self.ExploreData(self.data[['Time','RotTorq']],'RotTorq','kN.m',self.fs) 
        if 'RotThrust' in self.data.columns:
            self.ft = self.ExploreData(self.data[['Time','RotThrust']],'RotThrust','kN',self.fs)            
        if 'BldPitch1' in self.data.columns:
            self.pitch = self.ExploreData(self.data[['Time','BldPitch1']],'BldPitch1','degrees',self.fs)
        if 'GenPwr' in self.data.columns:
            self.power = self.ExploreData(self.data[['Time','GenPwr']],'GenPwr','MW',self.fs)
        if 'TwrBsMyt' in self.data.columns:
            self.my = self.ExploreData(self.data[['Time','TwrBsMyt']],'TwrBsMyt','kN.m',self.fs)
        if 'TwrBsMxt' in self.data.columns:
            self.mx = self.ExploreData(self.data[['Time','TwrBsMxt']],'TwrBsMxt','kN.m',self.fs)
        if 'YawBrTAxp' in self.data.columns:
            self.ax = self.ExploreData(self.data[['Time','YawBrTAxp']],' YawBrTAxp','m/s^2',self.fs)      
        if 'YawBrTAyp' in self.data.columns:
            self.ay = self.ExploreData(self.data[['Time','YawBrTAyp']],' YawBrTAyp','m/s^2',self.fs)             
        if 'YawBrTDxp' in self.data.columns:
            self.dx = self.ExploreData(self.data[['Time','YawBrTDxp']],' YawBrTDxp','m',self.fs)      
        if 'YawBrTDyp' in self.data.columns:
            self.dy = self.ExploreData(self.data[['Time','YawBrTDyp']],' YawBrTDyp','m',self.fs)
        
        self.stats(tmin)
        if tmin>0:
            print('Deleted the first %d s of the simulation'%tmin)

    def get_results(self,description=1):
        self.description = pd.read_csv(self.name + '.out' , skiprows=[0,1,2,3] , nrows=1,header=None).iloc[0,0][37:]
        self.data = pd.read_csv(self.name + '.out' , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
        self.units = pd.read_csv(self.name + '.out' , skiprows=[0,1,2,3,4,5,6] , nrows=0,delimiter=r"\s+",header=0).columns
        self.fs = 1/(self.data['Time'][1] - self.data['Time'][0])
        self.n_channel = self.data.shape[1]
        self.time = self.data['Time']
        self.channels = self.data.columns[:]
        
        if 'Wind1VelX' in self.data.columns:
            self.wind = self.ExploreData(self.data[['Time','Wind1VelX']],'Wind1VelX','m/s',self.fs)
        if 'RotSpeed' in self.data.columns:
            self.rpm = self.ExploreData(self.data[['Time','RotSpeed']],'RotSpeed','RPM',self.fs)
        if 'RotTorq' in self.data.columns:
            self.rt = self.ExploreData(self.data[['Time','RotTorq']],'RotTorq','kN.m',self.fs) 
        if 'RotThrust' in self.data.columns:
            self.ft = self.ExploreData(self.data[['Time','RotThrust']],'RotThrust','kN',self.fs)            
        if 'BldPitch1' in self.data.columns:
            self.pitch = self.ExploreData(self.data[['Time','BldPitch1']],'BldPitch1','degrees',self.fs)
        if 'GenPwr' in self.data.columns:
            self.power = self.ExploreData(self.data[['Time','GenPwr']],'GenPwr','MW',self.fs)
        if 'TwrBsMyt' in self.data.columns:
            self.my = self.ExploreData(self.data[['Time','TwrBsMyt']],'TwrBsMyt','kN.m',self.fs)
        if 'TwrBsMxt' in self.data.columns:
            self.mx = self.ExploreData(self.data[['Time','TwrBsMxt']],'TwrBsMxt','kN.m',self.fs)
        if 'YawBrTAxp' in self.data.columns:
            self.ax = self.ExploreData(self.data[['Time','YawBrTAxp']],' YawBrTAxp','m/s^2',self.fs)      
        if 'YawBrTAyp' in self.data.columns:
            self.ay = self.ExploreData(self.data[['Time','YawBrTAyp']],' YawBrTAyp','m/s^2',self.fs)             
        if 'YawBrTDxp' in self.data.columns:
            self.dx = self.ExploreData(self.data[['Time','YawBrTDxp']],' YawBrTDxp','m',self.fs)      
        if 'YawBrTDyp' in self.data.columns:
            self.dy = self.ExploreData(self.data[['Time','YawBrTDyp']],' YawBrTDyp','m',self.fs)
        
        self.stats()
        if description == 1:
            print('File description: ' + self.description)
        
    def compute_ps(self , channel = ['all'] , nfft=4096):
        '''
        Compute the welch power spectrum for requested channels.

        Parameters
        ----------
        nfft : TYPE, optional
            Number of points to consider in the fft computation. The default is 4096.
        channel : TYPE, optional
            Channels to compute. The default is 'all'.
        '''
        if channel[0] == 'all':
            channels = self.data.columns[1:]
        elif type(channel)==str:
            channels = [channel]        
        else:
            channels = list(channel)
            
        for j in channels:
            y = self.data[j]            
            f, Pxx = signal.welch(y, self.fs , nperseg=nfft , scaling='spectrum')
            if j==channels[0]:
                frf = np.zeros((len(f),self.n_channel))
            frf[:,np.where(self.data.columns == j)[0][0]] = Pxx
        self.ps = pd.DataFrame(frf)
        self.ps.columns = self.data.columns
        self.ps['Time'] = f
        self.ps.rename(columns={'Time': 'Frequency'}, inplace=True) 
        
        if 'Wind1VelX' in channels:
            self.wind.ps = self.ps[['Frequency','Wind1VelX']]
        if 'RotSpeed' in channels:
            self.rpm.ps = self.ps[['Frequency','RotSpeed']]
        if 'RotTorq' in channels:
            self.rt.ps = self.ps[['Frequency','RotTorq']]
        if 'RotThrust' in channels:
            self.ft.ps = self.ps[['Frequency','RotThrust']]          
        if 'BldPitch1' in channels:
            self.pitch.ps = self.ps[['Frequency','BldPitch1']]
        if 'GenPwr' in channels:
            self.power.ps = self.ps[['Frequency','GenPwr']]
        if 'TwrBsMyt' in channels:
            self.my.ps = self.ps[['Frequency','TwrBsMyt']]
        if 'TwrBsMxt' in channels:
            self.mx.ps = self.ps[['Frequency','TwrBsMxt']]
        if 'YawBrTAxp' in channels:
            self.ax.ps = self.ps[['Frequency','YawBrTAxp']]
        if 'YawBrTAyp' in channels:
            self.ay.ps = self.ps[['Frequency','YawBrTAyp']]
        if 'YawBrTDxp' in channels:
            self.dx.ps = self.ps[['Frequency','YawBrTDxp']]
        if 'YawBrTDyp' in channels:
            self.dy.ps = self.ps[['Frequency','YawBrTDyp']]            
                     
    def HPMethod(self , channel):
        '''
        Interactive application of the half power method to estimate the damping coefficient.

        Parameters
        ----------
        channel : TYPE
            Channels to consider.
        '''
        if type(channel) == list:
            FrequencyDampingCoeffEst(self.ps['Frequency'],self.ps[channel],N=len(channel))
        else:
            FrequencyDampingCoeffEst(self.ps['Frequency'],self.ps[channel],N=1)
    
    def FDEnv(self,channel):
        '''
        Interactive application of the free decay envelope method to estimate the damping coefficent.

        Parameters
        ----------
        channel : TYPE
            Channels to consider.
        '''
        if type(channel) == list:
            FreeDecayDampingCoeffEst(self.data['Time'],self.data[channel],N=len(channel))
        else:
            FreeDecayDampingCoeffEst(self.data['Time'],self.data[channel],N=1)
    
    def plot_ps(self,channel=['all'],fmin=0,fmax=None):
        '''
        Interactive plot of the power spectrum.

        Parameters
        ----------
        channel : TYPE, optional
            Channels to consider. The default is ['all'].
        fmax : TYPE, optional
            Maximum frequency range of the plot. The default is None.
        '''
        if channel[0] == 'all':
            y = np.array(self.ps.iloc[:,1:])
            titles = self.data.columns[1:]
            N = self.n_channel - 1
        elif type(channel)==list:
            y = np.array(self.ps[channel])
            titles = [c for c in channel]
            N = len(channel)
        else:
            y = np.zeros((len(self.ps[channel]),1))
            y[:,0] = np.array(self.ps[channel])
            titles = [c for c in [channel]]
            N = len([channel])            
        if fmax == None:
            interact(FrequencyPlot,
                     fmin = widgets.FloatSlider(min=fmin, max=self.ps['Frequency'].iloc[-1], step=0.1, value=0,description='f. min. (Hz):',readout=True,continuous_update=False),
                     fmax = widgets.FloatSlider(min=fmin, max=self.ps['Frequency'].iloc[-1], step=0.1, value=self.ps['Frequency'].iloc[-1],description='f max. (Hz):',readout=True,continuous_update=False),
                     f = fixed(self.ps['Frequency']),y = fixed(y),
                     title = fixed(titles), N = fixed(N))  
        else:
            interact(FrequencyPlot,
                     fmin = widgets.FloatSlider(min=fmin, max=fmax, step=0.1, value=0,description='f. min. (Hz):',readout=True,continuous_update=False),
                     fmax = widgets.FloatSlider(min=fmin, max=fmax, step=0.1, value=fmax,description='f max. (Hz):',readout=True,continuous_update=False),
                     f = fixed(self.ps['Frequency']),y = fixed(y),
                     title = fixed(titles), N = fixed(N))  
    
    def stats(self,tmin=0,tmax=None):
        '''
        Compute general statistics of the data available. 
        Computed values are assigned to var.mean and var.std properties

        Parameters
        ----------
        tmin : TYPE, optional
            First time instant to consider. The default is 0.
        tmax : TYPE, optional
            Last time instant to consider. The default is None.
        '''
        if tmax == None:
            self.mean = np.mean(self.data[self.data['Time']>=tmin])
            self.std = np.std(self.data[self.data['Time']>=tmin])
            self.max = np.max(self.data[self.data['Time']>=tmin])
            self.min = np.min(self.data[self.data['Time']>=tmin])
        else:
            self.mean = np.mean(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)])
            self.std = np.std(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)])
            self.max = np.max(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)])
            self.min = np.min(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)])

        if 'Wind1VelX' in self.data.columns:
            self.wind.mean = self.mean['Wind1VelX']
            self.wind.std = self.std['Wind1VelX']
            self.wind.max = self.max['Wind1VelX']
            self.wind.min = self.min['Wind1VelX']

        if 'RotSpeed' in self.data.columns:
            self.rpm.mean = self.mean['RotSpeed']
            self.rpm.std = self.std['RotSpeed']
            self.rpm.max = self.max['RotSpeed']
            self.rpm.min = self.min['RotSpeed']

        if 'RotTorq' in self.data.columns:
            self.rt.mean = self.mean['RotTorq']
            self.rt.std = self.std['RotTorq']
            self.rt.max = self.max['RotTorq']
            self.rt.min = self.min['RotTorq']

        if 'RotThrust' in self.data.columns:
            self.ft.mean = self.mean['RotThrust']
            self.ft.std = self.std['RotThrust']    
            self.ft.max = self.max['RotThrust']
            self.ft.min = self.min['RotThrust']    

        if 'BldPitch1' in self.data.columns:
            self.pitch.mean = self.mean['BldPitch1']
            self.pitch.std = self.std['BldPitch1']
            self.pitch.max = self.max['BldPitch1']
            self.pitch.min = self.min['BldPitch1']

        if 'GenPwr' in self.data.columns:
            self.power.mean = self.mean['GenPwr']
            self.power.std = self.std['GenPwr']
            self.power.max = self.max['GenPwr']
            self.power.min = self.min['GenPwr']

        if 'TwrBsMyt' in self.data.columns:
            self.my.mean = self.mean['TwrBsMyt']
            self.my.std = self.std['TwrBsMyt']
            self.my.max = self.max['TwrBsMyt']
            self.my.min = self.min['TwrBsMyt']

        if 'TwrBsMxt' in self.data.columns:
            self.mx.mean = self.mean['TwrBsMxt']
            self.mx.std = self.std['TwrBsMxt']
            self.mx.max = self.max['TwrBsMxt']
            self.mx.min = self.min['TwrBsMxt']

        if 'YawBrTAxp' in self.data.columns:
            self.ax.mean = self.mean['YawBrTAxp']
            self.ax.std = self.std['YawBrTAxp']   
            self.ax.max = self.max['YawBrTAxp']
            self.ax.min = self.min['YawBrTAxp']  

        if 'YawBrTAyp' in self.data.columns:
            self.ay.mean = self.mean['YawBrTAyp']
            self.ay.std = self.std['YawBrTAyp']    
            self.ay.max = self.max['YawBrTAyp']
            self.ay.min = self.min['YawBrTAyp'] 

        if 'YawBrTDxp' in self.data.columns:
            self.dx.mean = self.mean['YawBrTDxp']
            self.dx.std = self.std['YawBrTDxp']    
            self.dx.max = self.max['YawBrTDxp']
            self.dx.min = self.min['YawBrTDxp']  

        if 'YawBrTDyp' in self.data.columns:
            self.dy.mean = self.mean['YawBrTDyp']
            self.dy.std = self.std['YawBrTDyp']
            self.dy.max = self.max['YawBrTDyp']
            self.dy.min = self.min['YawBrTDyp']

    def plot(self,channel=['all'],tmin=0,tmax=None):
        '''
        Interactive plot of the time series.

        Parameters
        ----------
        channel : TYPE, optional
            Channels to consider. The default is ['all'].
        tmax : TYPE, optional
            Maximum time instant of the plot. The default is None.
        '''
        if channel[0] == 'all':
            y = np.array(self.data.iloc[:,1:])
            titles = self.data.columns[1:]
            units = self.units[1:]
            N = self.n_channel - 1
        elif type(channel)==list:
            y = np.array(self.data[channel])
            titles = [c for c in channel]
            units = [self.units[self.data.columns.get_loc(c)] for c in channel]
            N = len(channel)
        else:
            y = np.zeros((len(self.data[channel]),1))
            y[:,0] = np.array(self.data[channel])

            titles = [c for c in [channel]]
            units = [self.units[self.data.columns.get_loc(c)] for c in [channel]]
            N = len([channel])            
        if tmax == None:         
            interact(TimePlot,
                     tmin = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=self.data['Time'].iloc[-1], step=0.1, value=0,description='t min. (s):',readout=True,continuous_update=False),
                     tmax = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=self.data['Time'].iloc[-1], step=0.1, value=self.data['Time'].iloc[-1],description='t max. (s):',readout=True,continuous_update=False),
                     t = fixed(self.data['Time']),y = fixed(y),
                     title = fixed(titles), units = fixed(units) , N = fixed(N)) 
        else:    
            interact(TimePlot,
                     tmin = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=tmax, step=0.1, value=0,description='t min. (s):',readout=True,continuous_update=False),
                     tmax = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=tmax, step=0.1, value=tmax,description='t max. (s):',readout=True,continuous_update=False),
                     t = fixed(self.data['Time']),y = fixed(y),
                     title = fixed(titles), units = fixed(units) , N = fixed(N))   
        
    class ExploreData:
        def __init__(self,data,title,units,fs):
            self.data = data
            self.title = title
            self.units = units
            self.fs = fs
            self.ps = []
            self.stats()
        
        def stats(self,tmin=0,tmax=None):
            if tmax == None:
                self.mean = np.mean(self.data[self.data['Time']>=tmin].iloc[:,1])
                self.std = np.std(self.data[self.data['Time']>=tmin].iloc[:,1])
                self.max = np.max(self.data[self.data['Time']>=tmin].iloc[:,1])
                self.min = np.min(self.data[self.data['Time']>=tmin].iloc[:,1])
            else:
                self.mean = np.mean(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)].iloc[:,1])
                self.std = np.std(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)].iloc[:,1])
                self.max = np.max(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)].iloc[:,1])
                self.min = np.min(self.data[(self.data['Time']>=tmin) & (self.data['Time']<=tmax)].iloc[:,1])

        def plot_ps(self,fmin=0,fmax=None):
            y = np.zeros((len(self.ps),1))
            y[:,0] = np.array(self.ps.iloc[:,1])
            if fmax == None:
                interact(FrequencyPlot,
                         fmin = widgets.FloatSlider(min=fmin, max=self.ps['Frequency'].iloc[-1], step=0.1, value=0,description='f. min. (Hz):',readout=True,continuous_update=False),
                         fmax = widgets.FloatSlider(min=fmin, max=self.ps['Frequency'].iloc[-1], step=0.1, value=self.ps['Frequency'].iloc[-1],description='f max. (Hz):',readout=True,continuous_update=False),
                         f = fixed(self.ps['Frequency']),y = fixed(y),
                         title = fixed([self.title]), N = fixed(1))  
            else:
                interact(FrequencyPlot,
                         fmin = widgets.FloatSlider(min=fmin, max=fmax, step=0.1, value=0,description='f. min. (Hz):',readout=True,continuous_update=False),
                         fmax = widgets.FloatSlider(min=fmin, max=fmax, step=0.1, value=fmax,description='f max. (Hz):',readout=True,continuous_update=False),
                         f = fixed(self.ps['Frequency']),y = fixed(y),
                         title = fixed([self.title]), N = fixed(1))  
                
        def plot(self,channel=['all'],tmin=0,tmax=None):
            y = np.zeros((len(self.data),1))
            y[:,0] = np.array(self.data.iloc[:,1])           
            if tmax == None:
                interact(TimePlot,
                         tmin = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=self.data['Time'].iloc[-1], step=0.1, value=0,description='t min. (s):',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=self.data['Time'].iloc[-1], step=0.1, value=self.data['Time'].iloc[-1],description='t max. (s):',readout=True,continuous_update=False),
                         t = fixed(self.data['Time']),y = fixed(y),
                         title = fixed([self.title]), units = fixed([self.units]) , N = fixed(1))    
            else:
                interact(TimePlot,
                         tmin = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=tmax, step=0.1, value=0,description='t min. (s):',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=max(tmin,self.data['Time'].iloc[0]), max=tmax, step=0.1, value=tmax,description='t max. (s):',readout=True,continuous_update=False),
                         t = fixed(self.data['Time']),y = fixed(y),
                         title = fixed([self.title]), units = fixed([self.units]) , N = fixed(1))                 
            
        def compute_ps(self , nfft=4096):
            y = self.data.iloc[:,1]
            f, Pxx = signal.welch(y, self.fs , nperseg=nfft , scaling='spectrum')
            self.ps = pd.DataFrame(f)
            self.ps.rename(columns={0: 'Frequency'}, inplace=True) 
            self.ps[self.title] = Pxx            

        def HPMethod(self):
            FrequencyDampingCoeffEst(self.ps['Frequency'],self.ps.iloc[:,1])

        def FDEnv(self):
            FreeDecayDampingCoeffEst(self.data['Time'],self.data.iloc[:,1])
            
def TimePlot(t,y,tmin=0,tmax=None,title='',units='',N=1):    
    n_rows = int((N-1)/3)+1
    n_col = min(N,3)
    plt.figure(figsize = (5*n_col,4*n_rows))
    gs = gridspec.GridSpec(n_rows,n_col)
    gs.update(hspace=0.3,wspace=0.3)
    for i in range(N):
        ymin = min(y[(t>=tmin) & (t<=tmax),i])
        ymax = max(y[(t>=tmin) & (t<=tmax),i])
        if ymin == ymax:
            ymax = ymin+1
            ymin = ymin-1
        row = int(i/3)
        column = i-row*3
        ax = plt.subplot(gs[row,column])
        plt.plot(t,y[:,i]) 
        plt.xlim(tmin,tmax)
        plt.ylim(ymin,ymax)
        plt.xlabel('time (s)') 
        if '(' in units[i]:
            unit = units[i][1:(units[i].find(')'))]
            plt.ylabel(unit)
        else:
            plt.ylabel(units[i])
        plt.title(title[i])

def FrequencyPlot(f,y,fmin=0,fmax=None,title='',N=1):    
    n_rows = int((N-1)/3)+1
    n_col = min(N,3)
    plt.figure(figsize = (5*n_col,4*n_rows))
    gs = gridspec.GridSpec(n_rows,n_col)
    gs.update(hspace=0.3,wspace=0.3)
    for i in range(N):
        ymin = min(y[(f>=fmin) & (f<=fmax),i])/2
        ymax = max(y[(f>=fmin) & (f<=fmax),i])*2
        if ymin == ymax:
            ymax = ymin+1
            ymin = ymin-1
        row = int(i/3)
        column = i-row*3
        ax = plt.subplot(gs[row,column])
        plt.semilogy(f,y[:,i]) 
        plt.xlim(fmin,fmax)
        plt.ylim(ymin,ymax)
        plt.xlabel('Frequency (Hz)') 
        plt.ylabel('Spectral Amp.')
        plt.title(title[i])
         
def FrequencyDampingCoeffEst(f,frf,fmin=0,fmax=None,N=1):
    def frf_plot(f,frf,fmin=fmin,fmax=fmax):
        frf_filt = np.array(frf[(f>=fmin) & (f<=fmax)])
        f_filt = np.array(f[(f>=fmin) & (f<=fmax)])   

        plt.figure(figsize = (12,4))
        gs = gridspec.GridSpec(1,2)

        ax = plt.subplot(gs[0,0])
        ax_zoom = plt.subplot(gs[0,1])

        ax.semilogy(f_filt,frf_filt,'b')
        ax.set_xlabel('Frequency (Hz)')
        ax_zoom.set_xlabel('Frequency (Hz)')
        ax.set_xlim(fmin,fmax)
               
        filter1 = frf_filt[2:-2]>frf_filt[1:-3]
        filter2 = frf_filt[2:-2]>frf_filt[3:-1]

        frf_peak = max(frf_filt[2:-2][filter1 & filter2])
        f_peak = f_filt[2:-2][filter1 & filter2][np.argmax(frf_filt[2:-2][filter1 & filter2])] 
        ax.plot(f_peak,frf_peak,'or')
        
        filter_amp = (frf_filt<=frf_peak/4)
        delta_f = f_filt[filter_amp] - f_peak
        try:
            f_minus = max(f_filt[filter_amp][delta_f<=0])
            f_plus = min(f_filt[filter_amp][delta_f>=0])

            f_zoom = f_filt[(f_filt>=f_minus) & (f_filt<=f_plus)]
            frf_zoom = frf_filt[(f_filt>=f_minus) & (f_filt<=f_plus)]

            f_int = np.arange(f_minus,f_plus-0.0001,0.0001)
            frf_int = interp1d(f_zoom,frf_zoom,kind='slinear')
            amp_int = frf_int(f_int)
            
            frf_peak = max(frf_zoom)
            ax_zoom.plot(f_zoom,frf_zoom,'.b',label='Original')
            ax_zoom.plot(f_int,amp_int,'--b',label='Interpolated')
            ax_zoom.set_xlim(f_minus,f_plus)
            ax_zoom.set_ylim(frf_peak/4,frf_peak*1.2)

            idx_minus = np.argmin(abs((amp_int - frf_peak/2)[f_int<f_peak]))
            idx_plus = np.argmin(abs((amp_int - frf_peak/2)[f_int>f_peak]))

            ax_zoom.plot(f_int[f_int<f_peak][idx_minus],amp_int[f_int<f_peak][idx_minus],'or')
            ax_zoom.plot(f_int[f_int>f_peak][idx_plus],amp_int[f_int>f_peak][idx_plus],'or')
            ax_zoom.plot(f_int[np.argmax(amp_int)],frf_peak,'or')

            ax_zoom.plot([f_int[f_int<f_peak][idx_minus],f_int[f_int<f_peak][idx_minus]],[0,amp_int[f_int<f_peak][idx_minus]],'--k')
            ax_zoom.plot([f_int[f_int>f_peak][idx_plus],f_int[f_int>f_peak][idx_plus]],[0,amp_int[f_int>f_peak][idx_plus]],'--k')

            damping = 100*(f_int[f_int>f_peak][idx_plus]-f_int[f_int<f_peak][idx_minus])/(f_int[f_int>f_peak][idx_plus]+f_int[f_int<f_peak][idx_minus])

            ax_zoom.plot([f_minus,f_int[f_int>f_peak][idx_plus]],[frf_peak/2,frf_peak/2],'--k') 
            ax_zoom.plot([f_minus,f_int[np.argmax(amp_int)]],[frf_peak,frf_peak],'--k') 
            ax_zoom.set_title(r'$\xi$ = %.2f %s'%(damping,'%'))

            ax_zoom.legend()
            ax_zoom.set_xticks([f_int[f_int<f_peak][idx_minus],f_int[f_int>f_peak][idx_plus]])
            ax_zoom.set_yticks(ticks=[frf_peak/2,frf_peak])
            ax_zoom.set_yticklabels([0.5,1])
            ax_zoom.set_ylim(0,frf_peak*1.2)
        except:
            pass
    if N>1:
        for i in range(N):
            interact(frf_plot,
                     fmin = widgets.FloatSlider(min=0, max=5, step=0.1, value=0,description='F min. (Hz):',readout=True,continuous_update=False),
                     fmax = widgets.FloatSlider(min=0, max=5, step=0.1, value=1,description='F max. (Hz):',readout=True,continuous_update=False),
                     f=fixed(f),frf=fixed(np.array(frf)[:,i]))
    else:
        interact(frf_plot,
                 fmin = widgets.FloatSlider(min=0, max=5, step=0.1, value=0,description='F min. (Hz):',readout=True,continuous_update=False),
                 fmax = widgets.FloatSlider(min=0, max=5, step=0.1, value=1,description='F max. (Hz):',readout=True,continuous_update=False),
                 f=fixed(f),frf=fixed(frf))

def FreeDecayDampingCoeffEst(t,y,N=1):
    '''
    FreeDecayDampingCoeffEst computes the damping coefficient from the signal envelope in time domain:
    - The Hilbert transform is computed.
    - A log-linear fit is used to obtain the damping times natural frequency value
    - The welch power spectrum of the time series is used to identify the system natural frequency
    
    Parameters
    ----------
    t : array_like
        Time array.
    y : array_like
        System response.
    '''  
    def time_plot(t,y,t1,t2,tmax):
        from scipy.signal import hilbert
        f, Pxx = signal.welch(y, 1/(t[1]-t[0]) , nperseg=len(y) , scaling='spectrum')
        f_peak = f[np.argmax(Pxx)]
        
        test = abs(hilbert(y))
        m,_,b,_=LinearFitCoefficients(t[(t>=t1) & (t<=t2)],np.log(abs(test)[(t>=t1) & (t<=t2)]))
        
        plt.figure(figsize = (16,4))
        gs = gridspec.GridSpec(1,3)
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[0,2])
        ax3 = plt.subplot(gs[0,0])
        
        ax1.plot(t,y,'b') 
        ax1.plot(t[(t>=t1) & (t<=t2)],test[(t>=t1) & (t<=t2)],'r')
        ax1.plot(t[t<t1],test[(t<t1)],'--r')
        ax1.plot(t[t>t2],test[(t>t2)],'--r')
        ax1.plot(t[t>=t1][0],test[(t>=t1)][0],'or')
        ax1.plot(t[t<=t2][-1],test[(t<=t2)][-1],'or')

        ax1.set_xlim(0,tmax)
        ax1.set_ylim(min(y[t<=tmax]),max(y[t<=tmax]))
        ax1.set_xlabel('time (s)')
    
        ax2.plot(t[(t>=t1) & (t<=t2)],np.log(abs(test)[(t>=t1) & (t<=t2)]),'r')
        ax2.plot(t[(t<t1)],np.log(abs(test)[(t<t1)]),'r',alpha=0.2)
        ax2.plot(t[(t>t2)],np.log(abs(test)[(t>t2)]),'r',alpha=0.2)
        ax2.plot(t[(t>=t1) & (t<=t2)],b + m*t[(t>=t1) & (t<=t2)],'--k')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Envelope Amplitude')
        ax2.set_xlim(0,tmax)
        ax2.set_ylim(min(np.log(abs(test)[(t<=tmax)])),max(np.log(abs(test)[(t<=tmax)])))
        ax2.set_title(r"-m=$\xi\omega$=%.4f$\longrightarrow\xi=$%.2f %s"%(-m,-100*m/(2*np.pi*f_peak),'%'))
        ax3.semilogy(f,Pxx,'b')
        
        ax3.set_xlim(0,f_peak*2)
        ax3.set_ylim(min(Pxx[f<=f_peak])/2,max(Pxx)*2)
        ax3.semilogy(f[np.argmax(Pxx)],max(Pxx),'or')
        ax3.semilogy([f_peak,f_peak],[min(Pxx)/2,max(Pxx)],'--k')
        ax3.set_xticks([f[np.argmax(Pxx)]])
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_title('Power spectrum')
        
        ax1.set_title('Free decay')
    
    if N>1:
        for i in range(N):
            interact(time_plot,
                     t1 = widgets.FloatSlider(min=0, max=np.array(t)[-1], step=0.1, value=0,description='First point (s):',readout=True,continuous_update=False),
                     t2 = widgets.FloatSlider(min=0, max=np.array(t)[-1], step=0.1, value=np.array(t)[-1]/2,description='Last point (s):',readout=True,continuous_update=False),
                     tmax = widgets.FloatSlider(min=0, max=np.array(t)[-1], step=0.1, value=np.array(t)[-1],description='Time limit (s):',readout=True,continuous_update=False),
                     t=fixed(np.array(t)),y=fixed(np.array(y)[:,i]))    
    else:
        interact(time_plot,
                t1 = widgets.FloatSlider(min=0, max=np.array(t)[-1], step=0.1, value=0,description='First point (s):',readout=True,continuous_update=False),
                t2 = widgets.FloatSlider(min=0, max=np.array(t)[-1], step=0.1, value=np.array(t)[-1]/2,description='Last point (s):',readout=True,continuous_update=False),
                tmax = widgets.FloatSlider(min=0, max=np.array(t)[-1], step=0.1, value=np.array(t)[-1],description='Time limit (s):',readout=True,continuous_update=False),
                t=fixed(np.array(t)),y=fixed(np.array(y)))    
            
def LinearFitCoefficients(x,y):
    n = len(x)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x**2)
    Syy = np.sum(y**2)
    Sxy = np.sum(x*y)
    m = (n*Sxy-Sx*Sy)/(n*Sxx-Sx**2)
    b = Sy/n - m/n*Sx

    Se2 = 1/(n*(n-2)) * (n*Syy - Sy**2 - m**2*(n*Sxx-Sx**2))
    Sm2 = n*Se2/(n*Sxx-Sx**2)
    Sb2 = 1/n * Sm2 * Sxx
    
    return m,np.sqrt(Sm2),b,np.sqrt(Sb2)

def HilbertTransform_App(A=1,f=1,b=0.01,tmax=100,t1=0,t2=100):  
    Damping = b/100
    w = (2*np.pi)*f
    wd = w*np.sqrt(1-Damping**2)
    fa = np.linspace(0, 2,1000)
    wa = (2*np.pi)*fa
    t = np.linspace(0, tmax,1000)
    u_harmonic = np.cos(wd*t)
    u_decay = A*np.exp(-w*Damping*t)
    u = u_harmonic * u_decay
    test = abs(hilbert(u))
    m,_,b,_ = LinearFitCoefficients(t[(t>=t1) & (t<=t2)],np.log(abs(test)[(t>=t1) & (t<=t2)]))
    
    plt.figure(figsize = (12,4))
    gs = gridspec.GridSpec(1,2)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    
    ax1.plot(t,u,'b') 
    ax1.plot(t[(t>=t1) & (t<=t2)],test[(t>=t1) & (t<=t2)],'r')
    ax1.plot(t[t<t1],test[(t<t1)],'--r')
    ax1.plot(t[t>t2],test[(t>t2)],'--r')
    ax1.plot(t[t>=t1][0],test[(t>=t1)][0],'or')
    ax1.plot(t[t<=t2][-1],test[(t<=t2)][-1],'or')

    ax1.set_xlim(0,tmax)
    ax1.set_ylim(-A,A)
    ax1.set_xlabel('time (s)')
    
    ax2.plot(t[(t>=t1) & (t<=t2)],np.log(abs(test)[(t>=t1) & (t<=t2)]),'r')
    ax2.plot(t[(t>=t1) & (t<=t2)],b + m*t[(t>=t1) & (t<=t2)],'--k')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Envelope Amplitude')
    ax2.set_xlim(t1,t2)

def FRF_App(A=1,f=1,b=0.01,tmax=100):
    Damping = b/100
    w = (2*np.pi)*f
    wd = w*np.sqrt(1-Damping**2)
    fa = np.linspace(0, 2,1000)
    wa = (2*np.pi)*fa
    t = np.linspace(0, tmax,1000)
    u_harmonic = np.cos(wd*t)
    u_decay = A*np.exp(-w*Damping*t)
    frf = 1/(w**2-wa**2+2*w*wa*Damping*1j)
    
    plt.figure(figsize = (12,4))
    gs = gridspec.GridSpec(1,2)
    ax_t = plt.subplot(gs[0,0])
    ax_f = plt.subplot(gs[0,1])
    
    ax_t.plot(t,u_harmonic*u_decay,'b') 
    ax_t.plot(t,u_decay,'--r')  
    ax_t.plot(t,-u_decay,'--r') 
    ax_t.set_xlim(0,tmax)
    ax_t.set_ylim(-A,A)
    ax_t.set_xlabel('time (s)')
    
    ax_f.semilogy(fa,np.abs(frf),'b')
    ax_f.set_xlabel('Frequency (Hz)')
    
    ax_p = ax_f.twinx()
    ax_p.plot(fa,np.angle(frf),'r')

    ax_f.set_ylabel('FRF Ampltude', color='b')
    ax_p.set_ylabel('FRF phase (rad)', color='r')
    
    ax_f.set_ylim(max(min(abs(frf)),abs(frf[0])*1e-3),max(abs(frf))*1e1)
    ax_f.set_xlim(0,2)
    ax_p.set_ylim(-pi,0)
        
    ax_p.set_yticks(ticks=[0,-pi/4,-pi/2,-3*pi/4,-pi])
    ax_p.set_yticklabels(['0','$-\pi/4$','-$\pi/2$','-$3\pi/4$','-$\pi$'])


def MultiPlot(x_points,y_points,tmin=0,tmax=None,N=1,N_y=1,x='',x_units='',y='',y_units='',PlotStats=1,labels=''):
    n_rows = int((N_y-1)/3)+1
    n_col = min(N_y,3)
    
    if (PlotStats == 1) & (not (x=='Frequency')):       
        plt.figure(figsize = (12,4*N_y))
        gs = gridspec.GridSpec(N_y,2)
        gs.update(hspace=0.3,wspace=0.3)
        
        for j in range(N_y):
            ax1 = plt.subplot(gs[j,0])
            ax2 = plt.subplot(gs[j,1])

            ax1.set_title('Full series')
            ax1.set_xlabel('%s %s'%(x,x_units))
            ax1.set_ylabel('%s %s'%(y[j],y_units[j]))

            ax2.set_title('Mean values')
            ax2.set_xlabel('%s %s'%(x,x_units))
            ax2.set_ylabel('%s %s'%(y[j],y_units[j]))

            for i in range(N):
                ax1.plot(x_points[i],y_points[i*N_y+j],label=labels[i])
                ax2.errorbar(np.mean(x_points[i][(x_points[i]>=tmin) & (x_points[i]<=tmax)]),
                             np.mean(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)]),
                             yerr=np.std(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)]),
                             fmt='o',
                             label=labels[i],
                             ecolor='k',
                             elinewidth=1,
                             capsize=1,
                            )
                if (i==0):
                    ymin = min(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)])
                    ymax = max(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)])
                else:
                    ymin = min(ymin,min(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)]))
                    ymax = max(ymax,max(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)]))
            if ymin == ymax:
                ymin -= 1
                ymax+= 1
                
            ax1.set_xlim(tmin,tmax)   
            ax1.set_ylim(ymin,ymax)  
            
        ax1.legend(loc='upper center',
                   bbox_to_anchor=(1,-0.25),
                   ncol=min(10,N),
                   fancybox=False,
                   frameon=False)           
            
    else:        
        plt.figure(figsize = (6*n_col,4*n_rows))
        gs = gridspec.GridSpec(n_rows,n_col)
        gs.update(hspace=0.3,wspace=0.3)
        
        for j in range(N_y):
            row = int(j/3)
            column = j-row*3
            ax = plt.subplot(gs[row,column])

            ax.set_title(y[j])
            ax.set_xlabel('%s %s'%(x,x_units))
            ax.set_ylabel('%s'%(y_units[j]))

            for i in range(N):
                ax.plot(x_points[i],y_points[i*N_y+j],label=labels[i])
                if (i==0):
                    ymin = min(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)])
                    ymax = max(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)])
                else:
                    ymin = min(ymin,min(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)]))
                    ymax = max(ymax,max(y_points[i*N_y+j][(x_points[i]>=tmin) & (x_points[i]<=tmax)]))
            if ymin == ymax:
                ymin -= 1
                ymax+= 1
                
            ax.set_xlim(tmin,tmax)   
            ax.set_ylim(ymin,ymax) 
            
            if row == (n_rows-1):
                if (column == 1):
                    if (n_rows == 1) & (n_col == 2):
                        ax.legend(loc='upper center',
                                  bbox_to_anchor=(-0.15,-0.25),
                                  ncol=min(10,N),
                                  fancybox=False,
                                  frameon=False)                        
                    else:
                        ax.legend(loc='upper center',
                                  bbox_to_anchor=(0.5,-0.25),
                                  ncol=min(10,N),
                                  fancybox=False,
                                  frameon=False)
                elif (j == (N_y-1)) & (column==0) & (N_y>1):
                    ax.legend(loc='upper center',
                              bbox_to_anchor=(1.5,-0.25),
                              ncol=min(10,N),
                              fancybox=False,
                              frameon=False)   
                elif (j == (N_y-1)) & (column==0) & (N_y==1):
                    ax.legend(loc='upper center',
                              bbox_to_anchor=(0.5,-0.25),
                              ncol=min(10,N),
                              fancybox=False,
                              frameon=False)                      
            if x == 'Frequency':
                ax.set_yscale('log')
        
    
def ManySim(simulations,x,y,xmin_i=None,xmax_i=None,PlotStats=0):
    '''
    Interactive plot of different simulations in a single graph    

    Parameters
    ----------
    simulations : list
        List of OpenFAST class objects containing the simulations to plot.
    x : str
        Independent variable to consider.
    y : str or list
        Dependent(s) varibale to consider.
    PlotStats : switch. Optional
        If PlotStats = 1 a plot with the mean values and error bars representing the standard deviation will be produced.
        If PlotSats = 0 only the full series will be displayed. The default is PlotStats = 0.
    x_min_i : float. Optional
        Defines the independent variable lower limit to initiate the plot. 
        If nothing is passed it will be infered from the data.
    x_max_i : float. Optional
        Defines the independent variable upper limit to initiate the plot.
        If nothing is passed it will be infered from the data.
    '''
    N = len(simulations)
    x_points = []
    y_points = []
    labels = []
    
    if x == 'Frequency':
        if type(y) == str:
            N_y = 1
            y_label = [y]
        else:
            N_y = len(y)
            y_label = y

        x_units = '(Hz)'
        y_units = 'Spectral Amp.'
        for i in range(N):
            x_points.append(np.array(simulations[i].ps[x]))
            labels.append(simulations[i].label)
            if i==0:
                x_min = min(np.array(simulations[i].ps[x]))
            else:
                x_min = min(x_min,min(np.array(simulations[i].ps[x]))) 

            if type(y) == str:
                y_points.append(np.array(simulations[i].ps[y]))
            else:
                for j in range(N_y):
                    y_points.append(np.array(simulations[i].ps[y[j]]))
        x_max = xmax_i
        
        if xmin_i == None:
            if xmax_i == None:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_min,description='F Min. (Hz)',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_max,description='F Max. (Hz)',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels)) 
            else:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_min,description='F Min. (Hz)',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmax_i,description='F Max. (Hz)',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels))
        else:
            if xmax_i == None:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmin_i,description='F Min. (Hz)',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_max,description='F Max. (Hz)',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels)) 
            else:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmin_i,description='F Min. (Hz)',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmax_i,description='F Max. (Hz)',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels))     
    else:
        if type(y) == str:
            N_y = 1
            y_units = [[simulations[0].units[simulations[0].data.columns.get_loc(c)] for c in [y]][0]]
            y_label = [y]
        else:
            N_y = len(y)
            y_units = []
            for j in range(N_y):
                y_units.append([simulations[0].units[simulations[0].data.columns.get_loc(c)] for c in [y[j]]][0])
            y_label = y

        x_units = [simulations[0].units[simulations[0].data.columns.get_loc(c)] for c in [x]][0]

        for i in range(N):
            x_points.append(np.array(simulations[i].data[x]))
            labels.append(simulations[i].label)
            if i==0:
                x_min = min(np.array(simulations[i].data[x]))
                x_max = max(np.array(simulations[i].data[x]))
            else:
                x_min = min(x_min,min(np.array(simulations[i].data[x])))
                x_max = max(x_max,max(np.array(simulations[i].data[x])))   

            if type(y) == str:
                y_points.append(np.array(simulations[i].data[y]))
            else:
                for j in range(N_y):
                    y_points.append(np.array(simulations[i].data[y[j]]))

        if xmin_i == None:
            if xmax_i == None:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_min,description='First point:',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_max,description='Last point:',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels)) 
            else:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_min,description='First point:',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmax_i,description='Last point:',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels))
        else:
            if xmax_i == None:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmin_i,description='First point:',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=x_max,description='Last point:',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels)) 
            else:
                interact(MultiPlot,
                         tmin = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmin_i,description='First point:',readout=True,continuous_update=False),
                         tmax = widgets.FloatSlider(min=x_min, max=x_max, step=0.1, value=xmax_i,description='Last point:',readout=True,continuous_update=False),
                         x_points=fixed(x_points),x=fixed(x),x_units=fixed(x_units),
                         y=fixed(y_label),y_units=fixed(y_units),y_points=fixed(y_points),
                         N=fixed(N),N_y=fixed(N_y),PlotStats=fixed(PlotStats),labels=fixed(labels)) 