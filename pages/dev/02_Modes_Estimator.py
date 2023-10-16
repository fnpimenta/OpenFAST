import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import os 

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from io import StringIO
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ipywidgets import *
from IPython.display import display, HTML

pi = np.pi

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API - Modes estimator'
icon = Image.open(".\logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon , layout="wide")


# -- File type definition (tab1)
file_mode = ["📈 Upload file" , "📈 Select from database"]
page = st.sidebar.selectbox('Input data file', file_mode)

# -- Load data files
@st.cache_data()
def load_data(uploaded_files,n1,n2):
    try:
        data = pd.read_csv(uploaded_file,skiprows=n1,nrows=n2-n1,delimiter="\s+",encoding_errors='replace')
        error_check += 1
    except:
        tab1.write('Please select the file for analysis')
        data = 0

    return data 

if page == file_mode[0]:
    uploaded_file = st.sidebar.file_uploader("Choose a file",accept_multiple_files=False)

    count = 0
    for line in uploaded_file:
        count += 1
        if bytes("DISTRIBUTED", 'utf-8') in line:
            n1 = count
            if "TOWER" in line:
                element_type = 'tower'
            else:
                element_type = 'blade'
        if bytes('MODE SHAPES', 'utf-8') in line:
            n2 = count

    uploaded_file.seek(0)
    tower = pd.read_csv(uploaded_file,skiprows=np.concatenate((np.arange(n1),[n1+1])),nrows=n2-n1-3,delimiter='\s+',on_bad_lines='skip',encoding_errors='ignore')


else:
    # -- Load data files
    ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
    ref_model = st.sidebar.selectbox('Reference model', ref_models)
    ref_path = ref_models[ref_model]

    all_dir = os.listdir('./OpenFAST_models/' + ref_path )
    sel_dir = st.sidebar.selectbox('Available modules', all_dir)

    all_files = os.listdir('./OpenFAST_models/' + ref_path + '/' + sel_dir)
    uploaded_file = st.sidebar.selectbox('Available files', all_files)

    log = open('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + uploaded_file, 'r')
    count = 0
    for line in log:
        count += 1
        if 'DISTRIBUTED' in line:
            n1 = count
            if "TOWER" in line:
                element_type = 'tower'
            else:
                element_type = 'blade'
        if 'MODE SHAPES' in line:
            n2 = count

    tower = pd.read_csv('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + uploaded_file,skiprows=np.concatenate((np.arange(n1),[n1+1])),nrows=n2-n1-3,delimiter='\s+',on_bad_lines='skip',encoding_errors='ignore')




st.sidebar.write(tower)


def ModeFit(h,a2,a3,a4,a5):
    a6 = (1-a2-a3-a4-a5)
    return a2*h**2 + a3*h**3 + a4*h**4 + a5*h**5 + a6*h**6
    
def TowerModesPlot(h,mass,stiff,N=2,n_plot=2,L=100,mtop=0):
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
    
    fig = plt.figure(figsize = (12,12))
    gs = gridspec.GridSpec(3,2)
    gs.update(hspace=0,wspace=0.5)
    ax1 = plt.subplot(gs[0,0])
    ax3 = plt.subplot(gs[:-1,1])

    axl = plt.subplot(gs[-1,:])
    
    ax1.plot(rs_h,rs_m,'darkred',linewidth=1)
    for i in range(N):
        ax1.plot([points[i],points[i+1]],[mi[i],mi[i]],'r')
        if i<(N-1):
            ax1.plot([points[i+1],points[i+1]],[mi[i],mi[i+1]],'r')
    ax1.set_xlim(0,L)
    ax1.set_ylim(0)
    ax1.grid()
    
    ax2 = ax1.twinx()

    ax2.plot(rs_h,rs_s*1e-9,'darkblue',linewidth=1)
    for i in range(N):
        ax2.plot([points[i],points[i+1]],[si[i]*1e-9,si[i]*1e-9],'b')
        if i<(N-1):
            ax2.plot([points[i+1],points[i+1]],[si[i]*1e-9,si[i+1]*1e-9],'b')

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
        axl.plot(0,0,'--',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi)) 

    axl.axis('off')
    axl.legend(loc='upper left',
               bbox_to_anchor=(0,0.8),
               ncol=1,
               fancybox=False,
               framealpha=1,
               frameon=False,
               fontsize=14)

    st.pyplot(fig)  
    return 

def BladesModesPlot(h,mass,stiff_edge,stiff_wise,N=2,n_plot=2,L=100,mtop=0):
    seg = np.linspace(0,1,N+1)*L

    f_m = interp1d(h*L,mass)
    f_se = interp1d(h*L,stiff_edge)   
    f_sw = interp1d(h*L,stiff_wise)    
     
    points = np.linspace(0,1,N+1)*L

    mi = f_m(points)
    sie = f_se(points)
    siw = f_sw(points)
    
    mi = mi[1:]/2 + mi[:-1]/2
    sie = sie[1:]/2 + sie[:-1]/2
    siw = siw[1:]/2 + siw[:-1]/2

    rs_h = np.linspace(0,h[-1],1000)*L
    rs_m = f_m(rs_h)
    rs_se = f_se(rs_h)
    rs_sw = f_sw(rs_h)
    
    MassMatrix = np.zeros((2*N,2*N))
    for i in range(N-1):
        MassMatrix[i,i] = mi[i]*(points[i+1]-points[i])/2 + mi[i+1]*(points[i+2]-points[i+1])/2
        MassMatrix[N+i,N+i] = MassMatrix[i,i]

    MassMatrix[N-1,N-1] = mi[-1]*(points[-1]-points[-2])/2 + mtop
    MassMatrix[-1,-1] = mi[-1]*(points[-1]-points[-2])/2 + mtop
    
    StiffMatrix = np.zeros_like(MassMatrix)

    KMatrix = np.zeros((4*N,4*N))
    for i in range(N-1):
        L1 = points[i+1]-points[i]
        EI1 = sie[i]

        L2 = points[i+2]-points[i+1]
        EI2 = sie[i+1]
       
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


    KMatrix[2*N-2,2*N-2] += +12*EI2/L2**3
    KMatrix[2*N-2,2*N-1] +=  -6*EI2/L2**2    

    KMatrix[2*N-1,2*N-2] += -6*EI2/L2**2
    KMatrix[2*N-1,2*N-1] += +4*EI2/L2 

    for i in range(N-1):
        L1 = points[i+1]-points[i]
        EI1 = siw[i]

        L2 = points[i+2]-points[i+1]
        EI2 = siw[i+1]
       
        KMatrix[2*N+2*i,2*N+2*i] += 12*EI1/L1**3 + 12*EI2/L2**3
        KMatrix[2*N+2*i,2*N+2*i+1] += 6*EI2/L2**2 - 6*EI1/L1**2
        
        KMatrix[2*N+2*i+1,2*N+2*i] += 6*EI2/L2**2 - 6*EI1/L1**2
        KMatrix[2*N+2*i+1,2*N+2*i+1] += 4*EI1/L1 + 4*EI2/L2
        
        KMatrix[2*N+2*i,2*N+2*i+2] += -12*EI2/L2**3
        KMatrix[2*N+2*i+2,2*N+2*i] += -12*EI2/L2**3

        KMatrix[2*N+2*i,2*N+2*i+3] += 6*EI2/L2**2
        KMatrix[2*N+2*i+3,2*N+2*i] += 6*EI2/L2**2        
        
        KMatrix[2*N+2*i+1,2*N+2*i+2] += -6*EI2/L2**2
        KMatrix[2*N+2*i+2,2*N+2*i+1] += -6*EI2/L2**2

        KMatrix[2*N+2*i+1,2*N+2*i+3] += 2*EI2/L2
        KMatrix[2*N+2*i+3,2*N+2*i+1] += 2*EI2/L2          
    
    KMatrix[-2,-2] += +12*EI2/L2**3
    KMatrix[-2,-1] +=  -6*EI2/L2**2    

    KMatrix[-1,-2] += -6*EI2/L2**2
    KMatrix[-1,-1] += +4*EI2/L2 

    Kinv = np.linalg.inv(KMatrix) 
   
    CMatrix = np.zeros((2*N,2*N))
    for i in range(2*N):
        CMatrix[i,:] = Kinv[2*i,0::2]
        
    KMatrix = np.linalg.inv(CMatrix)

    for i in range(2*N):
        fv = np.zeros(2*N)
        fv[i] = 1
        StiffMatrix[:,i] = KMatrix@fv
        
    w,v = np.linalg.eig(np.linalg.inv(MassMatrix)@StiffMatrix)

    freq = np.sqrt(w)/(2*pi)
   
    fig = plt.figure(figsize = (12,12))
    gs = gridspec.GridSpec(3,2)
    gs.update(hspace=0,wspace=0.5)
    ax1 = plt.subplot(gs[0,0])
    ax3e = plt.subplot(gs[0,1])
    ax3w = plt.subplot(gs[1,1])

    axl = plt.subplot(gs[-1,:])
    
    ax1.plot(rs_h,rs_m,'darkred',linewidth=1)
    for i in range(N):
        ax1.plot([points[i],points[i+1]],[mi[i],mi[i]],'r')
        if i<(N-1):
            ax1.plot([points[i+1],points[i+1]],[mi[i],mi[i+1]],'r')
    ax1.set_xlim(0,L)
    ax1.set_ylim(0)
    ax1.grid()
    
    ax2 = ax1.twinx()

    ax2.plot(rs_h,rs_se*1e-9,'darkblue',linewidth=1)
    for i in range(N):
        ax2.plot([points[i],points[i+1]],[sie[i]*1e-9,sie[i]*1e-9],'b')
        ax2.plot([points[i],points[i+1]],[siw[i]*1e-9,siw[i]*1e-9],'c')
        if i<(N-1):
            ax2.plot([points[i+1],points[i+1]],[sie[i]*1e-9,sie[i+1]*1e-9],'b')
            ax2.plot([points[i+1],points[i+1]],[siw[i]*1e-9,siw[i+1]*1e-9],'c')

    ax2.set_xlim(0,L)
    ax2.set_ylim(0)
    ax2.grid()

    ax1.set_yticks(ax1.get_yticks()[ax1.get_yticks()<=int(max(mass))]/ax1.get_ylim()[1]*ax1.get_ylim()[1])
    ax2.set_yticks(ax1.get_yticks()/ax1.get_ylim()[1]*ax2.get_ylim()[1])

    ax3e.plot(h*0,h*L,'k')
    ax3e.plot(points*0,points,'ok')
    ax3e.set_ylim(0,L)

    ax3w.plot(h*0,h*L,'k')
    ax3w.plot(points*0,points,'ok')
    ax3w.set_ylim(0,L)

    ax1.set_xlabel('Blade span (m)')
    ax1.set_title('Blade distributed properties')
    ax3e.set_title('Modes')

    ax3e.set_ylabel('Blade span (m)')
    ax3e.set_xticklabels('')

    ax1.set_ylabel(u'Mass density (kg/m)',color='red')
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')

    ax2.set_ylabel('Stiffness (GN m$^2$)',color='blue')
    ax2.spines['right'].set_color('blue')
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.spines['left'].set_color('red')

    colors = cm.rainbow(np.linspace(0, 1, 2*n_plot))
    
    for i in range(1,1+n_plot):
        mode_e = np.zeros(N+1)
        mode_e[1:] = v[:N,N-i]

        mode_w = np.zeros(N+1)
        mode_w[1:] = v[N:,N-i]
        
        popt, pcov = curve_fit(ModeFit, points/L , mode_e/mode_e[-1])
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

        fitmode = ModeFit(rs_h/L,*popt)
        ax3e.plot(fitmode/fitmode[-1]*mode_e[-1],rs_h,'--',color=colors[i-1])
            
        ax3e.plot(mode_e,points,'o',color=colors[i-1])
        ax3w.plot(mode_w,points,'o',color=colors[i-1])
        ax3e.plot(0,0,'--o',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[N-i],phi))
        axl.plot(0,0,'--',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[N-i],phi)) 


    for i in range(1,1+n_plot):
        mode_e = np.zeros(N+1)
        mode_e[1:] = v[:N,-i]

        mode_w = np.zeros(N+1)
        mode_w[1:] = v[N:,-i]
        
        popt, pcov = curve_fit(ModeFit, points/L , mode_w/mode_w[-1])
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

        fitmode = ModeFit(rs_h/L,*popt)
        ax3w.plot(fitmode/fitmode[-1]*mode_w[-1],rs_h,'--',color=colors[n_plot+i-1])
            
        ax3e.plot(mode_e,points,'o',color=colors[n_plot+i-1])
        ax3w.plot(mode_w,points,'o',color=colors[n_plot+i-1])
        ax3w.plot(0,0,'--o',color=colors[n_plot+i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi)) 
        axl.plot(0,0,'--',color=colors[n_plot+i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi))

    axl.axis('off')
    axl.legend(loc='upper left',
               bbox_to_anchor=(0,0.8),
               ncol=1,
               fancybox=False,
               framealpha=1,
               frameon=False,
               fontsize=14)

    st.pyplot(fig)  
    return 


cols = st.columns(4)

n_modes = cols[0].number_input('Number of modes',2,None,3)
N = cols[1].number_input('Number of points',5,None,20)
mtop = cols[2].number_input('Rotor mass',0,None,0)
L= cols[3].number_input('Tower height',10,None,100)

st.write(element_type)

if element_type == 'tower':
    TowerModesPlot(np.array(tower.iloc[:,0]),
                   np.array(tower.iloc[:,1]),
                   np.array(tower.iloc[:,2]),
                   N=N,n_plot=n_modes,L=L,mtop=mtop)
else:
    BladesModesPlot(np.array(tower.iloc[:,0]),
          np.array(tower.iloc[:,3]),
          np.array(tower.iloc[:,4]),
          np.array(tower.iloc[:,5]),
          N=N,n_plot=n_modes,L=L,mtop=mtop)

    