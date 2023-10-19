import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

pi = np.pi


def ModeFit(h,a2,a3,a4,a5):
    a6 = (1-a2-a3-a4-a5)
    return a2*h**2 + a3*h**3 + a4*h**4 + a5*h**5 + a6*h**6
    
def TowerModesPlot(h,mass,stiff,N=2,n_plot=2,L=100,mtop=0,kphi=1):
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
    for i in range(N):
        L1 = points[i+1]-points[i]
        EI1 = si[i]

        if i<(N-1):
            L2 = points[i+2]-points[i+1]
            EI2 = si[i+1]

            KMatrix[2*i,2*i] += (12*EI1/L1**3 + 12*EI2/L2**3) * 1/2
            KMatrix[2*i+1,2*i+1] += (4*EI1/L1 + 4*EI2/L2) * 1/2
            KMatrix[2*i,2*i+1] += 6*EI2/L2**2 - 6*EI1/L1**2

        else:
            KMatrix[2*i,2*i] += (12*EI1/L1**3) * 1/2
            KMatrix[2*i+1,2*i+1] += (4*EI1/L1) * 1/2
            KMatrix[2*i,2*i+1] += -6*EI1/L1**2
        
        if i<(N-1):

            KMatrix[2*i,2*i+2] += -12*EI2/L2**3
            KMatrix[2*i,2*i+3] += 6*EI2/L2**2
            KMatrix[2*i+1,2*i+2] += -6*EI2/L2**2
            KMatrix[2*i+1,2*i+3] += 2*EI2/L2

    KMatrix += KMatrix.T

    Kinv = np.linalg.inv(KMatrix) 

    CMatrix = Kinv[::2,::2]
    StiffMatrix = np.linalg.inv(CMatrix)

    #for i in range(N):
    #    fv = np.zeros(N)
    #    fv[i] = 1
    #    StiffMatrix[:,i] = KMatrix@fv

    w,v = np.linalg.eig(np.linalg.inv(MassMatrix)@StiffMatrix)
    freq = np.sqrt(w)/(2*pi)
    
    fig = plt.figure(figsize = (12,9))
    gs = gridspec.GridSpec(3,3)
    gs.update(hspace=0,wspace=0.3)
    ax1 = plt.subplot(gs[0:-1,1:])
    ax3 = plt.subplot(gs[:,0])

    axl = plt.subplot(gs[-1,1:])
    
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
        phi = ' %.2f$x^2$'%(kphi*popt[0])
        for j in range(3):
            if popt[j+1]>0:
                phi += ' + %.2f$x^%d$'%(kphi*popt[j+1],j+3)
            else:
                phi += ' - %.2f$x^%d$'%(kphi*abs(popt[j+1]),j+3)
        if (1-np.sum(popt))>0:
            phi += ' + %.2f$x^%d$'%(kphi*(1-np.sum(popt)),6)
        else:
            phi += ' - %.2f$x^%d$'%(kphi*abs(1-np.sum(popt)),6)
            
        ax3.plot(mode,points,'o',color=colors[i-1])
        
        fitmode = ModeFit(rs_h/L,*popt)

        ax3.plot(fitmode/fitmode[-1]*mode[-1],rs_h,'--',color=colors[i-1])
        ax3.plot(0,0,'--o',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi)) 
        axl.plot(0,0,'--',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi)) 

    axl.axis('off')
    axl.legend(loc='lower left',
               bbox_to_anchor=(-0.1,0),
               ncol=1,
               fancybox=False,
               framealpha=1,
               frameon=False,
               fontsize=12)

    return fig 

def BladesModesPlot(h,mass,stiff_edge,stiff_wise,twist,N=2,n_plot=2,L=100,mtop=0):
    seg = np.linspace(0,1,N+1)*L

    f_m = interp1d(h*L,mass)
    f_se = interp1d(h*L,stiff_edge)   
    f_sw = interp1d(h*L,stiff_wise)    
    f_tw = interp1d(h*L,twist)    
     
    points = np.linspace(0,1,N+1)*L

    mi = f_m(points)
    sie = f_se(points)
    siw = f_sw(points)
    thetas = f_tw(points)
    
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

    for i in range(N):
        theta = thetas[i]*pi/180*0
        c = np.cos(theta)
        s = np.sin(theta)
        
        L1 = points[i+1]-points[i]
        EI1x = sie[i]
        EI1y = siw[i]

        if i<(N-1):
            L2 = points[i+2]-points[i+1]
            EI2x = sie[i+1]
            EI2y = siw[i+1]

            KMatrix[2*i,2*i] += ((12*EI1x/L1**3 + 12*EI2x/L2**3) * 1/2) * c**2 + ((12*EI1y/L1**3 + 12*EI2y/L2**3) * 1/2) * s**2 
            KMatrix[2*i,2*i+2*N] += ((12*EI1x/L1**3 + 12*EI2x/L2**3) - (12*EI1y/L1**3 + 12*EI2y/L2**3)) * c*s 
            
            KMatrix[2*i+1,2*i+1] += ((4*EI1x/L1 + 4*EI2x/L2) * 1/2) * c**2 + ((4*EI1y/L1 + 4*EI2y/L2) * 1/2) * s**2
            KMatrix[2*i+1,2*i+1+2*N] += ((4*EI1x/L1 + 4*EI2x/L2)  - (4*EI1y/L1 + 4*EI2y/L2)) * c*s

            KMatrix[2*i,2*i+1] += (6*EI2x/L2**2 - 6*EI1x/L1**2) * c**2 + (6*EI2y/L2**2 - 6*EI1y/L1**2) * s**2
            KMatrix[2*i,2*i+1+2*N] += ((6*EI2x/L2**2 - 6*EI1x/L1**2) - (6*EI2y/L2**2 - 6*EI1y/L1**2)) * c*s

        else:
            KMatrix[2*i,2*i] += ((12*EI1x/L1**3) * 1/2) * c**2 + ((12*EI1y/L1**3) * 1/2) * s**2
            KMatrix[2*i,2*i+2*N] += (12*EI1x/L1**3 - 12*EI1y/L1**3) * c*s 
            
            KMatrix[2*i+1,2*i+1] += ((4*EI1x/L1) * 1/2) * c**2 + ((4*EI1y/L1) * 1/2) * s**2
            KMatrix[2*i+1,2*i+1+2*N] += (4*EI1x/L1  - 4*EI1y/L1) * c*s

            KMatrix[2*i,2*i+1] += (-6*EI1x/L1**2) * c**2 + (-6*EI1y/L1**2) * s**2
            KMatrix[2*i,2*i+1+2*N] += ((-6*EI1x/L1**2) - (-6*EI1y/L1**2)) * c*s

        if i<(N-1):

            KMatrix[2*i,2*i+2] += (-12*EI2x/L2**3)* c**2 + (-12*EI2y/L2**3)* s**2 
            KMatrix[2*i,2*i+2+2*N] += ((-12*EI2x/L2**3) - (-12*EI2y/L2**3))* c*s 

            KMatrix[2*i,2*i+3] += (6*EI2x/L2**2) * c**2 + (6*EI2y/L2**2) * s**2 
            KMatrix[2*i,2*i+3+2*N] += ((6*EI2x/L2**2) - (6*EI2y/L2**2)) * c*s 

            KMatrix[2*i+1,2*i+2] += (-6*EI2x/L2**2) * c**2 + (-6*EI2y/L2**2) * s**2
            KMatrix[2*i+1,2*i+2+2*N] += ((-6*EI2x/L2**2) - (-6*EI2y/L2**2)) * c*s

            KMatrix[2*i+1,2*i+3] += (2*EI2x/L2) * c**2 + (2*EI2y/L2) * s**2
            KMatrix[2*i+1,2*i+3+2*N] += ((2*EI2x/L2) - (2*EI2y/L2)) * c*s


    for i in range(N):
        theta = thetas[i]*pi/180*0
        c = np.cos(theta)
        s = np.sin(theta)
        
        L1 = points[i+1]-points[i]
        EI1x = sie[i]
        EI1y = siw[i]

        if i<(N-1):
            L2 = points[i+2]-points[i+1]
            EI2x = sie[i+1]
            EI2y = siw[i+1]

            KMatrix[2*i+2*N,2*i+2*N] += ((12*EI1x/L1**3 + 12*EI2x/L2**3) * 1/2) * s**2 + ((12*EI1y/L1**3 + 12*EI2y/L2**3) * 1/2) * c**2 
            KMatrix[2*i+1+2*N,2*i+1+2*N] += ((4*EI1x/L1 + 4*EI2x/L2) * 1/2) * s**2 + ((4*EI1y/L1 + 4*EI2y/L2) * 1/2) * c**2

            KMatrix[2*i+2*N,2*i+1+2*N] += (6*EI2x/L2**2 - 6*EI1x/L1**2) * s**2 + (6*EI2y/L2**2 - 6*EI1y/L1**2) * c**2

        else:
            KMatrix[2*i+2*N,2*i+2*N] += ((12*EI1x/L1**3) * 1/2) * s**2 + ((12*EI1y/L1**3) * 1/2) * c**2
            
            KMatrix[2*i+1+2*N,2*i+1+2*N] += ((4*EI1x/L1) * 1/2) * s**2 + ((4*EI1y/L1) * 1/2) * c**2
         
            KMatrix[2*i+2*N,2*i+1+2*N] += (-6*EI1x/L1**2) * s**2 + (-6*EI1y/L1**2) * c**2
         
        if i<(N-1):

            KMatrix[2*i+2*N,2*i+2+2*N] += (-12*EI2x/L2**3)* s**2 + (-12*EI2y/L2**3)* c**2 
         
            KMatrix[2*i+2*N,2*i+3+2*N] += (6*EI2x/L2**2) * s**2 + (6*EI2y/L2**2) * c**2 
         
            KMatrix[2*i+1+2*N,2*i+2+2*N] += (-6*EI2x/L2**2) * s**2 + (-6*EI2y/L2**2) * c**2
         
            KMatrix[2*i+1+2*N,2*i+3+2*N] += (2*EI2x/L2) * s**2 + (2*EI2y/L2) * c**2

    KMatrix += KMatrix.T

    Kinv = np.linalg.inv(KMatrix) 

    CMatrix = Kinv[::2,::2]
    StiffMatrix = np.linalg.inv(CMatrix)

    #for i in range(2*N):
    #    fv = np.zeros(2*N)
    #    fv[i] = 1
    #    StiffMatrix[:,i] = KMatrix@fv

    w,v = np.linalg.eig(np.linalg.inv(MassMatrix)@StiffMatrix)

    freq = np.sqrt(w)/(2*pi)
   
    fig = plt.figure(figsize = (18,18))
    gs = gridspec.GridSpec(5,4)
    gs.update(hspace=0.1,wspace=0.1)
    ax1 = plt.subplot(gs[0:-3,2:])
    
    ax3e = plt.subplot(gs[0:-2,0])
    ax3w = plt.subplot(gs[-2,1:])
    ax3f = plt.subplot(gs[-2,0])

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


    ax3w.plot(h*L,h*0,'k')
    ax3w.plot(points,points*0,'ok')
    ax3w.set_xlim(0,L)

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
        ax3w.plot(points,mode_w,'o',color=colors[i-1])
        ax3e.plot(0,0,'--o',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[N-i],phi))
        axl.plot(0,0,'--',color=colors[i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[N-i],phi)) 

        ax3f.plot(mode_e,mode_w,'--.',color=colors[i-1])
        

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
        ax3w.plot(rs_h,fitmode/fitmode[-1]*mode_w[-1],'--',color=colors[n_plot+i-1])
            
        ax3e.plot(mode_e,points,'o',color=colors[n_plot+i-1])
        ax3w.plot(points,mode_w,'o',color=colors[n_plot+i-1])
        ax3w.plot(0,0,'--o',color=colors[n_plot+i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi)) 

        ax3f.plot(mode_e,mode_w,'--.',color=colors[n_plot+i-1])

        axl.plot(0,0,'--',color=colors[n_plot+i-1],label='$f$=%.2f Hz , $\phi(x)$:%s'%(freq[-i],phi))

    max_coord = 1.1*np.max((np.max(abs(v[:,-3:])),np.max(abs(v[:,N-3:N]))))

    ax3f.set_xlim(-max_coord,max_coord)
    ax3f.set_ylim(-max_coord,max_coord)

    ax3w.set_ylim(-max_coord,max_coord)
    ax3w.set_yticklabels('')

    ax3e.set_xlim(-max_coord,max_coord)
    ax3e.set_xticklabels('')

    ax3f.set_xlabel('Edgewise direction')
    ax3f.set_ylabel('Flapwise direction')

    ax3f.set_xticklabels('')
    ax3f.set_yticklabels('')

    axl.axis('off')
    axl.legend(loc='upper left',
               bbox_to_anchor=(0,0.8),
               ncol=1,
               fancybox=False,
               framealpha=1,
               frameon=False,
               fontsize=14)

    return fig
