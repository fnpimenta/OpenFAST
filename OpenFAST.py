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

def AeroCoefficients(filename):
    '''
    Interactive estimator of the aerodynamic coefficients. 
    The lift coefficient is obtained from the AeroPython code developed by Barba, Lorena A., Mesnard, Olivier (2019). Aero Python: classical aerodynamics of potential flow using Python. Journal of Open Source Education, 2(15), 45, https://doi.org/10.21105/jose.00045
    and available in https://github.com/barbagroup/AeroPython.
    The drag coefficient is estimated based on the solution for a flat plate. 
    In this case, the wind field is projected in the chord direction and in a direction orthogonal to it.
    For the parallel component, a combination of turbulence and laminar flow results over a flat are considered.
    For the perpendicular component, a flat plate face up to thw wind with a Cd=1.28 is considered.

    Parameters
    ----------
    filename : string
        File containing the geometry of the airfoil.
    '''

    if not(type(filename)==str):
        print('Error loading data. Input is not a string.')
        return
    else:
        if filename[-4:]=='.dat':
            x = np.array(pd.read_csv(filename,delimiter='\s+',skiprows=1).iloc[:,0])
            y = np.array(pd.read_csv(filename,delimiter='\s+',skiprows=1).iloc[:,1])
        else:
            x = np.array(pd.read_csv(filename + '.dat',delimiter='\s+',skiprows=1).iloc[:,0])
            y = np.array(pd.read_csv(filename + '.dat',delimiter='\s+',skiprows=1).iloc[:,1])          

        n_calc = IntSlider(value=20,min=10,max=100,step=2,description='N panels:',continuous_update=False,readout=True)    
        alpha = FloatSlider(value=0,min=-10,max=10,description='AoA (degrees):',continuous_update=False,readout=True)  
        u0 = FloatSlider(value=10,min=1,max=30,description='u (m/s):',continuous_update=False,readout=True)     
        chord = FloatText(value=1,description='c (m):',disabled=False)

        diagrams = widgets.interactive_output(CEstimator, {'x':fixed((x-x.min())/(x.max()-x.min())),
                                                           'y':fixed(y/(x.max()-x.min())),
                                                           'N':n_calc,'u0':u0,'alpha':alpha,'chord':chord})
        display(VBox([HBox([n_calc,chord]),HBox([alpha,u0]),diagrams]))   

        return

def CEstimator(x,y,N,u0,alpha,chord): 
    panels = define_panels(x-x.min(),y, N=N)
    A_source = source_contribution_normal(panels)
    B_vortex = vortex_contribution_normal(panels)
    A = build_singularity_matrix(A_source, B_vortex)

    freestream = Freestream(u_inf=u0, alpha=alpha)
    b = build_freestream_rhs(panels, freestream)

    # solve for singularity strengths
    strengths = np.linalg.solve(A, b)

    # store source strength on each panel
    for i , panel in enumerate(panels):
        panel.sigma = strengths[i]
        
    # store circulation density
    gamma = strengths[-1]
    compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex)

    # surface pressure coefficient
    compute_pressure_coefficient(panels, freestream)

    # compute the chord and lift coefficient
    c = abs(max(panel.xa for panel in panels) - min(panel.xa for panel in panels))
    cl  = (gamma * sum(panel.length for panel in panels) / (0.5 * freestream.u_inf * c))

    u_d = u0*np.cos(alpha*np.pi/180)
    u_f = u0*np.sin(alpha*np.pi/180)
    
    rho = 1.225
    mu = 18.5*1e-6
    Re = u_d*chord*rho/mu
    
    f_l = 1.328/np.sqrt(Re)
    f_t = 0.074/Re**(1/5)
    if Re<5e5:
        f_d = f_l
    if Re>1e7:
        f_d = f_t
    else:
        f_d = f_t/2 + f_l/2
    f_f = 1.28
    cd = ((f_d*u_d**2) + (f_f*u_f**2))/(u0**2)

    plt.figure(figsize=(15,8))

    gs = gridspec.GridSpec(2,1)
    gs.update(hspace=0)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])

    x_rot = np.zeros(len(x))
    y_rot = np.zeros(len(x))

    x_pan = np.append([panel.xc*np.cos(alpha*np.pi/180) + panel.yc*np.sin(alpha*np.pi/180) for panel in panels], 
                       panels[0].xa*np.cos(alpha*np.pi/180) + panels[0].ya*np.sin(alpha*np.pi/180))
    y_pan = np.append([-panel.xc*np.sin(alpha*np.pi/180) + panel.yc*np.cos(alpha*np.pi/180) for panel in panels], 
                       -panels[0].xa*np.sin(alpha*np.pi/180) + panels[0].ya*np.cos(alpha*np.pi/180))
    for i in range(len(x)):
        x_rot[i] = np.cos(alpha*np.pi/180)*x[i] + np.sin(alpha*np.pi/180)*y[i]
        y_rot[i] = -np.sin(alpha*np.pi/180)*x[i] + np.cos(alpha*np.pi/180)*y[i]
        
    ax1.grid()
    ax1.set_ylabel('$y/c$', fontsize=16)
    ax1.plot(x, y, color='k', linestyle='-', linewidth=2)
    ax1.plot([panel.xc for panel in panels],[panel.yc for panel in panels],
             linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_xticklabels('')
    ax1.set_title('$C_L$=%.3f , $C_D$=%.3f'%(cl,cd), fontsize=16)

    # plot surface pressure coefficient
    ax2.grid()
    ax2.set_xlabel('$x/c$', fontsize=16)
    ax2.set_ylabel('$y/c$', fontsize=16)


    #ax2.plot([panel.xc for panel in panels if panel.loc == 'upper'],
    #         [panel.cp for panel in panels if panel.loc == 'upper'],
    #         label='upper surface',
    #         color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
    #ax2.plot([panel.xc for panel in panels if panel.loc == 'lower'],
    #         [panel.cp for panel in panels if panel.loc == 'lower'],
    #         label= 'lower surface',
    #         color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
    ax2.set_xlim(-0.1,1.1)
    ax2.plot(x_rot, y_rot, color='k', linestyle='-', linewidth=2)
    arrow_width = 0.002
    ax2.plot(x_rot[0], y_rot[0], color='b',label='$C_p>0$')
    ax2.plot(x_rot[0], y_rot[0], color='r',label='$C_p<0$')

    max_cp = np.max([abs(panel.cp) for panel in panels])
    for panel in panels:
        if panel.cp>0:
            ax2.arrow(+panel.xc*np.cos(alpha*np.pi/180) + panel.yc*np.sin(alpha*np.pi/180) + np.cos(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      -panel.xc*np.sin(alpha*np.pi/180) + panel.yc*np.cos(alpha*np.pi/180) + np.sin(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      -np.cos(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      -np.sin(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      length_includes_head=True,color='b',width=arrow_width)
        else:
            ax2.arrow(+panel.xc*np.cos(alpha*np.pi/180) + panel.yc*np.sin(alpha*np.pi/180) + np.cos(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      -panel.xc*np.sin(alpha*np.pi/180) + panel.yc*np.cos(alpha*np.pi/180) + np.sin(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      -np.cos(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      -np.sin(panel.beta-alpha*np.pi/180)*panel.cp/max_cp * 1/10,
                      length_includes_head=True,color='r',width=arrow_width)
    ax2.legend(loc='best',
               ncol=1,
               fancybox=False,
               framealpha=0,
               frameon=True)

    return

class Panel:
    """
    Contains information related to a panel.
    """
    def __init__(self, xa, ya, xb, yb):
        """
        Initializes the panel.
        
        Sets the end-points and calculates the center-point, length,
        and angle (with the x-axis) of the panel.
        Defines if the panel is located on the upper or lower surface of the geometry.
        Initializes the source-strength, tangential velocity, and pressure coefficient
        of the panel to zero.
        
        Parameters
        ---------_
        xa: float
            x-coordinate of the first end-point.
        ya: float
            y-coordinate of the first end-point.
        xb: float
            x-coordinate of the second end-point.
        yb: float
            y-coordinate of the second end-point.
        """
        self.xa, self.ya = xa, ya  # panel starting-point
        self.xb, self.yb = xb, yb  # panel ending-point
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # panel center
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)  # panel length
        
        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)
        
        # panel location
        if self.beta <= np.pi:
            self.loc = 'upper'  # upper surface
        else:
            self.loc = 'lower'  # lower surface
        
        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient

def define_panels(x, y, N=40):
    """
    Discretizes the geometry into panels using 'cosine' method.
    
    Parameters
    ----------
    x: 1D array of floats
        x-coordinate of the points defining the geometry.
    y: 1D array of floats
        y-coordinate of the points defining the geometry.
    N: integer, optional
        Number of panels;
        default: 40.
    
    Returns
    -------
    panels: 1D Numpy array of Panel objects.
        The list of panels.
    """
    
    R = (x.max() - x.min()) / 2.0  # circle radius
  
    x_center = (x.max() + x.min()) / 2.0  # x-coordinate of circle center
    theta = np.linspace(0.0, 2.0 * np.pi, N + 1)  # array of angles
    x_circle = x_center + R * np.cos(theta)  # x-coordinates of circle

    x_ends = np.copy(x_circle)  # x-coordinate of panels end-points
    y_ends = np.empty_like(x_ends)  # y-coordinate of panels end-points
    
    # extend coordinates to consider closed surface
    #x, y = np.append(x, x[0]), np.append(y, y[0])
    # compute y-coordinate of end-points by projection
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]

    # create panels
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])
    
    return panels

class Freestream:
    """
    Freestream conditions.
    """
    def __init__(self, u_inf=1.0, alpha=0.0):
        """
        Sets the freestream speed and angle (in degrees).
        
        Parameters
        ----------
        u_inf: float, optional
            Freestream speed;
            default: 1.0.
        alpha: float, optional
            Angle of attack in degrees;
            default 0.0.
        """
        self.u_inf = u_inf
        self.alpha = np.radians(alpha)  # degrees to radians

def integral(x, y, panel, dxdk, dydk):
    """
    Evaluates the contribution from a panel at a given point.
    
    Parameters
    ----------
    x: float
        x-coordinate of the target point.
    y: float
        y-coordinate of the target point.
    panel: Panel object
        Panel whose contribution is evaluated.
    dxdk: float
        Value of the derivative of x in a certain direction.
    dydk: float
        Value of the derivative of y in a certain direction.
    
    Returns
    -------
    Contribution from the panel at a given point (x, y).
    """
    def integrand(s):
        return (((x - (panel.xa - np.sin(panel.beta) * s)) * dxdk +
                 (y - (panel.ya + np.cos(panel.beta) * s)) * dydk) /
                ((x - (panel.xa - np.sin(panel.beta) * s))**2 +
                 (y - (panel.ya + np.cos(panel.beta) * s))**2) )
    return integrate.quad(integrand, 0.0, panel.length)[0]

def source_contribution_normal(panels):
    """
    Builds the source contribution matrix for the normal velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    
    Returns
    -------
    A: 2D Numpy array of floats
        Source contribution matrix.
    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # source contribution on a panel from itself
    np.fill_diagonal(A, 0.5)
    # source contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral(panel_i.xc, panel_i.yc, 
                                                panel_j,
                                                np.cos(panel_i.beta),
                                                np.sin(panel_i.beta))
    return A

def vortex_contribution_normal(panels):
    """
    Builds the vortex contribution matrix for the normal velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    
    Returns
    -------
    A: 2D Numpy array of floats
        Vortex contribution matrix.
    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from itself
    np.fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / np.pi * integral(panel_i.xc, panel_i.yc, 
                                                  panel_j,
                                                  np.sin(panel_i.beta),
                                                  -np.cos(panel_i.beta))
    return A

def kutta_condition(A_source, B_vortex):
    """
    Builds the Kutta condition array.
    
    Parameters
    ----------
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    
    Returns
    -------
    b: 1D Numpy array of floats
        The left-hand side of the Kutta-condition equation.
    """
    b = np.empty(A_source.shape[0] + 1, dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    b[:-1] = B_vortex[0, :] + B_vortex[-1, :]
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    b[-1] = - np.sum(A_source[0, :] + A_source[-1, :])
    return b

def build_singularity_matrix(A_source, B_vortex):
    """
    Builds the left-hand side matrix of the system
    arising from source and vortex contributions.
    
    Parameters
    ----------
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    
    Returns
    -------
    A:  2D Numpy array of floats
        Matrix of the linear system.
    """
    A = np.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype=float)
    # source contribution matrix
    A[:-1, :-1] = A_source
    # vortex contribution array
    A[:-1, -1] = np.sum(B_vortex, axis=1)
    # Kutta condition array
    A[-1, :] = kutta_condition(A_source, B_vortex)
    return A

def build_freestream_rhs(panels, freestream):
    """
    Builds the right-hand side of the system 
    arising from the freestream contribution.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    
    Returns
    -------
    b: 1D Numpy array of floats
        Freestream contribution on each panel and on the Kutta condition.
    """
    b = np.empty(panels.size + 1, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * np.cos(freestream.alpha - panel.beta)
    # freestream contribution on the Kutta condition
    b[-1] = -freestream.u_inf * (np.sin(freestream.alpha - panels[0].beta) +
                                 np.sin(freestream.alpha - panels[-1].beta) )
    return b

def compute_tangential_velocity(panels, freestream, gamma, A_source, B_vortex):
    """
    Computes the tangential surface velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    gamma: float
        Circulation density.
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    """
    A = np.empty((panels.size, panels.size + 1), dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    A[:, :-1] = B_vortex
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    A[:, -1] = -np.sum(A_source, axis=1)
    # freestream contribution
    b = freestream.u_inf * np.sin([freestream.alpha - panel.beta for panel in panels])
    
    strengths = np.append([panel.sigma for panel in panels], gamma)
    
    tangential_velocities = np.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]

def compute_pressure_coefficient(panels, freestream):
    """
    Computes the surface pressure coefficients.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2


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