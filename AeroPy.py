import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import signal , integrate, linalg
from scipy.signal import hilbert

import os
import random
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import least_squares , curve_fit
from matplotlib import cm

pi = np.pi

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
        
        #st.write(xb,xa,yb,ya,self.length)
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



def AeroCoefficients(x,y,N,u0,alpha,chord): 
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

    fig = plt.figure(figsize=(15,8))

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

    st.pyplot(fig)
    return
