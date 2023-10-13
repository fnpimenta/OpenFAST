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