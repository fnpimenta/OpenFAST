import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import io
from scipy import signal
import seaborn as sns

from modes import *
from estimators import *
from plot_generators import *
from Print import *
from TurbSim import TurbSimData, TurbSimFile, FullFieldPlot


import struct
import time

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST Course - Task 3'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon )

st.title('Task 3 - Generate a 3D full wind field with TurbSim')

# -- Load data files
@st.cache_data()
def load_data(uploaded_files,n1,n2):
	try:
		data = pd.read_csv(uploaded_file,skiprows=n1,nrows=n2-n1,delimiter="\s+",encoding_errors='replace')
		error_check += 1
	except:
		c1.write('Please select the file for analysis')
		data = 0

	return data

with st.expander("**Objective**",True):

	st.write('''<div style="text-align: justify">
			\nFor this task it is suggested to generate a full 3D wind field using TurbSim for a mean wind speed of your choice and 10\% turbulence intensity.
			\nAfter running TurbSim, upload the .bts file in the data analysis section to visualize the generated wind field.
			\nYou may find a step-by-step procedure in the hints section.
			</div>''',unsafe_allow_html=True)

figs = []

PALETTE = [
	"#ff4b4b",
	"#ffa421",
	"#ffe312",
	"#21c354",
	"#00d4b1",
	"#00c0f2",
	"#1c83e1",
	"#803df5",
	"#808495",
]

onshore_color = "#cc000033"
offshore_color = "#0a75ad33"


ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
ref_path = ref_models['WP 1.5MW']

file_turb = open('./OpenFAST_models/' + ref_path + '/' + '10_TurbSim' + '/' + 'TurbSim.inp', 'r')

checkfile = 1
with st.expander("**Hints**",False):
	st.write('''<div style="text-align: justify">
			\nTo generate the TurbSim full 3D wind field you should firstly prepare the **TurbSim input file**
- Ensure that the output fill-field data in TurbSim/AeroDyn form is enabled in <b>Runtime options</b>
- Check the turbulence scaling method in <b>Runtime options</b>
- Check that the time length of the generated wind field (AnalsysiTime) is sufficient to run a 1000s simulation in <b>Turbine/Model Specifications</b>
- Define the grid dimensions. hub height and grid-point spacing in <b>Turbine/Model Specifications</b>. Do not forget that your grid should be big enough to include all the rotor.
- Define the reference wind velocity at the hub height and the turbulence intensity to 10\% in <b>Meteorological Boundary Conditions</b> section
- Define vertical wind profile as a power law with exponent 0.20 <b>Meteorological Boundary Conditions</b> section
	</div>''',unsafe_allow_html=True)

	#checkfile = st.checkbox('**Show input file details**')
	if checkfile:
		data = []
		for line in file_turb:
			data.append(line)

		
		tab1,tab2,tab3 = st.tabs(['**Runtime options**',
									   '**Turbine/Model Specifications**',
									   '**Meteorological Boundary Conditions**'])

		all_idx = range(3,16)
		on_sel_idx = [9,15]
		off_sel_idx = []
		with tab1:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(18,28)
		on_sel_idx = [18,19,21,23,24,25]
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(30,43)
		on_sel_idx = [33,36,38,39,41]
		off_sel_idx = []
		with tab3:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		st.divider()

		st.write('''<div style="text-align: justify">
				 \nAfter defining the input file, run the TurbSim generator by laucnhing a command window in the TurbSim.exe directory.
				 \nThe command <code>TubrSim InputFile.inp</code> will run TurbSim with InputFile.inp as input file. 
				 You may omit the argument and simply write <code>TubrSim InputFile.inp</code> to run TurbSim with TurbSim.inp as input file.
				 \nAfter running TurbSim and creating the .bts file, upload it in the Data Analsysi section.
			</div>''',unsafe_allow_html=True)

figs = []
with st.expander("**Data analysis**",True):
	st.write('Upload the output file from TurbSim')

	filename = st.file_uploader("TurbSim .bts file",accept_multiple_files=False)

	if not filename==None:
		fdata = TurbSimFile(filename)

		cols = st.columns(2)
		nstep = cols[0].number_input('Interval of time steps to plot (uniformly spaced)',10,len(fdata['t']),1000)
		cols = st.columns(2)

		with cols[0]:
			fig = FullFieldPlot(filename,cols[0],nstep)
			figs.append(fig)
		

		u = fdata.vertProfile(y_span='mid')

		colors = cm.rainbow(np.linspace(0,1,51))

		fig = plt.figure(figsize = (6,6))

		ax = plt.subplot()
		npoints = int(len(fdata['t'])/nstep)

		for i in range(npoints):
			fu = fdata._vertline(ix0=nstep*i , iy0=6)
			ax.plot(fu[0],fdata['z'] , '--',color='k',lw=0.5,alpha=0.5)

		ax.plot(u[1][0,:],fdata['z'],lw=4)
		ax.fill_betweenx(fdata['z'],u[1][0,:]-u[2][0,:],u[1][0,:]+u[2][0,:],alpha=0.25)

		ax.set_xlabel('Wind speed (m/s)')
		ax.set_ylabel('height above ground (m)')

		ax.set_title('Vertical wind profile')
		ax.set_ylim(fdata['z'][0],fdata['z'][-1])
		cols[1].pyplot(fig)

		figs.append(fig)

		cols = st.columns(2)
		zp = cols[0].number_input('Point height for the analysis',
									min_value=np.round(fdata['z'][0],0),
									max_value=np.round(fdata['z'][-1],0),
									value=fdata['zRef'])
		yp = cols[1].number_input('Horizontal position for the analysis',
									min_value=np.round(fdata['y'][0],0),
									max_value=np.round(fdata['y'][-1],0),
									value=0.0)
		u = fdata.valuesAt(y=yp,z=zp)

		cols = st.columns(3)

		cols[0].markdown('Wind speed $\mu$: %.1f m/s'%np.mean(u[0]))
		cols[1].markdown('Wind speed $\sigma$: %.1f m/s'%np.std(u[0]))
		cols[2].markdown('TI: %.1f'%(100*np.std(u[0])/np.mean(u[0])) + ' \%')

		fig = plt.figure(figsize = (12,4))

		gs = gridspec.GridSpec(1,3)
		gs.update(hspace=0.05,wspace=0.1)

		ax2 = plt.subplot(gs[0,1])
		ax1 = plt.subplot(gs[0,0])
		ax3 = plt.subplot(gs[0,2])

		ax1.plot(fdata['t'],u[0])
		ax1.axhline(np.mean(u[0]),ls='-.',c='k')

		nfft = 2048
		f, Pxx = signal.welch(u[0], 20 , nperseg=nfft , scaling='spectrum')

		sns.histplot(x=u[0],
		             kde=False,
		             element='step',
		             alpha=0.10,
		             stat='density',
		             line_kws={'linewidth':4},
		             ax=ax2)

		ws = np.linspace(np.min(u[0]),np.max(u[0]),100)
		wmean = np.mean(u[0])
		wstd = np.std(u[0])

		ndist = 1/(ws[1]-ws[0])*1/np.sum(np.exp(-1/2*((ws-wmean)/wstd)**2))*np.exp(-1/2*((ws-wmean)/wstd)**2)
		ax2.plot(ws,ndist)

		#ax2.axvline(wmean,0,1,ls='--',c='k')
		ax2.arrow(wmean,np.max(ndist)*np.exp(-1/2),wstd,0,
				 length_includes_head=True,shape='full',head_width=0.005,head_length=wstd*0.2,color='k')

		ax2.arrow(wmean,np.max(ndist)*np.exp(-1/2),-wstd,0,
				 length_includes_head=True,shape='full',head_width=0.005,head_length=wstd*0.2,color='k')

		ax2.annotate('$\sigma$=%.1f m/s'%wstd,(wmean,0.95*np.max(ndist)*np.exp(-1/2)),ha='center',va='top')

		ax1.annotate('  $\mu$=%.1f m/s'%wmean,(fdata['t'][-1],wmean),ha='left',va='center', annotation_clip=False,zorder=10)

		ax3.loglog(f[1:],Pxx[1:])

		ax1.set_title('Time series at the selected point')
		ax2.set_title('Amplitude distribution')
		ax3.set_title('Amplitude power spectrum')

		ax1.set_xlabel('Time (s)')
		ax2.set_xlabel('Wind speed (m/s)')
		ax3.set_xlabel('Frequency (Hz)')

		ax2.set_ylabel('')
		ax2.set_yticks([])

		ax3.set_ylabel('')
		ax3.set_yticks([])

		ax1.set_xlim(fdata['t'][0],fdata['t'][-1])
		ax3.set_xlim(f[1],f[-1])

		st.pyplot(fig)

		figs.append(fig)



exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	create_pdf_task3(figs,report_text,'Task 3: Full 3D wind field generator','Task3_report',exp_c[1],exp,yp,zp)
	#
	#try:
	#	create_pdf_task3(figs,report_text,'Task 3: Full 3D wind field generator','Task3_report',exp_c[1],exp,file_id+1)
	#except:
	#	exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")
