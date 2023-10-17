import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
from scipy import signal

from modes import *

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST Course - Task 2'
icon = Image.open('logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 2 - Simulations with prescribed initial conditions')

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
			\nAfter completing Task 1 you should have the tower and blades ElastoDyn input files ready to perform a first simulation. 
			For this task, you will perform a **200s** free virabtion simulation of the WindPact wind turbine **imposing a 1m displacement at the tower top** in the FA direction. 
			Compute and represent the power spectra from the simulated time series at a point at around 70% of the tower height considering the following scenarios:</div>''',unsafe_allow_html=True)
	st.write(r'''
			1. Considering only the 1$^\text{st}$ FA tower DoF
			2. Add the 2$^\text{nd}$  FA tower DoF
			3. Add all the blades' DoF
			
			Estimate the structural natural frequenciy and damping coefficient for the tower 1$^\text{st}$ FA mode.
		''')

plat_dof = Image.open('figures/floating_dof.png')

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

with st.expander("**Hints**",False):
	st.write('''<div style="text-align: justify">
			\nTo solve the tasks above you will need to prepare and run 3 different OpenFAST simulations.
			The **relevant files** to edit are listed below and the **relevant parameters and sections highlighted**. 
			\nOnce you have the 3 output files, you may uploaded below to conduct the data analysis.
			</div>''',unsafe_allow_html=True)
	c1,c2,c3 = st.columns(3)

	# -- Load data files
	ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
	ref_model = c1.selectbox('Reference model', ref_models, index=1,disabled=True)
	ref_path = ref_models[ref_model]

	all_dir = os.listdir('./OpenFAST_models/' + ref_path )
	sel_dir = c2.selectbox('Available modules', all_dir, index = 0 ,disabled=True)

	all_files = os.listdir('./OpenFAST_models/' + ref_path + '/' + sel_dir)
	sel_file = c3.selectbox('Available files', all_files, index = 1,disabled=True)
	sel_file = 'WP_ElastoDyn.dat'
			
	log = open('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + sel_file, 'r')
	data = []
	for line in log:
		data.append(line)

	tab1,tab2,tab3,tab4 = st.tabs(['Simulation Control','**Degrees of freedom**','**Initial conditions**','Turbine configuration'])
	for i in range(3,6):
		tab1.write(data[i])
		
	all_idx = range(9,26)
	on_sel_idx = [9,10,11,16,17,18,19]
	off_sel_idx = []

	with tab2:
		for i in all_idx:
			if i in on_sel_idx:
				st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
			elif i in off_sel_idx:
				st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
			else:
				st.write(data[i])

			
		st.image(plat_dof)

	all_idx = range(27,44)
	on_sel_idx = [34,36]
	off_sel_idx = []
	with tab3:
		for i in all_idx:
			if i in on_sel_idx:
				st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
			elif i in off_sel_idx:
				st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
			else:
				st.write(data[i])
	
	all_idx = range(45,71)
	on_sel_idx = [34,36]
	off_sel_idx = []
	with tab4:
		for i in all_idx:
			if i in on_sel_idx:
				st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
			elif i in off_sel_idx:
				st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
			else:
				st.write(data[i])

with st.expander("**Data analysis**",True):
	st.write('Uploaded the output files from OpenFAST')
	
	cols = st.columns(3)
	file = []

	file.append(cols[0].file_uploader("1 FA mode only ",accept_multiple_files=False))
	file.append(cols[1].file_uploader("2 FA modes",accept_multiple_files=False))
	file.append(cols[2].file_uploader("All DoFs",accept_multiple_files=False))
	
	nfiles = np.zeros(len(file))-1
	for i in range(len(file)):
		if not(file[i]==None):
			nfiles[i] = int(i)


	if sum(nfiles>=0)>0:
		file[0].seek(0)
		data = pd.read_csv(file[0] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
		keys = data.columns
		nvar = len(keys)

		cols = st.columns(2)

		tcol = cols[0].selectbox('Time column', data.columns,index=0)
		dof = cols[1].selectbox('Data column', data.columns,index=1)

		sep_plots = st.checkbox('Separate plots',value=True)
		tabs = st.tabs(['Time series analysis','Modal analysis'])
		
		
		with tabs[0]:

			cols = st.columns([0.5,0.1,0.5])

#			tmin = cols[0].number_input('$t_{min}$',0.0,None,0.0)
#			tmax = cols[1].number_input('$t_{max}$',0.0,None,float(data[tcol].iloc[-1]))
			tmin,tmax = cols[0].slider('Time range',0.0,float(data[tcol].iloc[-1]),(0.0,float(data[tcol].iloc[-1])))

#			fmin = cols[0].number_input('$f_{min}$',0.0,None,0.0)
#			fmax = cols[1].number_input('$f_{max}$',0.0,None,float(0.5/(data[tcol][1]-data[tcol][0])))
			fmin,fmax = cols[2].slider('Frequency range',0.0,float(0.5/(data[tcol][1]-data[tcol][0])),(0.0,float(0.25/(data[tcol][1]-data[tcol][0]))))
			nfft = cols[2].number_input('FFT number of points',2**4,None,4096)
			
			if sep_plots:
				fig = plt.figure(figsize = (12,10))

				gs = gridspec.GridSpec(3,2)
				gs.update(hspace=0.05,wspace=0.25)

				for i in range(len(file)):			
					ax1 = plt.subplot(gs[i,0])
					ax2 = plt.subplot(gs[i,1])
					if i>=0:
						file[i].seek(0)
						data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
						
						tfilter = (data[tcol]>=tmin) & (data[tcol]<=tmax)

						f, Pxx = signal.welch(data[dof][tfilter], 1/(data[tcol][1]-data[tcol][0]) , nperseg=nfft , scaling='spectrum')

						ax1.plot(data[tcol],data[dof])
						ax2.semilogy(f,Pxx)

						ax1.set_xlim(tmin,tmax)
						ax2.set_xlim(fmin,fmax)

						if i<(len(file)-1):
							ax1.set_xticklabels('')
							ax2.set_xticklabels('')
			else:
				fig = plt.figure(figsize = (12,4))

				gs = gridspec.GridSpec(1,2)
				gs.update(hspace=0.05,wspace=0.25)

				ax1 = plt.subplot(gs[0,0])
				ax2 = plt.subplot(gs[0,1])
				for i in range(len(file)):			
					if i>=0:
						file[i].seek(0)
						data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
						
						tfilter = (data[tcol]>=tmin) & (data[tcol]<=tmax)

									
						f, Pxx = signal.welch(data[dof][tfilter], 1/(data[tcol][1]-data[tcol][0]) , nperseg=nfft , scaling='spectrum')
						
						ax1.plot(data[tcol],data[dof])
						ax2.semilogy(f,Pxx)

						ax1.set_xlim(tmin,tmax)
						ax2.set_xlim(fmin,fmax)
			st.pyplot(fig  )