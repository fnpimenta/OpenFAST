import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
from scipy import signal

from modes import *
from estimators import *
from plot_generators import *

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST Course - Task 2'
icon = Image.open('feup_logo.ico')
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
			
			Estimate the structural natural frequency and damping coefficient for the tower 1$^\text{st}$ FA mode.
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
		
		fs = 1/np.array(data[tcol][1]-data[tcol][0])

		with tabs[0]:

			cols = st.columns([0.5,0.1,0.5])

			t_min,t_max = cols[0].slider('Time range',0.0,float(data[tcol].iloc[-1]),(0.0,float(data[tcol].iloc[-1])))
			f_min,f_max = cols[2].slider('Frequency range',0.0,float(0.5/(data[tcol][1]-data[tcol][0])),(0.0,float(0.25/(data[tcol][1]-data[tcol][0]))))
			
			tfilter = (data[tcol]>=t_min) & (data[tcol]<=t_max)

			nmin = 4
			nmax = np.max((int(np.log2(len(data[tfilter]))) + 1,8))
	
			nfft = cols[2].select_slider('FFT number of points',[int(2**x) for x in np.arange(nmin,nmax)],int(2**(nmax-3)))

			if sep_plots:
				fig = plt.figure(figsize = (12,10))

				gs = gridspec.GridSpec(3,2)
				gs.update(hspace=0.05,wspace=0.25)

				for i in range(len(file)):			
					ax1 = plt.subplot(gs[i,0])
					ax2 = plt.subplot(gs[i,1])
					if nfiles[i]>=0:
						file[i].seek(0)
						data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
						
						tfilter = (data[tcol]>=t_min) & (data[tcol]<=t_max)

						f, Pxx = signal.welch(data[dof][tfilter], 1/(data[tcol][1]-data[tcol][0]) , nperseg=nfft , scaling='spectrum')

						ax1.plot(data[tcol],data[dof])
						ax2.semilogy(f,Pxx)

						ax1.set_xlim(t_min,t_max)
						ax2.set_xlim(f_min,f_max)

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
					if nfiles[i]>=0:
						file[i].seek(0)
						data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
						
						tfilter = (data[tcol]>=t_min) & (data[tcol]<=t_max)

									
						f, Pxx = signal.welch(data[dof][tfilter], 1/(data[tcol][1]-data[tcol][0]) , nperseg=nfft , scaling='spectrum')
						
						ax1.plot(data[tcol],data[dof])
						ax2.semilogy(f,Pxx)

					ax1.set_xlim(t_min,t_max)
					ax2.set_xlim(f_min,f_max)
			st.pyplot(fig)


		with tabs[1]:
			peaks_types = {
						 "Positive peaks only": 0,
						 "All peaks": 1,
						}

			peaks_type = st.radio('Peaks to use', peaks_types.keys(),horizontal=True)
			filt_app = st.checkbox('Apply filter') 
			fit_types = {
						 "Trend line": 0,
						 "Mean value": 1,
						}

			fit_type = st.radio('Fit type', fit_types.keys(),horizontal=True,index=0)

			if filt_app == 1:
				filt_type = st.selectbox('Filter type', ['Low-pass','High-pass','Band-pass'],index=0)
				filt_order = st.slider('Filter order',4,12,8)
				col1_t2, col2_t2 = st.columns(2)
				
				if filt_type == 'Low-pass':
					fmin = col1_t2.number_input('Lower limit of the filter', min_value=0.0, max_value=f_max, value=0.0,disabled=True)  # min, max, default
					fmax = col2_t2.number_input('Upper limit of the filter', min_value=0.0, max_value=f_max, value=float(f_max/4))  # min, max, default
					sos = signal.butter(filt_order  , fmax, 'lowpass' , fs=fs , output='sos', analog=False)

				elif filt_type == 'High-pass':
					fmin = col1_t2.number_input('Lower limit of the filter', 0.0, f_max, value=float(f_max/4))  # min, max, default
					fmax = col2_t2.number_input('Upper limit of the filter', 0.0, f_max, value=float(f_max/2),disabled=True)  # min, max, default
					sos = signal.butter(filt_order  , fmin, 'highpass' , fs=fs , output='sos', analog=False)

				elif filt_type == 'Band-pass':
					fmin = col1_t2.number_input('Lower limit of the filter', 0.0, f_max, value=float(f_max/8))  # min, max, default
					fmax = col2_t2.number_input('Upper limit of the filter', 0.0, f_max, value=float(f_max/4))  # min, max, default
					sos = signal.butter(filt_order  , [fmin,fmax], 'bandpass' , fs=fs , output='sos', analog=False)
			else:
				sos = 0


			if sep_plots:
				for i in range(len(file)):		
					file[i].seek(0)
					data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
					if filt_app == 1:
						y = np.array(data[dof])
						t = np.array(data[tcol])
						y_doubled = np.zeros(2*len(y)-1)

						# Double the time series to avoid filter impact at the beginning
						y_doubled[len(y)-1:] = y
						y_doubled[:len(y)-1] = y[:0:-1]
						y_filt = signal.sosfiltfilt(sos, y_doubled)[len(y)-1:]


					else:
						y_filt = y = np.array(data[dof])
						t = np.array(data[tcol])

					# Define the time series limits for analysis
					time_filter = (t>=t_min) & (t<=t_max)

					# Estimate the offset and zero the time series
					offset = offset_estimator(y_filt[time_filter])
					y_zerod = y_filt - offset

					# Estimate the peaks of the zeroed time series
					peaks_time, peaks_amp = peaks_estimator(t,y_zerod[time_filter],peaks_types[peaks_type],t_min==0)

					# Estiamte of the dynamic properties of the free decay
					xi_est, f_est = dynamic_estimator(peaks_time,peaks_amp,peaks_type=peaks_types[peaks_type])

					# Compute the power spectrum of the time series for analysis (for representation purposes only)
					f, Pxx = signal.welch(y[time_filter], fs, nperseg=nfft , scaling='spectrum') 
					f, Pxx_filt = signal.welch(y_filt[time_filter], fs, nperseg=nfft , scaling='spectrum') 


					# Create the figure to plot
					fig = free_decay_plot(t,y,y_filt,offset,
										  peaks_time,peaks_amp,
										  xi_est,f_est,
										  f,Pxx, Pxx_filt,
										  fs,sos,f_max,filt_app,
										  time_filter,
										  fit_types[fit_type])

					st.pyplot(fig)