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
from Print import *

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST Course - Task 6'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 6 - Generate a 3D full wind field with TurbSim.')

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
			\nWe are now in conditions to perform some simple simulations. Prepare all the files needed to estimate the bending moments at the tower base and the thrust force in the rotor for a <b>steady wind</b>.
			\nSince the wind turbine controls are not yet implemented, you will have to define the adequate initial conditions for the simulation. 
			Additionally, as the aerodynamic torque sourced by the lift forces is still unbalanced, once again because the wind turbine controls are still not implemented, you will have to disable the generator degree of freedom to ensure that it will not accelerate indefinitely. 
			The suggested procedure is the following:
1. Modify the OpenFAST input file:
	- Modify the comment line for a description of your simulation
	- Modify the simulation time to 1000s (do not forget that we will reject the first 400s to ensure a physical solution) in the <b>SIMULATION CONTROL</b>
	- Activate the ElastoDyn, the AeroDyn and the InflowWind modules in <b>FEATURE SWITCHES AND FLAGS</b>
	- Include the paths to the corresponding input files <b>INPUT FILES</b> section
1. Modify the ElastoDyn file:
	- Disable the generator degree of freedom (GenDOF) in the <b>DEGREES OF FREEDOM</b> such that the rotor angular velocity prescribed is kept constant across the simulation
	- Modify the initial pitch value and rotor angular velocity for the values compatible with your wind speed in the <b>INITIAL CONDITIONS</b> (if you are using the file from the previous task, do not forget to remove the tower top initial displacement).
	- Verify that the output list defined in <b>OUTPUT</b> included the values that you want to estimate. You may find a list of all the available output variables in the file <code>09_AuxiliaryFiles\OutListParameters.xlsx</code>
1. Modify the InflowWind file:
	- Modify the wind type for a steady wind in <b>Wind model</b>
	- Modify the wind speed reference value for your wind speed in <b>Parameters for Steady Wind Conditions</b>
1. Run the OpenFAST simulation and upload the output files.
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

with st.expander("**Hints**",False):
	st.write('''<div style="text-align: justify">
			\nTo solve the tasks above you will need to prepare and run 3 different OpenFAST simulations.
			The **relevant files** to edit are listed below and the **relevant parameters and sections highlighted**.
			\nDo not forget to modify the **simulation length** and select **only the ElastoDyn module in the OpenFAST input file**.
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

	all_idx = range(27,44)
	on_sel_idx = [29,30,31,34,36]
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
		file[0].seek(0)
		units = pd.read_csv(file[0], skiprows=[0,1,2,3,4,5] , nrows=1,delimiter=r"\s+")


		keys = data.columns
		nvar = len(keys)

		cols = st.columns(2)

		tcol = cols[0].selectbox('Time column', data.columns,index=0)
		dof = cols[1].selectbox('Data column', data.columns,index=1)

		t_min,t_max = cols[0].slider('Time range',0.0,float(data[tcol].iloc[-1]),(0.0,float(data[tcol].iloc[-1])))
		f_min,f_max = cols[1].slider('Frequency range',0.0,float(0.5/(data[tcol][1]-data[tcol][0])),(0.0,float(0.25/(data[tcol][1]-data[tcol][0]))))

		tfilter = (data[tcol]>=t_min) & (data[tcol]<=t_max)

		nmin = 4
		nmax = np.max((int(np.log2(len(data[tfilter]))) + 1,8))

		nfft = cols[1].select_slider('FFT number of points',[int(2**x) for x in np.arange(nmin,nmax)],int(2**(nmax-3)))

		tabs = st.tabs(['Time series analysis','Modal analysis'])

		fs = 1/np.array(data[tcol][1]-data[tcol][0])

		with tabs[0]:

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

					ax1.plot(data[tcol],data[dof],label='Simulation %d'%(i+1))
					ax2.semilogy(f,Pxx)

				ax1.set_xlim(t_min,t_max)
				ax2.set_xlim(f_min,f_max)

				ax1.set_xlabel('Time (s)')
				ax2.set_xlabel('Frequency (Hz)')

				ax1.set_ylabel('%s %s'%(dof,units[dof].iloc[0]))
				ax2.set_ylabel('PSD')

			ax1.set_title('Time domain')
			ax2.set_title('Frequency domain')
			leg = ax1.legend(loc='upper center',
				 			bbox_to_anchor=(1.1,-0.2),
							ncol=3,
							fancybox=False,
							framealpha=1,
							frameon=False)


			figs.append(fig)
			st.pyplot(fig)

		with tabs[1]:
			cols = st.columns(4)

			file_id = int(cols[0].selectbox('File for the analysis',range(1,1+len(file))) - 1)
			#trange = cols[1].slider('Time range for the analysis',t_min,t_max,t_max)


			#make_analysis = st.checkbox('Run analysis')
			make_analysis = 1

			cols = st.columns(2)
			if make_analysis:

				peaks_types = {
							 "Positive peaks only": 0,
							 "All peaks": 1,
							}

				peaks_type = cols[0].radio('Peaks to use', peaks_types.keys(),horizontal=True)

				fit_types = {
							 "Trend line": 0,
							 "Mean value": 1,
							}

				fit_type = cols[1].radio('Fit type', fit_types.keys(),horizontal=True,index=0)

				cols = st.columns(4)
				filt_type = cols[0].selectbox('Filter type', ['No filter','Low-pass','High-pass','Band-pass'],index=0)

				filt_order = cols[1].slider('Filter order',4,12,8,disabled=(filt_type=='No filter'))

				if filt_type == 'Low-pass':
					fmin = cols[2].number_input('Lower limit of the filter', min_value=0.0, max_value=f_max, value=0.0,disabled=True)  # min, max, default
					fmax = cols[3].number_input('Upper limit of the filter', min_value=0.0, max_value=f_max, value=float(f_max/4),disabled=(filt_type=='No filter'))  # min, max, default
					sos = signal.butter(filt_order  , fmax, 'lowpass' , fs=fs , output='sos', analog=False)

				elif filt_type == 'High-pass':
					fmin = cols[2].number_input('Lower limit of the filter', 0.0, f_max, value=float(f_max/4),disabled=(filt_type=='No filter'))  # min, max, default
					fmax = cols[3].number_input('Upper limit of the filter', 0.0, f_max, value=float(f_max/2),disabled=True)  # min, max, default
					sos = signal.butter(filt_order  , fmin, 'highpass' , fs=fs , output='sos', analog=False)

				elif filt_type == 'Band-pass':
					fmin = cols[2].number_input('Lower limit of the filter', 0.0, f_max, value=float(f_max/8),disabled=(filt_type=='No filter'))  # min, max, default
					fmax = cols[3].number_input('Upper limit of the filter', 0.0, f_max, value=float(f_max/4),disabled=(filt_type=='No filter'))  # min, max, default
					sos = signal.butter(filt_order  , [fmin,fmax], 'bandpass' , fs=fs , output='sos', analog=False)
				else:
					fmin = cols[2].number_input('Lower limit of the filter', min_value=0.0, max_value=f_max, value=0.0,disabled=True)  # min, max, default
					fmax = cols[3].number_input('Upper limit of the filter', 0.0, f_max, value=float(f_max/2),disabled=True)  # min, max, default
					sos = 0

				cols = st.columns(2)

				if nfiles[file_id]>=0:

					file[file_id].seek(0)
					data = pd.read_csv(file[file_id] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)

					if not(filt_type=='No filter'):

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
					peaks_time, peaks_amp = peaks_estimator(t[time_filter],y_zerod[time_filter],peaks_types[peaks_type],0)

					# Estiamte of the dynamic properties of the free decay
					xi_est, f_est = dynamic_estimator(peaks_time,peaks_amp,peaks_type=peaks_types[peaks_type])

					# Compute the power spectrum of the time series for analysis (for representation purposes only)
					f, Pxx = signal.welch(y[time_filter], fs, nperseg=nfft , scaling='spectrum')
					f, Pxx_filt = signal.welch(y_filt[time_filter], fs, nperseg=nfft , scaling='spectrum')


					# Create the figure to plot
					fig_decay = free_decay_plot(t,y,y_filt,offset,
										  peaks_time,peaks_amp,
										  xi_est,f_est,
										  f,Pxx, Pxx_filt,
										  fs,sos,f_max,not(filt_type=='No filter')*1,
										  time_filter,
										  fit_types[fit_type])

					figs.append(fig_decay)
					st.pyplot(fig_decay)


				else:
					st.warning('The selected file has not been uploaded properly.', icon="⚠️")


exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	try:
		create_pdf_task4(figs,report_text,'Task 2: Free decay analysis','Task2_report',exp_c[1],exp,file_id+1)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")
