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
apptitle = 'OpenFAST Course - Task 4'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon , layout='wide')

st.title('Task 4 - Free decay analysis with AeroDyn')

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
			\nTo better understand the impact of the Aerodyn modules, you will repeat the analysis from Task 3, but now taking into account the aerodynamic forces on the rotor.
			Run a free decay analysis with the following pitch conditions:</div>''',unsafe_allow_html=True)
	st.write(r'''
			1. 90$º$ pitch angle
			2. 0$º$ pitch angle''')
	st.write(''' ''',unsafe_allow_html=True)
	st.write('''<div style="text-align: justify">
			\nEstimate the damping coefficient for the 2 cases above. **Do not forget to enable the Aerodyn module.**
			\n Once you have finished, you may download a report of the analysis.
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

file_OpenFAST = open('./OpenFAST_models/' + ref_path + '/' + 'TestFile.fst', 'r')
file_struct = open('./OpenFAST_models/' + ref_path + '/' + '01_ElastoDyn' + '/' + 'WP_ElastoDyn.dat', 'r')
file_wind =  open('./OpenFAST_models/' + ref_path + '/' + '02_InflowWind' + '/' + 'InflowWind_W0500_Steady.dat', 'r')
file_aero = open('./OpenFAST_models/' + ref_path + '/' + '03_AeroDyn' + '/' + 'WP_AeroDyn.dat', 'r')
checkfile = 1

with st.expander("**Hints**",False):

	st.write('''<div style="text-align: justify">
	\nThis task mimics most of the analysis conducted in Task 2, with some slightly modifications to include the aerodynamic forces computation.
1. Modify the **OpenFAST input file**:
	- Modify the comment line for a description of your simulation
	- Activate the AeroDyn module in <b>FEATURE SWITCHES AND FLAGS</b>
	- Include the paths to the corresponding input files <b>INPUT FILES</b> section
	</div>''',unsafe_allow_html=True)

	#checkfile = st.checkbox('**Show input file details**')
	if checkfile:
		data = []
		for line in file_OpenFAST:
			data.append(line)

		
		tab1,tab2,tab3,tab4 = st.tabs(['**Simulation Control**',
									   '**Feature switches and flags**',
									   '**Input files**',
									   'Output'])

		all_idx = range(3,11)
		on_sel_idx = []
		off_sel_idx = []
		with tab1:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(12,20)
		on_sel_idx = [12,14]
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(21,32)
		on_sel_idx = [21,26]
		off_sel_idx = []
		with tab3:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(33,41)
		on_sel_idx = []
		off_sel_idx = []
		with tab4:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		st.divider()

	st.write('''<div style="text-align: justify">	
		\n
2. Modify the **AeroDyn input file**:
	- Since you are making a parked simulation, disable the indcution wake model (WakeMod) 
	and unsteady aerodynamics (AFAeroMod) in the <b>GENERAL CONDITIONS</b> section
			</div>''',unsafe_allow_html=True)
	#checkfile = st.checkbox('**Show AeroDyn file details**')
	if checkfile:
		data = []
		for line in file_aero:
			data.append(line)

		tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['**General options**',
									   'Environmental conditions',
									   'BEMT options',
									   'Airfoil information',
									   'Blade properties',
									   'Tower aeroyncamis'])

		all_idx = range(3,14)
		on_sel_idx = [5,6]
		off_sel_idx = []
		with tab1:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])


		all_idx = range(15,21)
		on_sel_idx = []
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(22,31)
		on_sel_idx = []
		off_sel_idx = []
		with tab3:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])


		all_idx = range(40,51)
		on_sel_idx = []
		off_sel_idx = []
		with tab4:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])


		all_idx = range(52,56)
		on_sel_idx = []
		off_sel_idx = []
		with tab5:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])


		all_idx = range(57,62)
		on_sel_idx = []
		off_sel_idx = []
		with tab6:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		st.divider()



	st.write('''<div style="text-align: justify">	
		\n
3. Run the OpenFAST simulation and upload the output files.
			</div>''',unsafe_allow_html=True)


with st.expander("**Data analysis**",True):
	st.write('Uploaded the output files from OpenFAST')

	cols0 = st.columns(2)
	file = []
	file.append(cols0[0].file_uploader("90$º$ pitch",accept_multiple_files=False))
	file.append(cols0[1].file_uploader("0$º$ pitch",accept_multiple_files=False))
	
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


		cols = st.columns([0.25,0.25,0.5])

		nfft = cols[2].select_slider('FFT number of points',[int(2**x) for x in np.arange(nmin,nmax)],int(2**(nmax-3)))

		fs = 1/np.array(data[tcol][1]-data[tcol][0])

		make_analysis = 1
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

			for file_id in range(2):
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
					cols0[file_id].pyplot(fig_decay)


				else:
					cols[file_id].warning('File %d has not been uploaded properly.'%(file_id+1), icon="⚠️")


with st.expander('**See explanation**',False):
	st.write(r'''
		The free equation of motion (without any external load) for a single degree of freedom system is given by:
		$$
			m\ddot{x}(t) = -kx(t) - f_d(t)
		$$
		where $f_d(t)$ is the damping force, here taken to be a generic function of time.
		In the simplified numerical simulation above, the damping force is assumed to be such that:
		$$
		   f_d(t) = c_1x(t) + c_2|\dot{x}(t)-u_r|(\dot{x}(t)-u_r)
		$$
		where $u_r$ is external flow velocity.

		For the linear damping model ($c_2=0$), the differential equation above has the well known solution:
		$$
			x(t) = Ae^{-\xi\omega_0 t}\cos(w\sqrt{1-\xi^2}t+\phi) = Ae^{-\xi\omega_0 t}\cos(w_dt+\phi)
		$$
		where $\omega_0=\sqrt{\frac{k}{m}}$ is the system undamped natural frequency and $\xi$ is the damping ratio,
		defined as the ratio between the damping coefficient and its critical value as:
		$$
			\xi=\frac{c_1}{c_{cr}}=\frac{c_1}{2m\omega_0}
		$$
		One may immediately see that the response is given by a periodic function modulated by a negative exponential,
		implying that $\xi$ can be evaluated through the response amplitude, since:
		$$
			\ln(Ae^{-\xi\omega_0 t}) = \ln(A) - \xi\omega_0t
		$$
		meaning that the envelope amplitde natural logarithm, here obtained through the peak value in every oscillation, is a linear function of time with slope $-\xi\omega_0$.

		Although this is no longer true if $c_2\neq0$, for low damping forces, one may still make some general considerations based on energy dissipation.
		Firslty, it should be noted that the energy dissipation over a full cyle may be written as:
		$$
			W = \int_Tf_d(t)dx
		$$
		For the linear damping contribution, one finds:
		$$
			W_l = c_1\int_T\dot{x}(t)dx = c_1\int_T\dot{x}^2dt \approx  c_1A^2\omega_0^2\int_T\sin^2(\omega_0 t)dt = c_1\left(A^2\omega_0\pi\right)
		$$
		where it was assumed that to first order the motion may be approximated over a cycle as $x(t)=A\cos(\omega_0t)$.
		Under the same assumption, the quadratic contribution, for $u_r=0$, may be obtained as:
		$$
			W_q = c_2\int_T|\dot{x}(t)|\dot{x}(t)dx = 2c_2\int_{T/2}\dot{x}^3dt \approx 2c_2A^3\omega_0^3\int_{T/2}\sin^3(\omega_0t)dt = c_2 \frac{8A^3\omega_0^2}{3}
		$$
		By comparison with the linear damping result, it follows that the linear coefficient that best approximates the quadratic response in terms of energy dissipation is:
		$$
			 \tilde{c} = c_2 \frac{8\omega_0}{3\pi} A
		$$
		From the expression above, it can be seen that for a purely quadratic damping force, a linear dependency on the motion amplitude is expected.
	''')



exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	try:
		create_pdf_task4(figs,report_text,'Task 4: Free decay analysis with AeroDyn','Task4_report',exp_c[1],exp)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")
