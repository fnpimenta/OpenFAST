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
			3. Add all the blades' DoF''')
	st.write(''' ''',unsafe_allow_html=True)
	st.write('''<div style="text-align: justify">
			\nUsing any of the simulations above, estimate the structural natural frequency and damping coefficient for the tower 1$^{st}$ FA mode.
			\nIf you have time, you may repeat the analysis with the AeroDyn module enabled and different pitch angles.
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
	st.write('''<div style="text-align: justify">
		\nNote that you may need to reduce the time range for the modal analysis or filter the data to obtain better results.
		</div>''',unsafe_allow_html=True)


exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	try:
		create_pdf_week1_2(figs,report_text,'Task 2: Free decay analysis','Task2_report',exp_c[1],exp,file_id+1)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")
