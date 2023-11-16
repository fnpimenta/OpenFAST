import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import signal, interpolate

from modes import *
from estimators import *
from plot_generators import *
from Print import *

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST Course - Task 18'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 18 - Mooring system properties')

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
			\nThe mooring system effective stiffness depends strongly on the cables' mass and their unstretched length.
			Repeat Task 16 with the same initial conditions (1m tower top FA displacement and 10m platform surge), but reduce the cbales unstretched length from 835.35m to 825.35m.
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
ref_path = ref_models['NREL 5MW']

file_OpenFAST = open('./OpenFAST_models/' + ref_path + '/' + 'TestFile.fst', 'r')
file_struct = open('./OpenFAST_models/' + ref_path + '/' + '01_ElastoDyn' + '/' + 'NREL5MW_ElastoDyn.dat', 'r')
file_wind =  open('./OpenFAST_models/' + ref_path + '/' + '02_InflowWind' + '/' + 'InflowWind_W0500_Steady.dat', 'r')
file_aero = open('./OpenFAST_models/' + ref_path + '/' + '03_AeroDyn' + '/' + 'NREL5MW_AeroDyn.dat', 'r')
file_hydro = open('./OpenFAST_models/' + ref_path + '/' + '05_HydroDyn' + '/' + 'NREL_OC4_HydroDyn.dat', 'r')
file_mooring = open('./OpenFAST_models/' + ref_path + '/' + '06_Mooring' + '/' + 'NREL5MW_OC4_MoorDyn.dat', 'r')

onshore_color = "#cc000033"
offshore_color = "#0a75ad33"

checkfile=1

with st.expander("**Hints**",False):

	st.write('''<div style="text-align: justify">
	\nThe suggested procedure for the floating wind turbine simulation is the following:
1. Modify the **OpenFAST input file**:
	- Modify the comment line for a description of your simulation
	</div>''',unsafe_allow_html=True)

	st.divider()

	st.write('''<div style="text-align: justify">	
		\n
2. Modify the **ElastoDyn input file**:
	- **Add** a initial displacement of **10m in the surge direction** in <b>INITIAL CONDITIONS</b>.
	- Verify that the output list defined in <b>OUTPUT</b> included the values that you want to estimate. You may find a list of all the available output variables in the file <code>OutListParameters.xlsx</code>
	- Ensure that you have the following outputs:
		- Platform motions: "PtfmTDxt", "PtfmTDyt", "PtfmTDzt", "PtfmRDxi", "PtfmRDyi" and "PtfmRDzi"
		- Tower base bending moments: "TwrBsMxt" and "TwrBsMyt"
		- FA and SS tower-top displacements: "YawBrTDxp" and "YawBrTDyp"
		- FA and SS tower-top accelerations: "YawBrTAxp" and "YawBrTAyp"
			</div>''',unsafe_allow_html=True)
	#checkfile = st.checkbox('**Show ElastoDyn file details**')
	if checkfile:
		data = []
		for line in file_struct:
			data.append(line)

		
		tab1,tab2,tab3,tab4,tab5 = st.tabs(['Simulation Control',
									   		'Degrees of freedom',
									   		'**Initial conditions**',
									   		'Turbine configuration',
									   		'Mass and inertia'])
		all_idx = range(3,6)
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

		all_idx = range(9,26)
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

		all_idx = range(27,44)
		on_sel_idx = [36,38]
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

		all_idx = range(72,85)
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

		st.divider()


	st.write('''<div style="text-align: justify">	
		\n
3. Modify the **HydroDyn input file**:
	- Set the waves mode to still waters in <b>WAVES</b> section.
	</div>''',unsafe_allow_html=True)
	#checkfile = st.checkbox('**Show ElastoDyn file details**')
	if checkfile:
		data = []
		for line in file_hydro:
			data.append(line)

		
		tab1,tab2,tab3,tab4= st.tabs(['Environmental conditions',
									  '**Waves**',
									  'Current',
									  'Floating platform'])
		all_idx = range(4,7)
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

		all_idx = range(8,29)
		on_sel_idx = [8]
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(37,45)
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

		all_idx = range(46,62)
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
4. Modify the **MoorDyn input file**:
	- Modify the cables unstretched length in <b>LINE PROPERTIES</b> section.
	</div>''',unsafe_allow_html=True)
	#checkfile = st.checkbox('**Show ElastoDyn file details**')
	if checkfile:
		data = []
		for line in file_mooring:
			data.append(line)

		
		tab1,tab2,tab3 = st.tabs(['Line types',
								  'Connection properties',
								  '**Line properties**'])
		all_idx = range(4,8)
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

		all_idx = range(9,18)
		on_sel_idx = [8]
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(19,25)
		on_sel_idx = [22,23,24]
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
		\n
5. Run the OpenFAST simulation and upload the output files.
			</div>''',unsafe_allow_html=True)


with st.expander("**Data analysis**",True):
	st.write('Upload the output files from OpenFAST')

	cols0 = st.columns(2)
	file = []
	file.append(cols0[0].file_uploader("Reference mooring system",accept_multiple_files=False))
	file.append(cols0[1].file_uploader("Modified mooring system",accept_multiple_files=False))
	
	error_check = 0
	input_error = np.zeros(8)-2

	nmaxs = np.zeros(len(file))

	for i in range(len(file)):
		if not(file[i]==None):
			error_check += 1
			data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)
			nmaxs[i] = len(data)

	if error_check>0:
		cols = st.columns(2)
		t_min = cols[0].number_input('First time instant to plot',0.0,1000.0,0.0)
		t_max = cols[0].number_input('Last time instant to plot',0.0,1000.0,1000.0)
		
		nmin = 4
		nmax = np.max((int(np.log2(np.max(nmaxs))) + 1,8))

		nfft = cols[1].select_slider('FFT number of points',[int(2**x) for x in np.arange(nmin,nmax)],int(2**(nmax-1)))
		f_max = cols[1].number_input('Maximum frequency to plot',0.0,None,1.0)
				
		fig = plt.figure(figsize = (12,16))

		gs = gridspec.GridSpec(4,2,wspace=0.25,hspace=0.1)

		ax1 = plt.subplot(gs[0,0])
		ax2 = plt.subplot(gs[0,1])	

		ax3 = plt.subplot(gs[1,0])
		ax4 = plt.subplot(gs[1,1])

		ax5 = plt.subplot(gs[2,0])
		ax6 = plt.subplot(gs[2,1])	

		ax7 = plt.subplot(gs[3,0])
		ax8 = plt.subplot(gs[3,1])	

		labels = ['Reference mooring system', 'Modified mooring system']
		for i in range(len(file)):
			if not(file[i]==None):
				
				file[i].seek(0)
				data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)

				file[i].seek(0)
				units = pd.read_csv(file[i], skiprows=[0,1,2,3,4,5] , nrows=1,delimiter=r"\s+")
				
				time_filter = (data.Time>=t_min) & (data.Time<=t_max)
				fs = 1/np.array(data.Time[1]-data.Time[0])
					
				try:
					ax1.plot(data.Time[time_filter],data.PtfmTDxt[time_filter])
					ax1.set_ylabel('Platform sugre motion\n%s'%units.PtfmTDxt.iloc[0])

					f, Pxx = signal.welch(data.PtfmTDxt[time_filter], fs , nperseg=nfft , scaling='spectrum')
					ax2.semilogy(f[f<=f_max],Pxx[f<=f_max])
				except:
					input_error[0] += 1
					input_error[1] += 1

				try:
					ax3.plot(data.Time[time_filter],data.PtfmRDyi[time_filter])
					ax3.set_ylabel('Platform pitch motion\n%s'%units.PtfmRDyi.iloc[0])

					f, Pxx = signal.welch(data.PtfmRDyi[time_filter], fs , nperseg=nfft , scaling='spectrum')
					ax4.semilogy(f[f<=f_max],Pxx[f<=f_max])
				except:
					input_error[2] += 1	
					input_error[3] += 1				

				try:
					ax5.plot(data.Time[time_filter],data.YawBrTDxp[time_filter],label=labels[i])
					ax5.set_ylabel('Tower top FA displacement\n%s'%units.YawBrTDxp.iloc[0])

					f, Pxx = signal.welch(data.YawBrTDxp[time_filter], fs , nperseg=nfft , scaling='spectrum')
					ax6.semilogy(f[f<=f_max],Pxx[f<=f_max])					
				except:
					input_error[4] += 1
					input_error[5] += 1

				try:
					ax7.plot(data.Time[time_filter],data.TwrBsMyt[time_filter],label=labels[i])
					ax7.set_ylabel('Tower base FA bending moment\n%s'%units.TwrBsMyt.iloc[0])
				
					f, Pxx = signal.welch(data.TwrBsMyt[time_filter], fs , nperseg=nfft , scaling='spectrum')
					ax8.semilogy(f[f<=f_max],Pxx[f<=f_max])

				except:
					input_error[6] += 1
					input_error[7] += 1

			else:
				input_error += 1

		ax1.set_xticklabels('')
		ax2.set_xticklabels('')
		ax3.set_xticklabels('')
		ax4.set_xticklabels('')
		ax5.set_xticklabels('')
		ax6.set_xticklabels('')

		ax7.set_xlabel('Time (s)')
		ax8.set_xlabel('Frequency (Hz)')

		ax1.set_title('Time domain')
		ax2.set_title('Frequency domain')

		if input_error[0] == 0:
			ax1.annotate('No data found for platform surge motion',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[1] == 0:
			ax2.annotate('No data found for platform surge motion',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[2] == 0:
			ax3.annotate('No data found for platform pitch motion',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[3] == 0:
			ax4.annotate('No data found for platform pitch motion',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[4] == 0:
			ax5.annotate('No data found for tower top FA displacement',(0.5,0.5),ha='center',xycoords='axes fraction')

		if input_error[5] == 0:
			ax6.annotate('No data found for tower top FA displacement',(0.5,0.5),ha='center',xycoords='axes fraction')

		if input_error[6] == 0:
			ax6.annotate('No data found for tower FA bending moments',(0.5,0.5),ha='center',xycoords='axes fraction')
	
		if input_error[7] == 0:
			ax6.annotate('No data found for tower FA bending moments',(0.5,0.5),ha='center',xycoords='axes fraction')

		ax7.legend(loc='upper center',
		 			bbox_to_anchor=(1.1,-0.2),
					ncol=3,
					fancybox=False,
					framealpha=1,
					frameon=False)

		st.pyplot(fig)

exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	try:
		create_pdf_task18(fig,report_text,'Task 18: Mooring system properties','Task18_report',exp_c[1],exp)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")