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
apptitle = 'OpenFAST Course - Task 7'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 7 - Normal operation simulations with controls')

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
			\nWe are now in conditions to perform some simple **normal operation** simulations without the need to impose the relevant operation conditions. 
			Prepare all the files needed to estimate the bending moments at the tower base and the thrust force in the rotor for the complex wind field you have generated in Task 3 and used for Task 6.
			At the end, you can compare the results with the ones you have for Task 6, where the control systems were not implemented.
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

file_OpenFAST = open('./OpenFAST_models/' + ref_path + '/' + 'TestFile.fst', 'r')
file_struct = open('./OpenFAST_models/' + ref_path + '/' + '01_ElastoDyn' + '/' + 'WP_ElastoDyn.dat', 'r')
file_wind =  open('./OpenFAST_models/' + ref_path + '/' + '02_InflowWind' + '/' + 'InflowWind_W0500_Steady.dat', 'r')
file_aero = open('./OpenFAST_models/' + ref_path + '/' + '03_AeroDyn' + '/' + 'WP_AeroDyn.dat', 'r')

onshore_color = "#cc000033"
offshore_color = "#0a75ad33"

checkfile=1

with st.expander("**Hints**",False):

	st.write('''<div style="text-align: justify">
	\nThe suggested procedure is the following:
1. Modify the **OpenFAST input file**:
	- Modify the comment line for a description of your simulation
	- Add the ServoDyn module in <b>FEATURE SWITCHES AND FLAGS</b>
	- Include the paths to the corresponding input file in the <b>INPUT FILES</b> section
	</div>''',unsafe_allow_html=True)

	#checkfile = st.checkbox('**Show input file details**')
	if checkfile:
		data = []
		for line in file_OpenFAST:
			data.append(line)

		
		tab1,tab2,tab3,tab4 = st.tabs(['Simulation Control',
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
		on_sel_idx = [15]
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
		on_sel_idx = [27]
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
2. Modify the **ElastoDyn input file**:
	- **Enable the generator degree of freedom (GenDOF)** in the <b>DEGREES OF FREEDOM</b> such that the rotor angular velocity may vary across the simulation
	- To enhanced model convergence, keep the initial pitch value and rotor angular velocity for the values compatible with your wind speed in the <b>INITIAL CONDITIONS</b>.
	- Verify that the output list defined in <b>OUTPUT</b> included the values that you want to estimate. You may find a list of all the available output variables in the file <code>09_AuxiliaryFiles\OutListParameters.xlsx</code>
	- Ensure that you have the following outputs:
  		- Blade 1 pitch angle: "BldPitch1"
  		- Rotor speed, torque and thrust: "RotSpeed", "RotTorq" and "RotThrust"                         
  		- Tower base FA and SS bending moments: "TwrBsMyt" and "TwrBsMxt"
			</div>''',unsafe_allow_html=True)
	#checkfile = st.checkbox('**Show ElastoDyn file details**')
	if checkfile:
		data = []
		for line in file_struct:
			data.append(line)

		
		tab1,tab2,tab3,tab4,tab5 = st.tabs(['Simulation Control',
									   		'**Degrees of freedom**',
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
		on_sel_idx = [14]
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
		on_sel_idx = [29,30,31,34]
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
3. Modify the **AeroDyn input file**:
	- Since you are making a normal operation simulation, enable the indcution wake model (WakeMod) 
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
4. Modify the **InflowWind input file** (or use the same from Task 6):
	- Modify the wind type for TurbSim FF (full field) wind in <b>Wind model</b>
	- Include the paths to the corresponding TurbSim generated (.bts) file in <b>Parameters for Binary TurbSim Full-Field files</b> section
			</div>''',unsafe_allow_html=True)

	#checkfile = st.checkbox('**Show InflowWind file details**')
	if checkfile:
		data = []
		for line in file_wind:
			data.append(line)


		tab1,tab2,tab3,tab4 = st.tabs(['**Wind model**',
									   '**Steady wind properties**',
									   'Uniform wind properties',
									   '**TurbSim full field**'])

		all_idx = range(3,10)
		on_sel_idx = [4]
		off_sel_idx = []
		with tab1:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(11,14)
		on_sel_idx = [11,12,13]
		off_sel_idx = []
		with tab2:
			for i in all_idx:
				if i in on_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(onshore_color,data[i]),unsafe_allow_html=True)
				elif i in off_sel_idx:
					st.write('<span style="background-color: %s">%s</span>'%(offshore_color,data[i]),unsafe_allow_html=True)
				else:
					st.write(data[i])

		all_idx = range(15,18)
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

		all_idx = range(19,20)
		on_sel_idx = [19]
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
5. Run the OpenFAST simulation and upload the output files.
			</div>''',unsafe_allow_html=True)


with st.expander("**Data analysis**",True):
	st.write('Upload the output files from OpenFAST')

	cols0 = st.columns(2)
	file = []
	file.append(cols0[0].file_uploader("Simulation with controls",accept_multiple_files=False))
	file.append(cols0[1].file_uploader("Simulation without controls (Task 6)",accept_multiple_files=False))
	
	error_check = 0
	input_error = np.zeros(6)-2

	for i in range(len(file)):
		if not(file[i]==None):
			error_check += 1

	if error_check>0:
		cols = st.columns(2)
		t_min = cols[0].number_input('First time instant to plot',0.0,1000.0,0.0)
		t_max = cols[1].number_input('Last time instant to plot',0.0,1000.0,1000.0)
		
		fig = plt.figure(figsize = (12,12))

		gs = gridspec.GridSpec(3,2,wspace=0.25,hspace=0.1)

		ax1 = plt.subplot(gs[0,0])
		ax2 = plt.subplot(gs[0,1])	

		ax3 = plt.subplot(gs[1,0])
		ax4 = plt.subplot(gs[1,1])

		ax5 = plt.subplot(gs[2,0])
		ax6 = plt.subplot(gs[2,1])	

		labels = ['Simulation with controls', 'Simulation without controls']
		for i in range(len(file)):
			if not(file[i]==None):
				
				file[i].seek(0)
				data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)

				file[i].seek(0)
				units = pd.read_csv(file[i], skiprows=[0,1,2,3,4,5] , nrows=1,delimiter=r"\s+")
				
				time_filter = (data.Time>=t_min) & (data.Time<=t_max)

				try:
					ax1.plot(data.Time[time_filter],data.RotSpeed[time_filter])
					ax1.set_ylabel('Rotor angular velocity ($\Omega$)\n%s'%units.RotSpeed.iloc[0])
				except:
					input_error[0] += 1

				try:
					ax2.plot(data.Time[time_filter],data.RotTorq[time_filter])
					ax2.set_ylabel('Rotor torque\n%s'%units.RotTorq.iloc[0])
				except:
					input_error[1] += 1

				try:
					ax3.plot(data.Time[time_filter],data.BldPitch1[time_filter])
					ax3.set_ylabel('Blade pitch angle\n%s'%units.BldPitch1.iloc[0])
				except:
					input_error[2] += 1				

				try:
					ax4.plot(data.Time[time_filter],data.RotThrust[time_filter])
					ax4.set_ylabel('Rotor thrust\n%s'%units.RotThrust.iloc[0])
				except:
					input_error[3] += 1

				try:
					ax5.plot(data.Time[time_filter],data.TwrBsMyt[time_filter],label=labels[i])
					ax5.set_ylabel('Tower base FA bending moment\n%s'%units.TwrBsMyt.iloc[0])
				except:
					input_error[4] += 1

				try:
					ax6.plot(data.Time[time_filter],data.TwrBsMxt[time_filter])
					ax6.set_ylabel('Tower base SS bending moment\n%s'%units.TwrBsMxt.iloc[0])
				except:
					input_error[5] += 1
			else:
				input_error += 1

		ax1.set_xticklabels('')
		ax2.set_xticklabels('')
		ax3.set_xticklabels('')
		ax4.set_xticklabels('')

		ax5.set_xlabel('Time (s)')
		ax6.set_xlabel('Time (s)')

		if input_error[0] == 0:
			ax1.annotate('No data found for rotor angular velocity',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[1] == 0:
			ax2.annotate('No data found for rotor torque',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[2] == 0:
			ax3.annotate('No data found for blade pitch angle',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[3] == 0:
			ax4.annotate('No data found for rotor thrust',(0.5,0.5),ha='center',xycoords='axes fraction')
		
		if input_error[4] == 0:
			ax5.annotate('No data found for tower FA bending moments',(0.5,0.5),ha='center',xycoords='axes fraction')

		if input_error[5] == 0:
			ax6.annotate('No data found for tower SS bending moments',(0.5,0.5),ha='center',xycoords='axes fraction')


		ax5.legend(loc='upper center',
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
		create_pdf_task7(fig,report_text,'Task 7: Normal operation simulations with ServoDyn','Task7_report',exp_c[1],exp)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")