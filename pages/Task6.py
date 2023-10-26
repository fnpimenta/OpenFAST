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
apptitle = 'OpenFAST Course - Task 6'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 6 - Normal operation simulations with prescribed initial conditions')

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

@st.cache_data()
def WPDataPlot(w):
	ws = np.arange(5,21)
	ps = np.ones_like(ws)*2.6
	rpm = np.ones_like(ws)*20.45

	ps[7:] = 6.88 , 9.95 , 12.13 , 14.09 , 15.80 , 17.41 , 18.91 , 20.32 , 21.64
	rpm[0:7] = 9.09 , 10.91 , 12.72 , 14.52 , 16.33 , 18.12 , 19.90

	fps = interpolate.interp1d(ws, ps)
	frpm = interpolate.interp1d(ws, rpm)

	fig = plt.figure(figsize = (9,3))
	gs = gridspec.GridSpec(1,2,wspace=0.25,hspace=0.1)

	ax1 = plt.subplot(gs[0,0])
	ax2 = plt.subplot(gs[0,1])

	ax1.plot(ws,ps)
	ax2.plot(ws,rpm)

	ax1.set_xlabel('Wind speed (m/s)')
	ax2.set_xlabel('Wind speed (m/s)')

	ax1.set_ylabel('Pitch angle (deg)')
	ax2.set_ylabel('$\Omega$ (rpm)')

	ax1.set_title('Pitch curve')
	ax2.set_title('Rotor angular velocity curve')

	ax1.plot(w,fps(w),'or')
	ax2.plot(w,frpm(w),'or')

	return fps , frpm , fig


with st.expander("**Objective**",True):

	st.write('''<div style="text-align: justify">
			\nWe are now in conditions to perform some simple **normal operation** simulations. Prepare all the files needed to estimate the bending moments at the tower base and the thrust force in the rotor for different wind models.
			\nSince the wind turbine controls are not yet implemented, you will have to define the adequate initial conditions (rotor angular velocity and blade pitch angle) for the simulation.
			The normal operation values for stationary conditions for the WindPACT wind turbine is presented below and you can use the input box to estimate the relevant quantities for the wind speed you have selected:
			</div>''',unsafe_allow_html=True)
	cols = st.columns(2)
	w = cols[0].number_input('Wind speed (m/s)',5.0,20.0,10.0)
	fps,frpm,fig = WPDataPlot(w)

	cols[1].markdown('\n')
	cols[1].markdown('Pitch value: %.2f$º$'%fps(w))
	cols[1].markdown('RPM value: %.2f rpm'%frpm(w))

	st.pyplot(fig)

	st.write('''<div style="text-align: justify">
			\nAdditionally, as the aerodynamic torque sourced by the lift forces is still unbalanced, once again because the wind turbine controls are still not implemented, you will have to disable the generator degree of freedom to ensure that it will not accelerate indefinitely. 
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
	- Modify the simulation time to 1000s (do not forget that we will reject the first 400s to ensure a physical solution is being analysed) in the <b>SIMULATION CONTROL</b>
	- Activate the ElastoDyn, the AeroDyn and the InflowWind modules in <b>FEATURE SWITCHES AND FLAGS</b>
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
		on_sel_idx = [5]
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
		on_sel_idx = [12,13,14]
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
		on_sel_idx = [21,25,26]
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
	- Disable the generator degree of freedom (GenDOF) in the <b>DEGREES OF FREEDOM</b> such that the rotor angular velocity prescribed is kept constant across the simulation
	- Modify the initial pitch value and rotor angular velocity for the values compatible with your wind speed in the <b>INITIAL CONDITIONS</b>.
	- If you are using the file from the previous task, do not forget to remove the tower top initial displacement.
	- Set the rotor angular velocity to 0 and the blades' pitch angle to 90 degrees in the <b>INITIAL CONDITIONS</b> section.
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
		on_sel_idx = [9,10,11,13,14,16,17,18,19]
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
4. Modify the **InflowWind input file** (or use the same from Task 5):
	- Modify the wind type for a steady wind (Simulation 1) and then for TurbSim FF (full field) wind (Simulation 2) in <b>Wind model</b>
	- For Simulation 1, modify the wind vertical profile for the same reference wind speed you have used to generated the TurbSim data in <b>Steady wind properties</b> and set the power law exponent to 0.20.
	- For Simulation 2, include the paths to the corresponding TurbSim generated (.bts) file in <b>Parameters for Binary TurbSim Full-Field files</b> section
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
	st.write('Uploaded the output files from OpenFAST')

	cols0 = st.columns(2)
	file = []
	file.append(cols0[0].file_uploader("Steady wind",accept_multiple_files=False))
	file.append(cols0[1].file_uploader("TurbSim wind",accept_multiple_files=False))
	
	error_check = 0
	input_error = np.zeros(6)-2

	for i in range(len(file)):
		if not(file[i]==None):
			error_check += 1

	if error_check>0:
		fig = plt.figure(figsize = (12,12))

		gs = gridspec.GridSpec(3,2,wspace=0.25,hspace=0.1)

		ax1 = plt.subplot(gs[0,0])
		ax2 = plt.subplot(gs[0,1])	

		ax3 = plt.subplot(gs[1,0])
		ax4 = plt.subplot(gs[1,1])

		ax5 = plt.subplot(gs[2,0])
		ax6 = plt.subplot(gs[2,1])	

		labels = ['Steady wind', 'TurbSim wind field']
		for i in range(len(file)):
			if not(file[i]==None):
				
				file[i].seek(0)
				data = pd.read_csv(file[i] , skiprows=[0,1,2,3,4,5,7] , delimiter=r"\s+",header=0)

				file[i].seek(0)
				units = pd.read_csv(file[i], skiprows=[0,1,2,3,4,5] , nrows=1,delimiter=r"\s+")
				
				try:
					ax1.plot(data.Time,data.RotSpeed)
					ax1.set_ylabel('Rotor angular velocity ($\Omega$)\n%s'%units.RotSpeed.iloc[0])
				except:
					input_error[0] += 1

				try:
					ax2.plot(data.Time,data.RotTorq)
					ax2.set_ylabel('Rotor torque\n%s'%units.RotTorq.iloc[0])
				except:
					input_error[1] += 1

				try:
					ax3.plot(data.Time,data.BldPitch1)
					ax3.set_ylabel('Blade pitch angle\n%s'%units.BldPitch1.iloc[0])
				except:
					input_error[2] += 1				

				try:
					ax4.plot(data.Time,data.RotThrust)
					ax4.set_ylabel('Rotor thrust\n%s'%units.RotThrust.iloc[0])
				except:
					input_error[3] += 1

				try:
					ax5.plot(data.Time,data.TwrBsMyt,label=labels[i])
					ax5.set_ylabel('Tower base FA bending moment\n%s'%units.TwrBsMyt.iloc[0])
				except:
					input_error[4] += 1

				try:
					ax6.plot(data.Time,data.TwrBsMxt)
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
		create_pdf_task6(fig,report_text,'Task 6: Normal operation simulations','Task6_report',exp_c[1],exp)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")