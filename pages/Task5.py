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
apptitle = 'OpenFAST Course - Task 5'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 5 - Parked simulations for different wind speeds')

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
			\nWe are now in conditions to perform some simple **parked** simulations. Prepare all the files needed to estimate the bending moments at the tower base and the thrust force in the rotor for a the complex wind field generated in Task 3.
			\nThe suggested procedure is the following:
1. Modify the OpenFAST input file:
	- Modify the comment line for a description of your simulation
	- Modify the simulation time to 1000s (do not forget that we will reject the first 400s to ensure a physical solution) in the <b>SIMULATION CONTROL</b>
	- Activate the ElastoDyn, the AeroDyn and the InflowWind modules in <b>FEATURE SWITCHES AND FLAGS</b>
	- Include the paths to the corresponding input files <b>INPUT FILES</b> section
1. Modify the ElastoDyn file:
	- Disable the generator degree of freedom (GenDOF) in the <b>DEGREES OF FREEDOM</b> such that the rotor remains parked during the simulation
	- If you are using the file from the previous task, do not forget to remove the tower top initial displacement.
	- Verify that the output list defined in <b>OUTPUT</b> included the values that you want to estimate. You may find a list of all the available output variables in the file <code>09_AuxiliaryFiles\OutListParameters.xlsx</code>
	- Ensure that you have the following outputs:
  		- Blade 1 pitch angle: "BldPitch1"
  		- Rotor speed, torque and thrust: "RotSpeed", "RotTorq" and "RotThrust"                         
  		- Tower base FA and SS bending moments: "TwrBsMyt" and "TwrBsMxt"
1. Modify the InflowWind file:
	- Modify the wind type for a TurbSim FF (full field) wind in <b>Wind model</b>
	- Include the paths to the corresponding TurbSim generated (.bts) file in <b>Parameters for Binary TurbSim Full-Field files</b> section
1. Run the OpenFAST simulation and upload the output files.
			</div>''',unsafe_allow_html=True)

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

	cols0 = st.columns(2)
	file = []
	file.append(cols0[0].file_uploader("Steady wind",accept_multiple_files=False))
	file.append(cols0[1].file_uploader("TurbSim wind",accept_multiple_files=False))
	
	error_check = 0
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

				ax1.plot(data.Time,data.RotSpeed)
				ax2.plot(data.Time,data.RotTorq)

				ax3.plot(data.Time,data.BldPitch1)
				ax4.plot(data.Time,data.RotThrust)

				ax5.plot(data.Time,data.TwrBsMyt,label=labels[i])
				ax6.plot(data.Time,data.TwrBsMxt)

				ax1.set_ylabel('Rotor angular velocity ($\Omega$)\n%s'%units.RotSpeed.iloc[0])
				ax2.set_ylabel('Rotor torque\n%s'%units.RotTorq.iloc[0])
				ax3.set_ylabel('Blade pitch angle\n%s'%units.BldPitch1.iloc[0])
				ax4.set_ylabel('Rotor thrust\n%s'%units.RotThrust.iloc[0])
				ax5.set_ylabel('Tower base FA bending moment\n%s'%units.TwrBsMyt.iloc[0])
				ax6.set_ylabel('Tower base SS bending moment\n%s'%units.TwrBsMxt.iloc[0])

		ax1.set_xticklabels('')
		ax2.set_xticklabels('')
		ax3.set_xticklabels('')
		ax4.set_xticklabels('')

		ax5.set_xlabel('Time (s)')
		ax6.set_xlabel('Time (s)')

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
		create_pdf_task5(fig,report_text,'Task 5: Parked simulations','Task5_report',exp_c[1],exp)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")
