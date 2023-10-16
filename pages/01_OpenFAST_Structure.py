import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

import annotated_text
from annotated_text import annotated_text

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API - Course tasks'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon,layout='wide')

# Title the app
st.title('OpenFAST analysis')

fig_OpenFAST = Image.open('figures/OpenFAST_scheme.png')
fig_aero = Image.open('figures/OpenFAST_aero.png')
fig_control = Image.open('figures/OpenFAST_control.png')
fig_wind = Image.open('figures/OpenFAST_wind.png')
fig_struct = Image.open('figures/OpenFAST_struct.png')
fig_hydro = Image.open('figures/OpenFAST_hydro.png')
fig_mooring = Image.open('figures/OpenFAST_mooring.png')

figs = [fig_OpenFAST , fig_struct , fig_wind , fig_aero , fig_control , fig_hydro , fig_mooring]

cols = st.columns(2)
all_mod = ['OpenFAST' , 'ElastoDyn' , 'InflowWind' , 'AeroDyn' , 'ServoDyn' , 'HydroDyn' , 'MoorDyn']
sel_mod = cols[0].selectbox('Available modules', all_mod)

tab_idx = all_mod.index(sel_mod)

cols = st.columns(2)
cols[1].image(figs[tab_idx])

file_descriptions = ['''<div style="text-align: justify">
					The OpenFAST main file indicates all the modules to be used. 
					For each module a baseline file is required.
					\nThe total duration of the simulation and the time step are also defined here.
					</div>''']

file_descriptions.append('''<div style="text-align: justify">
					\nThe **ElastoDyn** module governs all the aspects related to the structural behaviour of structure.
					Here, are included the mass and stiffness distributions for the tower and the blades (defined int two additional files), 
					as well as the characterisation of the elements that are treated as rigid bodies (nacelle, hub...). 
					\nThe **initial conditions** of the simulations (rotor angular velocity, blade pitch, tower deflection...) are also defined here.
					</div>''')

file_descriptions.append('''<div style="text-align: justify">
					\nThe **InflowWind** module defines the properties of wind field that will be used in the simulations.
					Different wind types are allowed, but only the **steady wind speed**, that follows a power law profile, does not depend on any externally computed data.
					\nFor full 3D wind fields additional tools are required, in particular the <a href="https://www.nrel.gov/wind/nwtc/turbsim.html">TurbSim</a> generator.
					</div>''')

file_descriptions.append('''<div style="text-align: justify">
					\nThe **AeroDyn** module defines the aerodynamic properties of the blades and tower. 
					\nFor the tower, only the drag coefficient evolution over height is required.
					\nFor the blades *n* normalised aerofoil profiles (unitary chord) should be also provided that will be used to characterise the blade's cross section evolution over its span.
					</div>''')


file_descriptions.append('''<div style="text-align: justify">
					\nThe **ServoDyn** defines the control algorithms to be used in the simulation, and in particular the torque and pitch controls.

					\nThe simplest models are a built-in variable speed torque control and the <a href="https://www.nrel.gov/docs/fy06osti/32495.pdf">WindPACT control</a> 
					or the <a href="https://rosco.readthedocs.io/en/latest/index.html">NREL’s Reference OpenSource Controller (ROSCO) tool-set</a>  for the pitch control.
					\nSpecial events simulations may require a manual implementation of the desired controls behaviour.
					</div>''')

file_descriptions.append('''<div style="text-align: justify">
					\nThe **HydroDyn** defines governs everything that is related to the hydrodynamic response of the structure, except the mooring system, by definind the frequency dependent mass and damping terms (externally computed).
					Contrary to the wind case, where the wind field properties are defined separately from the aerodynamic ones, the sea states are also defined here.
					\nThis module also allows for additional stiffness and damping matrixes that can be used to mimic the foundation of an onshore wind turbine.
					</div>''')

file_descriptions.append('''<div style="text-align: justify">
					\nThe **MoorDyn** module defines the properties of the mooring system. Different cable models can be used.
					</div>''')


cols[0].markdown("### **Module description**")
cols[0].write(file_descriptions[tab_idx],unsafe_allow_html=True)

ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
ref_path = ref_models['WP 1.5MW']

all_dir = os.listdir('./OpenFAST_models/' + ref_path )

file_OpenFAST = open('./OpenFAST_models/' + ref_path + '/' + 'TestFile.fst', 'r')
file_aero = open('./OpenFAST_models/' + ref_path + '/' + all_dir[2] + '/' + 'WP_AeroDyn.dat', 'r')
file_control =  open('./OpenFAST_models/' + ref_path + '/' + all_dir[3] + '/' + 'WP_ServoDyn.dat', 'r')
file_wind =  open('./OpenFAST_models/' + ref_path + '/' + all_dir[1] + '/' + 'InflowWind_W0500_Steady.dat', 'r')
file_struct = open('./OpenFAST_models/' + ref_path + '/' + all_dir[0] + '/' + 'WP_ElastoDyn.dat', 'r')
#file_hydro = Image.open('figures/OpenFAST_hydro.png')
#file_mooring = Image.open('figures/OpenFAST_mooring.png')

PALETTE = [	"#ff4b4b",
			"#ffa421",
			"#ffe312",
			"#21c354",
			"#00d4b1",
			"#00c0f2",
			"#1c83e1",
			"#803df5",
			"#808495",]

if tab_idx == 0:
	data = []
	for line in file_OpenFAST:
		data.append(line)

	with st.expander("**File explorer**",False):
		tab1,tab2,tab3,tab4 = st.tabs(['Simulation Control',
									   'Feature switches and flags',
									   'Input files',
									   'Output'])

		all_idx = range(3,11)
		sel_idx = [5,6]
		with tab1:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(12,20)
		sel_idx = [12,13,14,15,16,18]
		with tab2:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(21,32)
		sel_idx = [21,25,26,27,28,30]
		with tab3:
			for i in all_idx:
				if i in sel_idx:
					if i>27:
						annotated_text(annotation(data[i], "" ,PALETTE[1]+"66"))
					else:
						annotated_text((data[i], "" ,PALETTE[4]+"66"))
				else:
					annotated_text(data[i])

		all_idx = range(33,41)
		sel_idx = [12,13,14,15,16,18]
		with tab4:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])


if tab_idx == 1:
	data = []
	for line in file_struct:
		data.append(line)

	with st.expander("**File explorer**",False):
		tab1,tab2,tab3,tab4,tab5 = st.tabs(['Simulation Control',
									   'Environmental conditions',
									   'Initial conditions',
									   'Turbine configuration',
									   'Mass and inertia'])

		all_idx = range(3,6)
		sel_idx = []
		with tab1:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(9,27)
		sel_idx = []
		with tab2:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(27,44)
		sel_idx = []
		with tab3:
			for i in all_idx:
				if i in sel_idx:
					if i>27:
						annotated_text(annotation(data[i], "" ,PALETTE[1]+"66"))
					else:
						annotated_text((data[i], "" ,PALETTE[4]+"66"))
				else:
					annotated_text(data[i])

		all_idx = range(45,71)
		sel_idx = []
		with tab4:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(72,85)
		sel_idx = []
		with tab5:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

if tab_idx == 2:
	data = []
	for line in file_wind:
		data.append(line)

	with st.expander("**File explorer**",False):
		tab1,tab2,tab3,tab4 = st.tabs(['Wind model',
									   'Steady wind properties',
									   'Uniform wind properties',
									   'TurbSim full field'])

		all_idx = range(3,10)
		sel_idx = []
		with tab1:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(11,14)
		sel_idx = []
		with tab2:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(15,18)
		sel_idx = []
		with tab3:
			for i in all_idx:
				if i in sel_idx:
					if i>27:
						annotated_text(annotation(data[i], "" ,PALETTE[1]+"66"))
					else:
						annotated_text((data[i], "" ,PALETTE[4]+"66"))
				else:
					annotated_text(data[i])

		all_idx = range(19,20)
		sel_idx = []
		with tab4:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

if tab_idx == 3:
	data = []
	for line in file_aero:
		data.append(line)

	with st.expander("**File explorer**",False):
		tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['General options',
									   'Environmental conditions',
									   'BEMT options',
									   'Airfoil information',
									   'Blade properties',
									   'Tower aeroyncamis'])

		all_idx = range(3,14)
		sel_idx = []
		with tab1:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(15,21)
		sel_idx = []
		with tab2:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(22,31)
		sel_idx = []
		with tab3:
			for i in all_idx:
				if i in sel_idx:
					if i>27:
						annotated_text(annotation(data[i], "" ,PALETTE[1]+"66"))
					else:
						annotated_text((data[i], "" ,PALETTE[4]+"66"))
				else:
					annotated_text(data[i])

		all_idx = range(40,51)
		sel_idx = []
		with tab4:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(52,56)
		sel_idx = []
		with tab5:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(57,62)
		sel_idx = []
		with tab6:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

if tab_idx == 4:
	data = []
	for line in file_control:
		data.append(line)

	with st.expander("**File explorer**",False):
		tab1,tab2,tab3 = st.tabs(['Pitch control',
								  'Generator and torque control',
								  'Simple variable-speed torque control'])

		all_idx = range(6,17)
		sel_idx = []
		with tab1:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(18,26)
		sel_idx = []
		with tab2:
			for i in all_idx:
				if i in sel_idx:
					annotated_text((data[i], "" ))
				else:
					annotated_text(data[i])

		all_idx = range(27,31)
		sel_idx = []
		with tab3:
			for i in all_idx:
				if i in sel_idx:
					if i>27:
						annotated_text(annotation(data[i], "" ,PALETTE[1]+"66"))
					else:
						annotated_text((data[i], "" ,PALETTE[4]+"66"))
				else:
					annotated_text(data[i])