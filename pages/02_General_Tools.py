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

from AeroPy import * 

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API - Course tasks'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

# Title the app
st.title('OpenFAST tools')
cols = st.columns([0.4,0.6])

all_mod = ['Aerodynamic properties' , 'Modes estimator' , ' Data visualisation']
sel_mod = cols[0].selectbox('Available modules', all_mod)

tab_idx = all_mod.index(sel_mod)

onshore_color = "#bada5566"
offshore_color = "#0a75ad33"

@st.cache_data()
def load_data(uploaded_files):
	N = len(uploaded_files)

	data = []
	keys = []

	error_check = 0

	for uploaded_file in uploaded_files:
		try:
			data.append(pd.read_csv(uploaded_file,skiprows=[0,1,2,3,4,5,7],delimiter="\s+",encoding_errors='replace'))
			keys.append(list(data[-1]))
			
			error_check += 1
		except:
			st.write('Please select the file for analysis')

	return data , keys , error_check 

if tab_idx == 0:
	st.write('''<div style="text-align: justify">
			Interactive estimator of the aerodynamic coefficients. 
			The lift coefficient is obtained from the 
			<a href="https://doi.org/10.21105/jose.00045">AeroPython code developed by Barba, Lorena A., Mesnard, Olivier (2019). AeroPython: classical aerodynamics of potential flow using Python. Journal of Open Source Education, 2(15), 45</a>, 
			and available in <a href="https://github.com/barbagroup/AeroPython">GitHub</a>.
			The drag coefficient is estimated based on the solution for a flat plate. In this case, the wind field is projected in the chord direction and in a direction orthogonal to it.
			For the parallel component, a combination of turbulence and laminar flow results over a flat are considered.
			For the perpendicular component, a flat plate face up to the wind with a C<sub>d</sub>=1.28 is considered.
			</div>''',unsafe_allow_html=True)
			
	##
	st.write('')

	cols = st.columns(2)
	# -- Load data files
	ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
	ref_model = cols[0].selectbox('Reference model', ref_models)
	ref_path = ref_models[ref_model]

	sel_dir = '03_AeroDyn/Airfoils'

	all_files = os.listdir('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/')
	uploaded_file = cols[1].selectbox('Available aerofoils', all_files)

	try:
		x = np.array(pd.read_csv('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + uploaded_file,delimiter='\s+',skiprows=1).iloc[:,0])
		y = np.array(pd.read_csv('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + uploaded_file,delimiter='\s+',skiprows=1).iloc[:,1])  

		cols = st.columns(4)

		n_calc = cols[0].slider('Nº panels',10,80,20,step=2)
		alpha = cols[1].number_input('Angle of attack',-10,10,0)
		u0 = cols[2].number_input('Free stream velocity',1,None,10)
		chord = cols[3].number_input('Chord',None,None,1)

		x = (x-x.min())/(x.max()-x.min())
		y = (y/(x.max()-x.min()))

		AeroCoefficients(x,y,n_calc,u0,alpha,chord)
	except:
		st.error('Something went wrong. Check the file format or try reloading it.', icon="⚠️")

if tab_idx == 1:
	##
	st.write('') 

if tab_idx == 2:
	##
	st.write('')

	# # -- Load data files
	# uploaded_files = st.file_uploader("Choose a file",accept_multiple_files=True)
	# N = len(uploaded_files)

	# data , keys , error_check  = load_data(uploaded_files)
	# cols = st.columns(4)
	# nvar = len(keys)

	# Nplots = st.number_input('Number of simulations',min_value=1,max_value=None,value=1)
	
	# cols = st.columns([0.3,0.7])

	# for i in range(Nplots):
	# 	with cols[0].expander("Simulation %d data"%(i+1),False):
			
	# 		col1, col2 = st.columns(2)
		


	# if (error_check == N) and (N>0) :
	# 	selected = []

	# 	for i in range(1,len(keys[0])):
	# 		key = keys[0][i]
	# 		selected.append(cols[int((i-1)%4)].checkbox(key))

	# 	sel_var = [keys[0][x+1] for x in range(len(keys[0])-1) if selected[x]]

	# 	n_sel_var = len(sel_var)
	# 	tcol = cols[0].selectbox('x-axis column', data[0].columns,index=0)
	# 	sep_plots = cols[1].checkbox('Separate plots',value=True)
		
	# 	# -- File type definition (tab1)
	# 	fig = plt.figure(figsize=(12,8))
	# 	gs = gridspec.GridSpec(N,n_sel_var,hspace=0.1,wspace=0.25)

	# 	if sep_plots:
	# 		for i in range(len(data)):
	# 			t = np.array(data[i][tcol])

	# 			for j in  range(n_sel_var):
	# 				ax = plt.subplot(gs[i,j])
	# 				dof = sel_var[j]
	# 				y = np.array(data[i][dof])

	# 				ax.plot(t,y)
	# 				ax.set_xticklabels('')
	# 	else:
	# 		ax = plt.subplot(gs[0,0])
		
	# 		for i in range(len(data)):
	# 			t = np.array(data[i][tcol])

	# 			for dof in sel_var:
	# 				y = np.array(data[i][dof])
	# 				ax.plot(t,y)	

	# 	st.pyplot(fig,theme='streamlit')  	

	# 	#st.plotly_chart(fig,theme='streamlit')  

