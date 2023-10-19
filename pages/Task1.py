import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
import scipy.integrate as integrate

from modes import *
from Print import * 

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST Course - Task 1'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon )

st.title('Task 1 - Compute the modal configurations for the tower and the blades')

with st.expander("**Objective**",True):
	st.write(r'''
			
			Before running any analysis, you will need to compute the structural
			properties of the wind turbine. Considering the tower stiffness and mass evolution described in the 
			ElastoDyn input files for the tower, compute and represent the tower modes and complete the input file WP_Tower.dat.
			''')
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

with st.expander("**File upload**",True):
	c1,c2 = st.columns([0.35,0.65])

	# -- File type definition (tab1)
	file_mode = ["📚 Select from database" , "⬆️ Upload file" ]
	page = c1.selectbox('Input data file', file_mode)

	if page == file_mode[1]:
		uploaded_file = c2.file_uploader("Choose a file",accept_multiple_files=False,label_visibility='hidden')

		count = 0
		if not(uploaded_file==None):
			for line in uploaded_file:
				count += 1
				if bytes("DISTRIBUTED", 'utf-8') in line:
					n1 = count
					if bytes("TOWER",'utf-8') in line:
						element_type = 'tower'
					else:
						element_type = 'blade'
				if bytes('MODE SHAPES', 'utf-8') in line:
					n2 = count

			uploaded_file.seek(0)
		#tower = pd.read_csv(uploaded_file,skiprows=np.concatenate((np.arange(n1),[n1+1])),nrows=n2-n1-3,delimiter='\s+',on_bad_lines='skip',encoding_errors='ignore')

	else:
		# -- Load data files
		ref_models = {'NREL 5MW':'01_NREL_5MW', 'WP 1.5MW':'02_WINDPACT_1500kW'}
		ref_model = c1.selectbox('Reference model', ref_models,index=1)
		ref_path = ref_models[ref_model]

		all_dir = np.sort(os.listdir('./OpenFAST_models/' + ref_path ))
		sel_dir = c1.selectbox('Available modules', all_dir)

		all_files = np.sort(os.listdir('./OpenFAST_models/' + ref_path + '/' + sel_dir))
		sel_file = c1.selectbox('Available files', all_files)

		log = open('./OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + sel_file, 'r')
		count = 0
		for line in log:
			count += 1
			if 'DISTRIBUTED' in line:
				n1 = count
				if "TOWER" in line:
					element_type = 'tower'
				else:
					element_type = 'blade'
			if 'MODE SHAPES' in line:
				n2 = count

		uploaded_file = 'OpenFAST_models/' + ref_path + '/' + sel_dir + '/' + sel_file

try:
	tower = pd.read_csv(uploaded_file,skiprows=np.concatenate((np.arange(n1),[n1+1])),nrows=n2-n1-3,delimiter='\s+',on_bad_lines='skip',encoding_errors='ignore')
	c2.write(tower)

	if element_type == 'tower':
		Mass = integrate.simpson(np.array(tower.iloc[:,1]),np.array(tower.iloc[:,0]))
	else:
		Mass = integrate.simpson(np.array(tower.iloc[:,3]),np.array(tower.iloc[:,0]))
	c1.write(r'Total mass = %.2f $\times L$ (kg)'%Mass )

	with st.expander("**Data analysis**",True):
		cols = st.columns(4)

		if element_type == 'tower':
			
			n_modes = cols[0].number_input('Number of modes',2,None,3)
			N = cols[1].number_input('Number of points',5,None,20)
			mtop = cols[2].number_input('Rotor mass',0,None,0)
			L= cols[3].number_input('Tower height',10,None,100)

			fig = TowerModesPlot(np.array(tower.iloc[:,0]),
						   		 np.array(tower.iloc[:,1]),
						   		 np.array(tower.iloc[:,2]),
						   		 N=N,n_plot=n_modes,L=L,mtop=mtop,kphi=2)
		else:

			n_modes = cols[0].number_input('Number of modes',2,None,3)
			N = cols[1].number_input('Number of points',5,None,20)
			mtop = cols[2].number_input('Tip mass',0,None,0)
			L= cols[3].number_input('Blade length',10,None,100)

			fig = BladesModesPlot(np.array(tower.iloc[:,0]),
								  np.array(tower.iloc[:,3]),
								  np.array(tower.iloc[:,4]),
								  np.array(tower.iloc[:,5]),
								  np.array(tower.iloc[:,2]),
								  N=N,n_plot=n_modes,L=L,mtop=mtop)
		st.pyplot(fig)
except:
	c2.warning('Please select or upload an adequate file with the blades or tower distributed properties.', icon="⚠️")

exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")
	st.write('''<div style="text-align: justify">
		\nNote that the modes should be properly normalised to be used as input for OpenFAST. 
		From the identified mode shapes, please indicate the scaling factor you should apply to each mode:
		</div>''',unsafe_allow_html=True)

	cols = st.columns(2)

	s1 = cols[0].number_input("Scaling factor for the 1$^{st}$ mode")
	s2 = cols[1].number_input("Scaling factor for the 2$^{nd}$ mode")
	

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")

if export_as_pdf:
    create_pdf_week1([fig],report_text,'Task 1: Modal configurations','Task1_report',exp_c[1],exp,s1,s2)
