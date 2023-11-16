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
apptitle = 'OpenFAST Course - Task 15'
icon = Image.open('feup_logo.ico')
st.set_page_config(page_title=apptitle, page_icon=icon)

st.title('Task 15 - Added mass and radiation damping evaluation')

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
			\nJust for illustration purposes, upload the added mass and radiation damping file in the OpenFAST app to analyse their evolution over frequency.
			You may find the relevant file in <b>05_HydroDyn/00_PotFiles/marin_semi.1</b>
			</div>''',unsafe_allow_html=True)

with st.expander("**Data analysis**",True):
	file = []
	file = st.file_uploader("Upload the potential theory file (.1)",accept_multiple_files=False)
	
	if not(file==None):
		try:	
			file.seek(0)	
			data = pd.read_csv(file , skiprows=36 , delimiter=r"\s+",header=None)
			data.columns = ['T','i','j','A','B']

			fig = plt.figure(figsize = (12,6))

			gs = gridspec.GridSpec(2,3,wspace=0.25,hspace=0.1)

			ax1 = plt.subplot(gs[0,0])
			ax2 = plt.subplot(gs[0,1])	
			ax3 = plt.subplot(gs[0,2])

			ax4 = plt.subplot(gs[1,0])
			ax5 = plt.subplot(gs[1,1])
			ax6 = plt.subplot(gs[1,2])	

			for i in [1,2,3]:
				f = (data['i'] == i) & (data['j'] == i)
				ax1.plot(1/data['T'][f],data['A'][f])
				ax4.plot(1/data['T'][f],data['B'][f])

				f = (data['i'] == i+3) & (data['j'] == i+3)
				ax2.plot(1/data['T'][f],data['A'][f])
				ax5.plot(1/data['T'][f],data['B'][f])

			f = (data['i'] == 1) & (data['j'] == 5)
			ax3.plot(1/data['T'][f],data['A'][f])
			ax6.plot(1/data['T'][f],data['B'][f])

			f = (data['i'] == 2) & (data['j'] == 4)
			ax3.plot(1/data['T'][f],data['A'][f])
			ax6.plot(1/data['T'][f],data['B'][f])

			ax1.set_xticklabels('')
			ax2.set_xticklabels('')
			ax3.set_xticklabels('')

			ax1.set_title('Translation DoF')
			ax2.set_title('Rotation DoF')
			ax3.set_title('Cross terms')

			ax4.set_xlabel('Frequency (Hz)')
			ax5.set_xlabel('Frequency (Hz)')
			ax6.set_xlabel('Frequency (Hz)')

			ax1.set_ylabel('Added mass')
			ax4.set_ylabel('Radiation damping')

			ax1.set_ylim(0)
			ax2.set_ylim(0)

			ax4.set_ylim(0)
			ax5.set_ylim(0)

			st.pyplot(fig)
		except:
			st.error('Something went wrong. Please check the uploaded file format.', icon="⚠️")
	
exp = st.expander('**Export report**',False)

with exp:
	report_text = st.text_input("Name")

exp_c = exp.columns([0.25,0.25,0.5])
export_as_pdf = exp_c[0].button("Generate Report")
if export_as_pdf:
	try:
		create_pdf_task15(fig,report_text,'Task 15: Hydrodynamic coefficients analysis','Task15_report',exp_c[1],exp)
	except:
		exp.error('Something went wrong. No file available for the analysis.', icon="⚠️")