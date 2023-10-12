import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from PIL import Image

# -- Set page config
apptitle = 'OpenFAST API'
icon = Image.open("logo.ico")
st.set_page_config(page_title=apptitle, page_icon=icon)

# Title the app
st.title('OpenFAST analysis')

st.markdown("""
 * Use the menu at left to select data from the different analysis possibilities
 * To tune the analysis parameters use the **Analysis** tab
""")

# -- Side bar definition
tabs = st.sidebar.tabs(["📈 File upload" , "📈 Data analysis" , "🌊 File generator"])

tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]

# -- Load data files

@st.cache_data()
def load_data(uploaded_files):
	N = len(uploaded_files)
	data = []

	for uploaded_file in uploaded_files:
		try:
			data.append(pd.read_csv(uploaded_file,skiprows=[0,1,2,3,4,5,7],delimiter="\s+",encoding_errors='replace'))
			error_check = 0
		except:
			tab1.write('Please select the file for analysis')
			error_check = 1

	return data , error_check


uploaded_files = tab1.file_uploader("Choose a file",accept_multiple_files=True)
N = len(uploaded_files)
st.write(N)
st.write(uploaded_files[1].name)

data , error_check = load_data(uploaded_files)

if error_check == 0 :
	

	tcol = tab2.selectbox('Time column', data[0].columns,index=0)
	dof = tab2.selectbox('Data column', data[0].columns,index=4)

	# -- File type definition (tab1)
	fig = plt.figure(figsize=(8,6))
	gs = gridspec.GridSpec(3,2,hspace=0.1,wspace=0.25)

	ax = plt.subplot(gs[0,0])

	for i in range(len(data)):

		t = np.array(data[i][tcol])
		y = np.array(data[i][dof])

		ax.plot(t,y)


	st.plotly_chart(fig,use_container_width=False,theme='streamlit')  
